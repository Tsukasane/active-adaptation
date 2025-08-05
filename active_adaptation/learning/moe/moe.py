# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import functools

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..modules.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.common import *

torch.set_float32_matmul_precision('high')

@dataclass
class MoeConfig:
    _target_: str = "active_adaptation.learning.moe.moe.MoePolicy"
    name: str = "moe"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.0005
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    vecnorm: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_PRIV_KEY])

    n_experts: int = 4

    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_PRIV_KEY])

cs = ConfigStore.instance()
cs.store("moe", node=MoeConfig, group="algo")

class MoECore(nn.Module):
    def __init__(self, n_experts: int, action_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Sequential(make_mlp([512, 256, 128]), nn.Linear(128, n_experts))
        self.experts = nn.ModuleList([nn.Sequential(make_mlp([512, 256, 128]), Actor(action_dim)) for _ in range(n_experts)])
    
    @staticmethod
    def moment_match_mog(g_logits: torch.Tensor,
                     locs: torch.Tensor,
                     scales: torch.Tensor,
                     eps: float = 1e-6):
        """
        g_logits: [B, N]
        locs:     [B, N, A]
        scales:   [B, N, A]   (std)
        returns:  mu [B, A], std [B, A]
        """
        probs = g_logits.unsqueeze(-1)                                          # [B, N, 1]
        mu = (probs * locs).sum(dim=1)                                          # [B, A]
        # Var[X] = E[Var] + Var[E]
        var = (probs * (scales**2 + (locs - mu.unsqueeze(1))**2)).sum(dim=1)    # [B, A]
        std = var.clamp_min(eps).sqrt()
        return mu, std

    def forward(self, x: torch.Tensor):
        logits = self.gate(x).softmax(dim=-1)
        locs, scales = (torch.stack(t, dim=1) for t in zip(*[expert(x) for expert in self.experts]))
        mu, std = self.moment_match_mog(logits, locs, scales)
        return mu, std, logits

class MoePolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: MoeConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        print(fake_input)
        
        def make_actor(out_key: str):
            modules = [
                CatTensors([OBS_KEY, OBS_REF_KEY, OBS_PRIV_KEY], "a_in"),
                TensorDictModule(make_mlp([2048, 1024]), ["a_in"], [out_key])
            ]
            return modules
        
        def make_critic(out_key: str):
            modules = [
                CatTensors([OBS_KEY, OBS_REF_KEY, OBS_PRIV_KEY], "c_in"),
                TensorDictModule(make_mlp([2048, 1024, 512]), ["c_in"], [out_key])
            ]
            return modules

        actor_module = TensorDictSequential(
            *make_actor("_actor_feature"),
            TensorDictModule(MoECore(self.cfg.n_experts, self.action_dim), ["_actor_feature"], ["loc", "scale", "g_logits"])
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(make_mlp([256, 128]), nn.Linear(128, 1))
        self.critic = TensorDictSequential(
            *make_critic("_critic_feature"),
            TensorDictModule(_critic, ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        from termcolor import colored
        print(colored(f"[Info]: create VecNorm for keys: {self.cfg.vecnorm}", "green"))
        self.vecnorm: VecNorm = VecNorm(self.cfg.vecnorm, decay=0.9999)

        self.count_parameters()

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, -1.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

    def count_parameters(self):
        num_actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        num_critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        actor_params_m = num_actor_params / 1e6
        critic_params_m = num_critic_params / 1e6
        print(f'Number of actor parameters: {actor_params_m:.2f}M')
        print(f'Number of critic parameters: {critic_params_m:.2f}M')
    
    def get_rollout_policy(self, mode: str="train"):
        if mode == "train":
            policy = TensorDictSequential(
                self.vecnorm,
                self.actor,
            )
        else:
            policy = TensorDictSequential(
                self.vecnorm.to_observation_norm(),
                self.actor,
            )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self._update(minibatch), []))
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TensorDictModule, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        with tensordict.view(-1) as tensordict_flat:
            critic(tensordict_flat)
            self.vecnorm.freeze()
            self.vecnorm(tensordict_flat["next"])
            critic(tensordict_flat["next"])
            self.vecnorm.unfreeze()

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    # @torch.compile
    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()

        gate_probs = tensordict["g_logits"]
        gate_entropy = -(gate_probs * gate_probs.clamp_min(1e-8).log()).sum(-1).mean()
        return {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "actor/gate_entropy": gate_entropy,
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["vecnorm"] = self.vecnorm.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)