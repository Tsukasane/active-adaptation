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

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm, TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from .common import *
import einops

from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)

torch.set_float32_matmul_precision('high')


class GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        allow_none: bool = False,
        burn_in: bool = False
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.allow_none = allow_none
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if x.ndim == 2: # single step

            N = x.shape[0]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            # assert (hx[is_init.squeeze()] == 0.).all()
            hx = self.gru(x, hx)
            output = self.ln(hx)
            return output, hx

        elif x.ndim == 3: # multi-step

            N, T = x.shape[:2]
            if hx is None and self.allow_none:
                hx = torch.zeros(N, self.gru.hidden_size, device=x.device)
            else:
                hx = hx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx = self.gru(x_t, hx * reset_t)
                if self.burn_in and i < T // 4:
                    hx = hx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.ln(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)


class GRUModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = make_mlp([512, 256])
        self.gru = GRU(256, hidden_size=128, allow_none=False)
        self.out = nn.LazyLinear(dim)
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, is_init, hx)
        x = self.out(x)
        return x, hx.contiguous()

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_adapt.PPOAdaptPolicy"
    name: str = "ppo_adapt"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    vecnorm: Union[str, None] = None

    total_frames: int = 300_000_000

    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_HIST_KEY, OBS_REF_KEY, 
                                                        OBS_PRIV_KEY, "priv_", 
                                                        "aux_target_"])

cs = ConfigStore.instance()
cs.store("ppo_adapt", node=PPOConfig, group="algo")

class PPOAdaptPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
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
        
        self.context_dim = observation_spec["aux_target_"].shape[-1]

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        self.observation_spec = observation_spec
        fake_input = observation_spec.zero()
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["context_adapt_hx"] = torch.zeros(fake_input.shape[0], 128)
        print(fake_input)

        def make_adapt():
            module = TensorDictSequential(
                CatTensors([OBS_KEY, OBS_REF_KEY], "gru_in"),
                TensorDictModule(
                        GRUModule(self.context_dim),
                        ["gru_in", "is_init", "context_adapt_hx"], 
                        ["context_adapt", ("next", "context_adapt_hx")]
                    )
            ).to(self.device)
            return module
             
        def make_actor(out_key: str):
            modules = [
                    CatTensors([OBS_KEY, OBS_HIST_KEY, OBS_REF_KEY, "context_adapt"], "a_in"),
                    TensorDictModule(make_mlp([1024, 512]), ["a_in"], out_key),
                ]
            return modules
        
        def make_critics(out_key: str):
            modules = [
                    CatTensors([OBS_KEY, OBS_HIST_KEY, OBS_REF_KEY, OBS_PRIV_KEY, "priv_", "aux_target_"], "c_in"),
                    TensorDictModule(make_mlp([1024, 512]), ["c_in"], out_key),
                ]
            return modules
        
        self.adapt = make_adapt()

        _actor = nn.Sequential(make_mlp([256]), Actor(self.action_dim))
        actor_module = TensorDictSequential(
            *make_actor("_actor_feature"),
            TensorDictModule(_actor, ["_actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(make_mlp([256]), nn.Linear(256, 1))
        self.critic = TensorDictSequential(
            *make_critics("_critic_feature"),
            TensorDictModule(_critic, ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.adapt(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        self.vecnorm: VecNorm = VecNorm([OBS_KEY, OBS_HIST_KEY, OBS_PRIV_KEY], decay=0.9999)

        self.count_parameters()

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt.parameters()},
            ]
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.adapt.apply(init_)
        self.actor.apply(init_)
        self.critic.apply(init_)

        self.step_cnt = 0
        frames_per_batch = self.cfg.train_every * self.observation_spec.shape[0]
        total_frames = self.cfg.total_frames // frames_per_batch * frames_per_batch
        self.total_iters = total_frames // frames_per_batch

    def count_parameters(self):
        num_actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        num_critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        num_adapt_params = sum(p.numel() for p in self.adapt.parameters() if p.requires_grad)
        actor_params_m = num_actor_params / 1e6
        critic_params_m = num_critic_params / 1e6
        adapt_params_m = num_adapt_params / 1e6
        print(f'Number of actor parameters: {actor_params_m:.2f}M')
        print(f'Number of critic parameters: {critic_params_m:.2f}M')
        print(f'Number of adapt parameters: {adapt_params_m:.2f}M')

    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = UnboundedContinuousTensorSpec((num_envs, 128), device=self.device)
        return TensorDictPrimer({"context_adapt_hx": spec}, reset_key="done")
    
    def get_rollout_policy(self, mode: str="train"):
        if mode == "train":
            policy = TensorDictSequential(
                self.vecnorm,
                self.adapt,
                self.actor,
            )
        else:
            policy = TensorDictSequential(
                self.vecnorm.to_observation_norm(),
                self.adapt,
                self.actor,
            )
        return policy

    # @torch.compile
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()
        tensordict_adapt = tensordict.clone()
        
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        def train_actor_critic():
            infos = []
            for epoch in range(self.cfg.ppo_epochs):
                batch = make_batch(tensordict, self.cfg.num_minibatches)
                for minibatch in batch:
                    infos.append(TensorDict({
                        **self._update(minibatch),
                    }, []))
            return {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}

        def train_adaptation():
            infos = []
            for epoch in range(self.cfg.ppo_epochs):
                batch = make_batch(tensordict_adapt, 8, self.cfg.train_every)
                for minibatch in batch:
                    infos.append(TensorDict({
                        **self._update_adaptation(minibatch),
                    }, []))
            return {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        
        infos = {}
        infos.update(train_actor_critic())
        infos.update(train_adaptation())
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
        return {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }
    
    def _update_adaptation(self, tensordict: TensorDict):
        tdict_local = tensordict.copy()
        self.adapt(tdict_local)
        context_adapt = tdict_local["context_adapt"].reshape(-1, self.context_dim)
        context_target = tdict_local["aux_target_"].reshape(-1, self.context_dim)
        loss = F.mse_loss(context_adapt, context_target)

        self.opt_adapt.zero_grad()
        loss.backward()
        self.opt_adapt.step()

        root_vel_error = (context_adapt[:, :3] - context_target[:, :3]).norm(dim=-1).mean()
        body_pos_error = (context_adapt[:, 3:] - context_target[:, 3:]).norm(dim=-1).mean()

        return {
            "adapt/loss": loss,
            "adapt/root_vel_error": root_vel_error,
            "adapt/body_pos_error": body_pos_error,
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
    
def sample(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std