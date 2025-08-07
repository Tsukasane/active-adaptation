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
import hydra

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
class TeacherConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo.PPOPolicy"
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.0005
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    vecnorm: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_PRIV_KEY])

    checkpoint_path: Union[str, None] = None
    # checkpoint_path: Union[str, None] = "/home/ubuntu/Desktop/g1-sim2real/scripts/outputs/2025-08-01/15-07-36-more_ref_horizon-ppo/wandb/latest-run/files/checkpoint_final.pt"
    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_PRIV_KEY])

@dataclass
class PPOStudentConfig:
    _target_: str = "active_adaptation.learning.ppo_student.ppo_student.PPOStudentPolicy"
    name: str = "ppo_student"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.0005
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    vecnorm: List[str] = field(default_factory=lambda: ["robot_h"])
    kl_coef: float = 0.01
    short_history: int = 5

    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = field(default_factory=lambda: ["robot_h", OBS_PRIV_KEY])

    teacher: TeacherConfig = field(default_factory=TeacherConfig)

cs = ConfigStore.instance()
cs.store("ppo_student", node=PPOStudentConfig, group="algo")

class PPOStudentPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOStudentConfig, 
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
        # print(fake_input)
        
        def make_actor(out_key: str):
            modules = [
                CatTensors(["robot_h", "stu_ref_motion_"], "a_in"),
                TensorDictModule(make_mlp([2048, 1024, 512]), ["a_in"], [out_key])
            ]
            return modules
        
        def make_critic(out_key: str):
            modules = [
                CatTensors(["robot_h", OBS_REF_KEY, OBS_PRIV_KEY], "c_in"),
                TensorDictModule(make_mlp([2048, 1024, 512]), ["c_in"], [out_key])
            ]
            return modules

        _actor = nn.Sequential(make_mlp([256, 128]), Actor(self.action_dim))
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

        if self.cfg.teacher.checkpoint_path is not None:
            print(colored(f"[Info]: create teacher {self.cfg.teacher._target_}", "green"))
            TeacherCls = hydra.utils.get_class(self.cfg.teacher._target_)
            self.teacher = TeacherCls(self.cfg.teacher, observation_spec, action_spec, reward_spec, device)
            self.teacher.load_state_dict(torch.load(self.cfg.teacher.checkpoint_path)["policy"])
            self.teacher_actor = self.teacher.actor.eval()
            self.teacher_vecnorm = self.teacher.vecnorm.freeze()
            self.teacher_vecnorm = self.teacher_vecnorm.to_observation_norm()

    def count_parameters(self):
        num_actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        num_critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        actor_params_m = num_actor_params / 1e6
        critic_params_m = num_critic_params / 1e6
        print(f'Number of actor parameters: {actor_params_m:.2f}M')
        print(f'Number of critic parameters: {critic_params_m:.2f}M')
    
    def get_rollout_policy(self, mode: str="train"):
        def cache_raw_keys(tensordict: TensorDictBase):
            for k in [OBS_KEY, OBS_PRIV_KEY]:
                tensordict.set(f"{k}_raw", tensordict.get(k).clone())
            return tensordict
        
        if mode == "train":
            policy = TensorDictSequential(
                cache_raw_keys,
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

        # teacher imitation loss
        td_teacher = tensordict.clone()
        for k in [OBS_KEY, OBS_PRIV_KEY]:
            td_teacher.set(k, tensordict.get(f"{k}_raw"))
        self.teacher_vecnorm(td_teacher)
        dist_teacher = self.teacher_actor.get_dist(td_teacher)
        kl_div = D.kl_divergence(dist, dist_teacher).mean()
        kl_loss = kl_div * self.cfg.kl_coef

        loss = policy_loss + entropy_loss + value_loss + kl_loss
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
            "actor/kl_loss": kl_loss,
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