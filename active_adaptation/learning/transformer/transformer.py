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
import numpy as np

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
class TransformerConfig:
    _target_: str = "active_adaptation.learning.transformer.transformer.TransformerPolicy"
    name: str = "transformer"
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
    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_PRIV_KEY])

    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 4
    ff_size: int = 512
    
    tokens_per_field: dict[str, int] = field(default_factory=lambda: {
        OBS_KEY: 1,
        OBS_REF_KEY: 5,
        OBS_PRIV_KEY: 1,
    })

cs = ConfigStore.instance()
cs.store("transformer", node=TransformerConfig, group="algo")


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.shape[1], :x.shape[2]]
        return x

class TransformerCore(nn.Module):
    def __init__(self,  latent_dim: int, 
                        n_heads: int, 
                        n_layers: int, 
                        ff_size: int, 
                        dropout: float = 0.0, 
                        activation = F.relu):
        super().__init__()
        self.pos_encoder = PositionalEmbedding(d_model=latent_dim)
        enc_layer = nn.TransformerEncoderLayer( d_model=latent_dim, 
                                                        nhead=n_heads,
                                                        dim_feedforward=ff_size,
                                                        dropout=dropout,
                                                        activation=activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor):
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2).contiguous()             # [seq_len, batch_size, d_model]
        x = self.encoder(x)[0]                          # [batch_size, d_model]
        return x

class MultiObsTokenizer(nn.Module):
    def __init__(self,  fields: List[tuple[str, int, int]],
                        d_model: int):
        super().__init__()
        self.fields = fields
        self.d_model = d_model

        self.n_tokens_per_field = [n_tok for (key, in_dim, n_tok) in fields]
        self.proj = nn.ModuleList(
            [nn.Linear(in_dim, d_model * n_tok) for (key, in_dim, n_tok) in fields]
        )

    def forward(self, *xs: torch.Tensor):
        """
        Parameters
        ----------
        *xs : sequence of Tensors
            One tensor per field, same order as `fields`.
            Expected shape per tensor: [batch_size, in_dim].

        Returns
        -------
        tokens : Tensor
            Concatenated tokens of shape [batch_size, seq_len, d_model],
            where seq_len = sum(n_tokens per field).
        """
        assert len(xs) == len(self.fields), f"Expected {len(self.fields)} inputs, got {len(xs)}"
        B = xs[0].shape[0]
        tokens = []
        for x, proj, n_tok in zip(xs, self.proj, self.n_tokens_per_field):
            x = proj(x).view(B, -1, self.d_model)
            tokens.append(x)
        return torch.cat(tokens, dim=1) # [B, seq_len, d_model]

class TransformerPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: TransformerConfig, 
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
        
        def make_critic(out_key: str):
            modules = [
                CatTensors([OBS_KEY, OBS_REF_KEY, OBS_PRIV_KEY], "c_in"),
                TensorDictModule(make_mlp([2048, 1024, 512]), ["c_in"], [out_key])
            ]
            return modules

        fields = []
        for key, n_tok in self.cfg.tokens_per_field.items():
            fields.append((key, observation_spec[key].shape[-1], n_tok))
        self.tokenizer = MultiObsTokenizer(
            fields=fields,
            d_model=self.cfg.d_model
        )

        self.transformer_core = TransformerCore(
            latent_dim=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            ff_size=self.cfg.ff_size,
        )

        _actor = nn.Sequential(make_mlp([256, 128]), Actor(self.action_dim))
        actor_module = TensorDictSequential(
            TensorDictModule(self.tokenizer, [OBS_KEY, OBS_REF_KEY, OBS_PRIV_KEY], ["tokens"]),
            TensorDictModule(self.transformer_core, ["tokens"], ["_actor_feature"]),
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
        print(fake_input)
        exit()

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