from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.envs.transforms import VecNorm
from torchrl.data import UnboundedContinuousTensorSpec
from collections import OrderedDict

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, List

from .ppo.common import *
from .modules.distributions import IndependentNormal

@dataclass
class BCConfig:
    _target_: str = "active_adaptation.learning.bc.BCPolicy"
    name: str = "bc"
    epoch: int = 50
    batch_size: int = 256
    lr: float = 1e-4

    in_keys: List[str] = field(default_factory=lambda: [OBS_KEY, OBS_HIST_KEY, OBS_REF_KEY])

cs = ConfigStore.instance()
cs.store("bc", node=BCConfig, group="algo")

class BCPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: CompositeSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_dim = action_spec.shape[-1]
        self.device = device

        self.epoch = cfg.epoch
        self.batch_size = cfg.batch_size

        fake_input = observation_spec.zero()

        def make_encoder(out_key: str):
            modules = [
                TensorDictModule(make_mlp([256]), [OBS_KEY], ["_robot"]),
                TensorDictModule(make_mlp([256]), [OBS_HIST_KEY], ["_hist"]),
                TensorDictModule(make_mlp([256]), [OBS_REF_KEY], ["_ref_motion_"]),
                CatTensors(["_robot", "_hist", "_ref_motion_"], out_key),
            ]
            return modules
        
        _actor = nn.Sequential(make_mlp([256, 128]), Actor(self.action_dim))
        actor_module = TensorDictSequential(
            *make_encoder("_actor_feature"),
            TensorDictModule(_actor, ["_actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.vecnorm: VecNorm = VecNorm([OBS_KEY, OBS_HIST_KEY], decay=0.9999)

        self.actor(fake_input)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

    def count_parameters(self):
        num_actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        actor_params_m = num_actor_params / 1e6
        print(f'Number of actor parameters: {actor_params_m:.2f}M')
    
    def forward(self, tensordict: TensorDictBase):
        tensordict = self.vecnorm(tensordict)
        return self.actor(tensordict)
    
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
    
    def kl_loss_a(self, tensordict: TensorDictBase):
        loc, scale = self(tensordict)["loc"], self(tensordict)["scale"]
        gt_loc, gt_scale = tensordict["gt_loc"], tensordict["gt_scale"]
        kl = D.kl_divergence(D.Normal(gt_loc, gt_scale), D.Normal(loc, scale)).mean()
        return kl
    
    def kl_loss_b(self, tensordict: TensorDictBase):
        loc, scale = self(tensordict)["loc"], self(tensordict)["scale"]
        gt_loc, gt_scale = tensordict["gt_loc"], tensordict["gt_scale"]
        kl = D.kl_divergence(D.Normal(loc, scale), D.Normal(gt_loc, gt_scale)).mean()
        return kl

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