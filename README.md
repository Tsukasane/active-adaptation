# active-adaptation

## Installation

* [Isaac Sim 4.5.0]()
* [Isaac Lab](https://github.com/isaac-sim/IsaacLab)  with Isaac Lab 2.0
* [TensorDict](https://github.com/btx0424/tensordict) from GitHub source.
* [TorchRL](https://github.com/btx0424/rl) from GitHub source.

```bash
# bash environment setup
conda create -n <isaaclab> python=3.10 # or any other env name you like
conda activate <isaaclab>

cd IsaacLab
ln -s <path to isaac sim 4.5> _isaac_sim
./isaaclab.sh -c <isaaclab>
./isaaclab.sh -i none
conda activate <isaaclab>
echo $PYTHONPATH      # ensure isaac-sim related dependencies are added

cd tensordict
python setup.py develop

cd rl
python setup.py develop

cd active_adaptation
pip install -e .
```

**DO NOT** install tensordict and torchrl using `pip install`. They are under active development so the release versions on PyPi might have bugs and lack new functionalities.

## Basic Usage

For processing AMASS data or retargeting fit keypoints, please refer to [`phc/README.md`](phc/README.md).

For video estimation and teleoperation in simulation, please refer to [`metrabs/README.md`](metrabs/README.md).

Each task is specified by a yaml file under `cfg/task`, for example:

```yaml
# @package task
name: Test
task: Humanoid

defaults:
  # see https://hydra.cc/docs/advanced/overriding_packages/
  - /task/Track@_here_
  - override /task/action@action: null
  - _self_

payload: false

action:
  _target_: active_adaptation.envs.mdp.action.JointPosition
  action_scaling:
    torso_joint: 0.5
    .*hip.*: 0.5
    .*shoulder.*: 0.5
    .*knee.*: 0.5
    .*ankle_pitch.*: 0.5
    .*elbow_pitch.*: 0.5
  max_delay: 1
  alpha: [0.5, 1.0]

command:
  _target_: active_adaptation.envs.mdp.MotionLib
  motion_clip_dir: "scripts/data"
  dataset: amass
  occlusion: "amass_copycat_occlusion_v3.pkl"
  mode: train
  eval_id: null
  
observation:
  robot:
    root_quat_w:
    root_angvel_b:        {noise_std: 0.05}
    projected_gravity_b:  {noise_std: 0.01}
    joint_pos:            {noise_std: 0.05}
    joint_vel:            {noise_std: 0.2}
    body_pos:             {body_names: [left_hip_pitch_link, left_knee_link, left_ankle_roll_link, 
                                        right_hip_pitch_link, right_knee_link, right_ankle_roll_link,
                                        left_shoulder_roll_link, left_elbow_pitch_link, left_zero_link, 
                                        right_shoulder_roll_link, right_elbow_pitch_link, right_zero_link], 
                                        yaw_only: false}
    prev_actions:         {steps: 1}
  ref_motion_:
    ref_orientation:      {steps: 5}
    ref_height:           {steps: 5}
    ref_qpos:             { joint_names: [.*hip_pitch.*, torso_joint, .*hip_roll.*,
                                          .*shoulder_pitch.*, .*hip_yaw.*, .*shoulder_roll.*,
                                          .*knee.*, .*shoulder_yaw.*,
                                          .*ankle_pitch.*, .*elbow_pitch.*],
                            steps: 5}
    ref_keypoints:        {steps: 5}
    ref_keypoints_gap:    { body_names: [left_hip_pitch_link, left_knee_link, left_ankle_roll_link, 
                                        right_hip_pitch_link, right_knee_link, right_ankle_roll_link,
                                        left_shoulder_roll_link, left_elbow_pitch_link, left_zero_link, 
                                        right_shoulder_roll_link, right_elbow_pitch_link, right_zero_link],
                            steps: 5}
    ref_trans_gap:        {steps: 5}
  priv:
    root_height:      {}
    root_linvel_b:    {}
    body_vel:         {body_names: [left_hip_pitch_link, left_knee_link, left_ankle_roll_link, 
                                        right_hip_pitch_link, right_knee_link, right_ankle_roll_link,
                                        left_shoulder_roll_link, left_elbow_pitch_link, left_zero_link, 
                                        right_shoulder_roll_link, right_elbow_pitch_link, right_zero_link], 
                                        yaw_only: false}
    joint_forces:     {}

reward:
  loco:
    tracking_root_trans:    {weight: 4., enabled: true, sigma: 0.16}
    tracking_root_rot:      {weight: 2., enabled: true, sigma: 0.16}
    tracking_qpos:          {weight: 2., enabled: true, sigma: 0.16,
                              joint_names: [.*hip_pitch.*, torso_joint, .*hip_roll.*,
                                            .*shoulder_pitch.*, .*hip_yaw.*, .*shoulder_roll.*,
                                            .*knee.*, .*shoulder_yaw.*,
                                            .*ankle_pitch.*, .*elbow_pitch.*]}
    tracking_keypoints:     {weight: 6., enabled: true, sigma: 0.16,
                              body_names: [left_hip_pitch_link, left_knee_link,
                                        right_hip_pitch_link, right_knee_link,
                                        left_shoulder_roll_link, left_elbow_pitch_link, 
                                        right_shoulder_roll_link, right_elbow_pitch_link]}
    tracking_eff:           {weight: 7.5, enabled: true, sigma: 0.16,
                              body_names: [left_ankle_roll_link, left_zero_link,
                                        right_ankle_roll_link, right_zero_link]}

    feet_orientation:   {weight: 0.5, enabled: true, feet_names: .*ankle_roll_link, body_name: pelvis}
    feet_slip:          {weight: 2., enabled: true, body_names: .*ankle_roll_link, sigma: 0.16}
    max_feet_height:    {weight: 1.5, enabled: true, body_names: .*ankle_roll_link, target_height: 0.15}

    joint_acc_l2:       {weight: 1.0e-8, enabled: true, joint_names: [torso_joint, .*hip.*, .*knee.*,
                                                                  .*shoulder.*, .*elbow.*, .*ankle.*]}
    joint_vel_l2:       {weight: 0.002, enabled: true, joint_names: [torso_joint, .*hip.*, .*knee.*,
                                                                  .*shoulder.*, .*elbow.*, .*ankle.*]}
    joint_torques_l2:   {weight: 1.0e-5, enabled: true, joint_names: [torso_joint, .*hip.*, .*knee.*,
                                                                  .*shoulder.*, .*elbow.*, .*ankle.*]}
    action_rate_l2:     {weight: 0.02, enabled: true}
    action_rate2_l2:    {weight: 0.01, enabled: true}

randomization:
  push:     {body_names: ["torso_link", "pelvis"], force_range: [0.0, 0.1]}
  perturb_body_mass:
    .*: [0.9, 1.1]
  perturb_body_materials:
    body_names: ".*ankle_roll_link"
    static_friction_range: [0.6, 4.0]
    dynamic_friction_range: [0.6, 4.0]
    restitution_range: [0.0, 0.2]
  motor_params:
    actuator_name:    legs
    stiffness_range:  [0.7, 1.0]
    damping_range:    [0.7, 1.0]
  reset_joint_states_uniform:
    pos_ranges:
      .*: [-0.1, 0.1]
    rel: true

termination:
  dummy: {}
  root_deviation: {max_distance: 0.5}
  root_rot_deviation: {max_theta: 30}
  track_kp_error: {max_distance: 0.4, 
                  body_names: [left_zero_link, right_zero_link,
                                left_ankle_roll_link, right_ankle_roll_link]}

```

Observations are grouped by keys and the observation of the same group is concatenated.

Rewards are grouped by keys and the rewards of the same group is summed up, excluding those marked with `enabled=false`. However, rewards with `enabled=false` will still be computed and logged as metrics for debugging purposes.


### Training

Examples:

```bash
python test_env.py task=motion algo=ppo
python test_env.py task=motion algo=ppo task.num_envs=4 total_frames=250000000 headless=false wandb.mode=disabled
```

### Evaluation and Visualization

Examples:

```bash
python eval.py \
    eval_render=true \                      # whether to record video
    headless=false \                        # whether to run in headless mode (no GUI)
    algo=ppo \                              # algorithm to use
    task=motion \                           # task to use
    task.num_envs=4 \                       # number of environments to run
    task.command.dataset=sfu \              # dataset to use
    task.command.eval_id=0 \                # evaluation motion id
    checkpoint_path=<path to checkpoint>    # path to checkpoint

python play.py \
    algo=ppo \                              # algorithm to use
    task=motion \                           # task to use
    task.num_envs=4 \                       # number of environments to run
    task.command.dataset=sfu \              # dataset to use
    checkpoint_path=<path to checkpoint>    # path to checkpoint
```

## Adding New Tasks

All of observation, reward, termination, randomization and command follow a similar protocol, for example:

```python

class Observation:
    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def update(self):
        """Called at each step **after** simulation"""

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""

```

Inherit from and extend the classes in `envs/mdp/xxx.py` to implement environment logic.
