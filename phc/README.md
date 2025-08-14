Credit to [PHC](https://github.com/ZhengyiLuo/PHC)

Here we use `uv` to manage the environment (python 3.8 for PHC, python 3.10 for loco-mujoco and jax).

```bash
uv venv --python 3.8 .venv-py38
source .venv-py38/bin/activate
uv pip install -U pip setuptools wheel
uv pip install numpy==1.23.5 scipy open3d
uv pip install torch torchvision --torch-backend=cu118
uv pip install --no-build-isolation "chumpy==0.70"
uv pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master
```

Modifications:
- set OMP_NUM_THREADS to 1 for multiprocessing
- simplified the config for retargeting process

For fitting motion data from AMASS:

```bash
uv run scripts/1-fit_smpl_shape.py
uv run scripts/2-fit_smpl_motion.py +amass_root=./data/AMASS
uv run scripts/3-vis_q_mj.py +motion_file=./data/g1/v1/amass_all.pkl
```

For 27dof model (the current tracking humanoid model), we can use the following command:
```bash
uv run scripts/1-fit_smpl_shape.py --config-nameunitree_g1_27dof_fitting
uv run scripts/2-fit_smpl_motion.py --config-name unitree_g1_27dof_fitting +amass_root=./data/AMASS
uv run scripts/3-vis_q_mj.py --config-name unitree_g1_27dof_fitting +motion_file=./data/g1_27dof/v1/amass_all.pkl
```

For motion files or HOI sequences that have keypoints, we can directly solve the dof values from the keypoints instead of forward through smpl model:

```bash
uv run scripts/0-fit_keypoints.py +motion_file=<path_to_motion_file>
uv run scripts/3-vis_q_mj.py +motion_file=<path_to_motion_file>
```

Feel free to modify this template file and config file for specific needs.

Beyond above, we also integrate the LAFAN1 dataset credit to [Loco-mujoco](https://github.com/robfiras/loco-mujoco) and its collected data.

For installing the loco-mujoco support:
```bash
# loco-mujoco requires python >= 3.10
uv venv --python 3.10 .venv-py310
source .venv-py310/bin/activate

cd loco-mujoco
uv pip install -e .
uv pip install "jax[cuda12]" joblib
```

For playing the trajectory:
```bash
uv run scripts/m-play.py
```

For collecting trajectories into pkl file for training:
```bash
uv run scripts/m-collect.py
```
