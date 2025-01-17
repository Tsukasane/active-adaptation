# # Amp with privileged information
python test_env.py task=MotionTracking/amp algo=amp total_frames=600000000

# # Reset & Distill
python rollout.py task=MotionTracking/walk task.num_envs=4 algo=ppo checkpoint_path=ckpt_path
python distill.py checkpoint_path=ckpt_path

# # Continual Reinforcement Learning
python test_env.py task=MotionTracking/walk algo=ppo_im
python test_env.py task=MotionTracking/backwalk algo=ppo_im
python test_env.py task=MotionTracking/joint_walk algo=ppo_im
python test_env.py task=MotionTracking/mickey_walk algo=ppo_im
python test_env.py task=MotionTracking/cat_walk algo=ppo_im
python test_env.py task=MotionTracking/angry_walk algo=ppo_im
python test_env.py task=MotionTracking/stealthy_walk algo=ppo_im
python test_env.py task=MotionTracking/jog algo=ppo_im
python test_env.py task=MotionTracking/trot algo=ppo_im
python test_env.py task=MotionTracking/boxing algo=ppo_im
python test_env.py task=MotionTracking/indian algo=ppo_im
python test_env.py task=MotionTracking/chacha algo=ppo_im
python test_env.py task=MotionTracking/lambada algo=ppo_im

# Single Policy w. Dynamic Module, wo. privileged information
python test_env.py task=DynaEst/walk algo=ppo_adv total_frames=400000000 task.name=walk-hq
python test_env.py task=DynaEst/backwalk algo=ppo_adv total_frames=400000000 task.name=backwalk-hq
python test_env.py task=DynaEst/joint_walk algo=ppo_adv total_frames=400000000 task.name=joint_walk-hq
python test_env.py task=DynaEst/mickey_walk algo=ppo_adv total_frames=400000000 task.name=mickey_walk-hq
python test_env.py task=DynaEst/cat_walk algo=ppo_adv total_frames=400000000 task.name=cat_walk-hq
python test_env.py task=DynaEst/angry_walk algo=ppo_adv total_frames=400000000 task.name=angry_walk-hq
python test_env.py task=DynaEst/stealthy_walk algo=ppo_adv total_frames=400000000 task.name=stealthy_walk-hq
python test_env.py task=DynaEst/jog algo=ppo_adv total_frames=400000000 task.name=jog-hq
python test_env.py task=DynaEst/trot algo=ppo_adv total_frames=400000000 task.name=trot-hq
python test_env.py task=DynaEst/boxing algo=ppo_adv total_frames=400000000 task.name=boxing-hq
python test_env.py task=DynaEst/indian algo=ppo_adv total_frames=400000000 task.name=indian-hq
python test_env.py task=DynaEst/chacha algo=ppo_adv total_frames=400000000 task.name=chacha-hq
python test_env.py task=DynaEst/lambada algo=ppo_adv total_frames=400000000 task.name=lambada-hq

# Evaluation
python eval.py task.num_envs=4096 algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path                               # for metric
python eval.py task.num_envs=1 headless=false eval_render=true algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path  # for visualization