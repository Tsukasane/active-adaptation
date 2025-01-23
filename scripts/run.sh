# Reset & Distill
python rollout.py task=MotionTracking/walk algo=ppo checkpoint_path=ckpt_path
python distill.py epoch=100 checkpoint_path=ckpt_path
python eval.py task.num_envs=4096 algo=bc task=MotionTracking/walk checkpoint_path=ckpt_path 

# # Continual Reinforcement Learning with priv. info
python test_env.py task=MotionTracking/walk algo=ppo_im

# Single Policy w. VAE Module, estimating privileged information
python test_env.py task=DynaEst/walk algo=ppo_adv
python test_env.py task=DynaEst/backwalk algo=ppo_vim checkpoint_path=ckpt_path

# Ablation w.o. privileged information estimation
python test_env.py task=DynaEst/walk algo=ppo_v
python test_env.py task=DynaEst/backwalk algo=ppo_v_ab checkpoint_path=ckpt_path

# Abalation w.o. VAE Module
python test_env.py task=DynaEst/walk algo=ppo_ad
python test_env.py task=DynaEst/backwalk algo=ppo_ad_ab checkpoint_path=ckpt_path

# Evaluation
python eval.py task.num_envs=4096 algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path                               # for metric
python eval.py task.num_envs=1 headless=false eval_render=true algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path  # for visualization