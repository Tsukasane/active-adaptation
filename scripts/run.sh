# # Single Oracle Policy
# python test_env.py task=MotionTracking/walk
# python test_env.py task=MotionTracking/backwalk
# python test_env.py task=MotionTracking/joint_walk
# python test_env.py task=MotionTracking/mickey_walk
# python test_env.py task=MotionTracking/cat_walk
# python test_env.py task=MotionTracking/angry_walk
# python test_env.py task=MotionTracking/stealthy_walk      
# python test_env.py task=MotionTracking/jog 
# python test_env.py task=MotionTracking/trot
# python test_env.py task=MotionTracking/boxing
# python test_env.py task=MotionTracking/indian
# python test_env.py task=MotionTracking/chacha
# python test_env.py task=MotionTracking/lambada

# # Amp with privileged information
# python test_env.py task=MotionTracking/amp algo=amp total_frames=600000000

# # Reset & Distill
# python rollout.py task=MotionTracking/walk task.num_envs=4 algo=ppo checkpoint_path=ckpt_path
# python distill.py checkpoint_path=ckpt_path

# # Continual Reinforcement Learning
# python test_env.py task=MotionTracking/walk algo=ppo_im
# python test_env.py task=MotionTracking/backwalk algo=ppo_im
# python test_env.py task=MotionTracking/joint_walk algo=ppo_im
# python test_env.py task=MotionTracking/mickey_walk algo=ppo_im
# python test_env.py task=MotionTracking/cat_walk algo=ppo_im
# python test_env.py task=MotionTracking/angry_walk algo=ppo_im
# python test_env.py task=MotionTracking/stealthy_walk algo=ppo_im
# python test_env.py task=MotionTracking/jog algo=ppo_im
# python test_env.py task=MotionTracking/trot algo=ppo_im
# python test_env.py task=MotionTracking/boxing algo=ppo_im
# python test_env.py task=MotionTracking/indian algo=ppo_im
# python test_env.py task=MotionTracking/chacha algo=ppo_im
# python test_env.py task=MotionTracking/lambada algo=ppo_im

# Single Policy w. Dynamic Module, wo. privileged information
python test_env.py task=DynaEst/walk algo=ppo_ad
python test_env.py task=DynaEst/backwalk algo=ppo_ad
python test_env.py task=DynaEst/joint_walk algo=ppo_ad
python test_env.py task=DynaEst/mickey_walk algo=ppo_ad
python test_env.py task=DynaEst/cat_walk algo=ppo_ad
python test_env.py task=DynaEst/angry_walk algo=ppo_ad
python test_env.py task=DynaEst/stealthy_walk algo=ppo_ad
python test_env.py task=DynaEst/jog algo=ppo_ad
python test_env.py task=DynaEst/trot algo=ppo_ad
python test_env.py task=DynaEst/boxing algo=ppo_ad
python test_env.py task=DynaEst/indian algo=ppo_ad
python test_env.py task=DynaEst/chacha algo=ppo_ad
python test_env.py task=DynaEst/lambada algo=ppo_ad

# Evaluation
# python eval.py task.num_envs=4096 algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path                               # for metric
# python eval.py task.num_envs=1 headless=false eval_render=true algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path  # for visualization