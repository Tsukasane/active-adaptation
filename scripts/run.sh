# Single Oracle Policy
python test_env.py task=MotionTracking/walk
python test_env.py task=MotionTracking/backwalk
python test_env.py task=MotionTracking/joint_walk
python test_env.py task=MotionTracking/mickey_walk
python test_env.py task=MotionTracking/cat_walk
python test_env.py task=MotionTracking/angry_walk
python test_env.py task=MotionTracking/stealthy_walk      
python test_env.py task=MotionTracking/jog 
python test_env.py task=MotionTracking/trot
python test_env.py task=MotionTracking/boxing
python test_env.py task=MotionTracking/indian
python test_env.py task=MotionTracking/chacha
python test_env.py task=MotionTracking/lambada

# Amp with privileged information
python test_env.py task=MotionTracking/amp algo=amp total_frames=600000000

# Reset & Distill
python rollout.py task=MotionTracking/walk task.num_envs=4 algo=ppo checkpoint_path=ckpt_path
python distill.py checkpoint_path=ckpt_path

# Evaluation
python eval.py task.num_envs=4096 algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path                               # for metric
python eval.py task.num_envs=1 headless=false eval_render=true algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path  # for visualization