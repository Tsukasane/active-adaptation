python test_env.py task=MotionTracking/walk

python test_env.py task=MotionTracking/backwalk

python test_env.py task=MotionTracking/joint_walk         # slip penalty, max feet height

python test_env.py task=MotionTracking/mickey_walk        # add cum_error_kp, remove cum_error_qpos

python test_env.py task=MotionTracking/cat_walk           # slip penalty, max feet height

python test_env.py task=MotionTracking/angry_walk         # remove cum_error_qpos, track eff

python test_env.py task=MotionTracking/stealthy_walk      

python test_env.py task=MotionTracking/jog                # action rate penalty + 

python test_env.py task=MotionTracking/trot

python test_env.py task=MotionTracking/boxing             # add cum_error_kp, remove cum_error_qpos, hand eff weight, root deviation loose

python test_env.py task=MotionTracking/indian             # remove cum_error_qpos

python test_env.py task=MotionTracking/chacha

python test_env.py task=MotionTracking/lambada            # action rate penalty 

# python test_env.py task=MotionTracking/amp algo=amp task.name=amp_logic_loss total_frames=600000000

# python eval.py task.num_envs=2048 algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path

# python eval.py task.num_envs=8 eval_render=true algo=ppo task=MotionTracking/walk checkpoint_path=ckpt_path

