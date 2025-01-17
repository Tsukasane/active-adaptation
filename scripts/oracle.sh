# # Single Oracle Policy
for seed in 0 3 2025
do
    python test_env.py task=MotionTracking/walk seed=$seed task.name=walk-$seed
    python test_env.py task=MotionTracking/backwalk seed=$seed task.name=backwalk-$seed
    python test_env.py task=MotionTracking/joint_walk seed=$seed task.name=joint_walk-$seed
    python test_env.py task=MotionTracking/mickey_walk seed=$seed task.name=mickey_walk-$seed
    python test_env.py task=MotionTracking/cat_walk seed=$seed task.name=cat_walk-$seed
    python test_env.py task=MotionTracking/angry_walk seed=$seed task.name=angry_walk-$seed
    python test_env.py task=MotionTracking/stealthy_walk seed=$seed task.name=stealthy_walk-$seed
    python test_env.py task=MotionTracking/jog seed=$seed task.name=jog-$seed
    python test_env.py task=MotionTracking/trot seed=$seed task.name=trot-$seed
    python test_env.py task=MotionTracking/boxing seed=$seed task.name=boxing-$seed
    python test_env.py task=MotionTracking/indian seed=$seed task.name=indian-$seed
    python test_env.py task=MotionTracking/chacha seed=$seed task.name=chacha-$seed
    python test_env.py task=MotionTracking/lambada seed=$seed task.name=lambada-$seed
done
