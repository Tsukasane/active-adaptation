# # Single Oracle Policy
for seed in 0 3 2025
do
    for task in walk backwalk joint_walk mickey_walk cat_walk angry_walk stealthy_walk \
                jog trot boxing indian chacha lambada
        do
            python test_env.py task=MotionTracking/$task seed=$seed task.name=$task-$seed
        done
done

# Amp with privileged information
for seed in 0 3 2025
do
    python test_env.py task=MotionTracking/amp algo=amp total_frames=600000000 seed=$seed task.name=Amp-$seed
done

# OmniH2o-teacher
for seed in 0 3 2025
do
    python test_env.py task=MotionTracking/amp algo=h2o total_frames=600000000 seed=$seed task.name=OmniH2o-$seed
done