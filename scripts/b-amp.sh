# # Amp with privileged information
for seed in 0 3 2025
do
    python test_env.py task=MotionTracking/amp algo=amp total_frames=600000000 seed=$seed task.name=Amp-$seed
done