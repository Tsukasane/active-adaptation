# ckpt_dir=$(pwd)/checkpoints/cosmo-64
# replay_dir=$(pwd)/checkpoints/replay_cosmo-64
# output_dir=$(pwd)/outputs/

# get_final_ckpt() {
#     local latest_date=$(ls  $output_dir | tail -n 1)
#     local latest_date_path="$output_dir/$latest_date"
#     local latest_run=$(ls $latest_date_path | tail -n 1)
#     local latest_run_path="$latest_date_path/$latest_run"
#     mv $latest_run_path/wandb/latest-run/files/checkpoint_final.pt $ckpt_dir/$1-$2-64.pt
#     echo $ckpt_dir/$1-$2-64.pt
# }

# python test_env.py task=Cosmo/walk algo=ppo_adv

# ckpt_path=$(get_final_ckpt 1 walk)
# python rollout.py task=Cosmo/walk algo=ppo_adv checkpoint_path=$ckpt_path
# mv $(pwd)/rollout-walk.pt $replay_dir/1-rollout-walk.pt

# tasks=("backwalk", "joint_walk", "mickey_walk", "cat_walk", "angry_walk", \
#         "stealthy_walk", "jog", "trot", "boxing", "indian", "chacha", "lambada")

# counter=1

# for task in ${tasks[@]}; do
#     python test_env.py task=Cosmo/$task algo=ppo_vim checkpoint_path=$ckpt_path

#     counter=$((counter+1))
#     ckpt_path=$(get_final_ckpt $counter $task)
#     python rollout.py task=Cosmo/$task algo=ppo_adv checkpoint_path=$ckpt_path
#     mv $(pwd)/rollout-$task.pt $replay_dir/$counter-rollout-$task.pt
# done


counter=1
for task in walk backwalk joint_walk mickey_walk cat_walk angry_walk stealthy_walk \
                jog trot boxing indian chacha lambada
    do
        python test_env.py task=Cosmo/$counter-$task algo=ppo_adapt
        counter=$((counter+1))
        echo $counter
    done