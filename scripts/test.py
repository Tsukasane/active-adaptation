import torch
import torchvision
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

from omni.isaac.lab.app import AppLauncher
# from omni_drones.utils.wandb import init_wandb
from active_adaptation.utils.torchrl import SyncDataCollector

# local import
from helpers import make_env_policy, EpisodeStats, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def log_video(env, it, render_interval, render_decimation):
    if it == 0 or it - env.last_recording_it >= render_interval:
        env.start_recording(render_decimation)
        env.last_recording_it = it

    frames = env.get_complete_frames()
    if len(frames) > 0:
        env.pause_recording()
        video_array = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        video_tensor = torch.from_numpy(video_array)

        run_dir = wandb.run.dir
        video_path = os.path.join(run_dir, f"video_{it}.mp4")
        torchvision.io.write_video(
            video_path,
            video_tensor.permute(0, 2, 3, 1),  # Change to (T, H, W, C) format
            fps=1 / env.step_dt / env.render_decimation
        )
        
        wandb.log({"video": wandb.Video(video_path)}, step=it)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, policy, vecnorm = make_env_policy(cfg)

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    eval_interval = cfg.get("eval_interval", -1)
    render_interval = cfg.get("render_interval", -1)
    render_decimation = cfg.get("render_decimation", 1)
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    rollout_policy = policy.get_rollout_policy("train")
    compile_policy = cfg.get("compile", False)
    assert compile_policy in (True, False, "auto")
    if compile_policy or compile_policy == "auto":
        fake_td = env.fake_tensordict()
        rollout_policy_compiled = torch.compile(rollout_policy)
        for _ in range(16): 
            rollout_policy_compiled(fake_td)
    if compile_policy == "auto":
        @torch.inference_mode()
        def _timeit(policy):
            start = time.perf_counter()
            for _ in range(128): 
                policy(fake_td)
            return (time.perf_counter() - start) / 128
        inference_time = _timeit(rollout_policy)
        inference_time_compiled = _timeit(rollout_policy_compiled)
        print(f"Inference time: {inference_time:.4f} -> {inference_time_compiled:.4f}")
        if inference_time_compiled < inference_time:
            rollout_policy = rollout_policy_compiled
            print("Using compiled policy")

    collector = SyncDataCollector(
        env,
        policy=rollout_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    pbar = tqdm(collector, total=total_iters)
    
    for i, data in enumerate(pbar):
        start = time.perf_counter()
        
        info = {}

        episode_stats.add(data)

        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
            info.update(env.extra)
        
        info.update(policy.train_op(data))
        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = collector._frames
        info["rollout_fps"] = collector._fps
        info["training_time"] = time.perf_counter() - start

        if render_interval > 0:
            log_video(env, i, render_interval, render_decimation)

        print()
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    info["env_frames"] = collector._frames

    exit(0)
    
    base_env.close()
    simulation_app.close()
    exit(0)


if __name__ == "__main__":
    main()

