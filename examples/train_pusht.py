"""Training pipeline for PushT dataset.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import time

import hydra
import loguru
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.pusht_dataset import make_dataset
from mip.envs.pusht import make_vec_env
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed
from mip.losses import bridge_loss, bridge_v2_loss  # Explicit import
from mip.losses import generate_ou_noise    ##!! BPv3
from mip.losses import _build_tau_ladder, _sample_noise, _forward_noise



def get_batch_bp(dataset, batch_size, T_pred, H, device):
    """Custom Batch Sampler for Bridge Policy (Strategy B: Padding)."""
    # 1. Sample Indices (Strategy B: Allow sampling up to T_total)
    # We sample indices corresponding to t_start
    episode_indices = np.random.randint(0, dataset.replay_buffer.n_episodes, size=batch_size)

    # Get episode boundaries
    # ReplayBuffer stores all data concatenated, but SequenceSampler knows boundaries.
    # We'll use dataset.sampler.episode_ends logic manually or rely on replay buffer slicing.
    # dataset.replay_buffer.episode_ends gives the end index of each episode in the buffer.
    episode_ends = dataset.replay_buffer.episode_ends[:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    # For each sampled episode, sample a valid t_start
    # Valid range: [0, Episode_Length] (We allow full range due to padding)
    start_indices = []

    for i in range(batch_size):
        ep_idx = episode_indices[i]
        ep_start = episode_starts[ep_idx]
        ep_end = episode_ends[ep_idx]
        ep_len = ep_end - ep_start

        # Sample relative t_start in [0, ep_len)
        # Note: We can sample right up to the end!
        t_rel = np.random.randint(0, ep_len) # Strategy B
        start_indices.append(ep_start + t_rel)

    start_indices = np.array(start_indices) # (B,)

    # 2. Sample Tau in [0, 1]
    tau = np.random.rand(batch_size) # (B,)

    # 3. Calculate time indices
    # t_start indices: start_indices
    # t_end indices: start_indices + T_pred
    # t_curr indices: start_indices + tau * T_pred

    # Since we need Chunks of length H, we need:
    # C_start: [t_start, t_start + H]
    # C_end:   [t_start + T_pred, t_start + T_pred + H]
    # h_curr:  at t_curr = t_start + int(tau * T_pred)

    # To do this efficiently, we can fetch max required range and slice, or fetch separately.
    # Fetching separately is easier for padding logic.

    def fetch_chunk(indices, length, key="action"):
        """Fetch chunk with Repeat Last Frame padding."""
        batch_data = []
        for i, idx in enumerate(indices):
            ep_idx = episode_indices[i]
            ep_end = episode_ends[ep_idx]

            # Calculate range
            curr_idx = idx
            valid_len = ep_end - curr_idx

            if valid_len >= length:
                # No padding needed
                chunk = dataset.replay_buffer[key][curr_idx : curr_idx + length]
            elif valid_len > 0:
                # Partial valid
                valid_chunk = dataset.replay_buffer[key][curr_idx : ep_end]
                last_val = valid_chunk[-1:]
                pad_len = length - valid_len
                pad_chunk = np.repeat(last_val, pad_len, axis=0)
                chunk = np.concatenate([valid_chunk, pad_chunk], axis=0)
            else:
                # Completely out of bounds (shouldn't happen with t_rel < ep_len, but for C_end it might)
                # Use last frame of episode
                last_val = dataset.replay_buffer[key][ep_end - 1 : ep_end]
                chunk = np.repeat(last_val, length, axis=0)

            batch_data.append(chunk)
        return np.array(batch_data) # (B, L, D)

    # Fetch C_start (at t_start)
    C_start_raw = fetch_chunk(start_indices, H, key="action")

    # [工程 Trick] 给起点加一点点扰动 (Jitter)，模拟推理时的 Handover 误差
    # 这个噪声不用很大，0.01 左右即可
    if np.random.rand() < 0.5:  # 50% 的概率加噪，保留 50% 的完美情况
       C_start_raw = C_start_raw + np.random.normal(0, 0.1, size=C_start_raw.shape)

    # Fetch C_end (at t_start + T_pred)
    end_indices = start_indices + T_pred
    C_end_raw = fetch_chunk(end_indices, H, key="action")

    # Fetch h_curr (at t_curr)
    # Obs requires special handling for Image vs State
    curr_indices = start_indices + (tau * T_pred).astype(int)

    obs_batch = {}
    if "image" in dataset.normalizer["obs"]: # PushTImageDataset
        # Need to handle image fetching + normalization
        # Image is (B, C, H, W)
        # ReplayBuffer stores (B, H, W, C) usually? No, check dataset code.
        # PushTImageDataset: sample["img"] is (T, H, W, C)
        # And we need sequence of length obs_steps?
        # Wait, BP.md says "h_curr = GetObs(t_curr)".
        # Typically h_curr includes history.
        # Let's assume obs_steps is 2 (default).
        # We need [t_curr - obs_steps + 1, t_curr + 1]

        obs_steps = dataset.n_obs_steps
        # We need to fetch chunk of length obs_steps ending at curr_indices
        # Actually SequenceSampler usually fetches [t, t+obs_steps] ??
        # Let's stick to standard: obs at time t usually means window ending at t.
        # But for simplicity and consistency with dataset:
        # If we request idx, we get [idx, idx+obs_steps].
        # Let's use that.

        # Fetch images
        imgs_raw = fetch_chunk(curr_indices, obs_steps, key="img") # (B, T, H, W, C)

        # Process Images
        # (B, T, H, W, C) -> (B, T, C, H, W)
        imgs = np.moveaxis(imgs_raw, -1, 2).astype(np.float32) / 255.0
        # Normalize
        obs_batch["image"] = dataset.normalizer["obs"]["image"].normalize(imgs)

        # Agent Pos
        state_raw = fetch_chunk(curr_indices, obs_steps, key="state")
        agent_pos = state_raw[..., :2].astype(np.float32)
        obs_batch["agent_pos"] = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos)

    elif "keypoint" in dataset.normalizer["obs"]: # PushTKeypointDataset
         # Similar logic for keypoints
         kp_raw = fetch_chunk(curr_indices, dataset.horizon, key="keypoint") # Just fetch horizon
         # Extract obs_steps
         kp_raw = kp_raw[:, :dataset.pad_before + 1, :] # Simplified
         # ... (Implementation detail omitted for brevity, focusing on State/Image)
         pass
    else: # State
        state_raw = fetch_chunk(curr_indices, dataset.horizon, key="state") # Fetch enough
        # Take first obs_steps
        # Note: dataset.pad_before = obs_steps - 1
        obs_steps = dataset.pad_before + 1
        state_obs = state_raw[:, :obs_steps, :].astype(np.float32)
        obs_batch["state"] = dataset.normalizer["obs"]["state"].normalize(state_obs)

    # Normalize Actions
    C_start = dataset.normalizer["action"].normalize(C_start_raw.astype(np.float32))
    C_end = dataset.normalizer["action"].normalize(C_end_raw.astype(np.float32))

    # Convert to Tensor
    def to_device(x):
        if isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        return torch.tensor(x, dtype=torch.float32, device=device)

    return (
        to_device(C_start),
        to_device(C_end),
        to_device(obs_batch),
        to_device(tau) # (B,)
    )


def train_bp(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Bridge Policy Training Loop (Custom)."""
    # 1. Setup
    optimizer = agent.optimizer
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.optimization.gradient_steps
    )

    # Constants
    T_pred = config.optimization.prediction_offset
    H = config.task.horizon - T_pred
    batch_size = config.optimization.batch_size
    device = config.optimization.device

    # Resume logic ... (same as standard train)
    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()

    loguru.logger.info(f"Starting Bridge Policy Training (H={H}, T_pred={T_pred})")

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # 2. Get Data (Custom Sampling)
        C_start, C_end, h_curr, tau = get_batch_bp(dataset, batch_size, T_pred, H, device)

        # 3. Compute Loss
        # We call bridge_loss directly or via a wrapper if we want to keep agent structure
        # Here we bypass agent.update for flexibility with new signature

        agent.train()
        optimizer.zero_grad()

        loss, info = bridge_loss(
            config.optimization,
            agent.flow_map,
            agent.encoder,
            agent.interpolant,
            C_start,
            C_end,
            h_curr,
            tau
        )

        loss.backward()

        # Gradient Clip
        params = list(agent.encoder.parameters()) + list(agent.flow_map.parameters())
        if config.optimization.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(params, config.optimization.grad_clip_norm)

        optimizer.step()

        # EMA Update
        if agent.config.optimization.ema_rate < 1:
            agent.ema_update()

        lr_scheduler.step()

        # Logging
        info["loss"] = loss.item()
        info_list.append(info)

        # ... (Log metrics logic same as standard train) ...
        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([item[key] for item in info_list])
                except Exception:
                    pass
            logger.log(metrics, category="train")
            info_list = []

        # ... (Save & Eval logic same as standard train) ...
        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            # BP Evaluation
            metrics.update(evaluate_bp(config, envs, dataset, agent, logger))

            # Save Best (Same logic)
            # Use mean_success as primary metric
            primary_metric_key = "mean_success"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in best_metrics
                    or metrics[primary_metric_key]
                    > best_metrics.get(primary_metric_key, -1.0)
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    logger.save_agent(agent=agent, identifier="best")
                    best_metrics[primary_metric_key] = success_rate

            logger.log(metrics, category="eval")
            agent.train()

        if (n_gradient_step + 1) % config.log.save_freq == 0:
             logger.save_agent(agent=agent, identifier="latest")


def train_bpv2(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Bridge Policy V2 Training Loop."""
    # 1. Setup
    optimizer = agent.optimizer
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.optimization.gradient_steps
    )

    # Dataloader (Standard, returns chunks of length H+1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4 if config.task.obs_type in ["state", "keypoint"] else 8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    loop_loader = loop_dataloader(dataloader)

    # Resume logic
    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()

    loguru.logger.info(f"Starting Bridge Policy V2 Training")

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # 2. Get Data
        batch = next(loop_loader)

        # Preprocess Obs
        if config.task.obs_type == "image":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs[k] = obs_batch[k][:, : config.task.obs_steps, :].to(
                    config.optimization.device
                )
        elif config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]
        elif config.task.obs_type == "keypoint":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs_data = obs_batch[k].to(config.optimization.device)
                obs[k] = obs_data[:, : config.task.obs_steps, :]

        # Act: (B, H+1, D)
        act = batch["action"].to(config.optimization.device)
        # Note: We assume dataset.horizon is set to H+1 in config or managed by user
        act = act[:, : config.task.horizon, :]

        # 3. Compute Loss
        agent.train()
        optimizer.zero_grad()

        # Dummy delta_t (not used in bpv2_loss but required by signature match if using agent.update,
        # but here we call loss directly)
        delta_t = torch.zeros((act.shape[0],), device=config.optimization.device)

        loss, info = bridge_v2_loss(
            config.optimization,
            agent.flow_map,
            agent.encoder,
            agent.interpolant,
            act,
            obs,
            delta_t
        )

        loss.backward()

        # Gradient Clip
        params = list(agent.encoder.parameters()) + list(agent.flow_map.parameters())
        if config.optimization.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(params, config.optimization.grad_clip_norm)

        optimizer.step()
        if agent.config.optimization.ema_rate < 1:
            agent.ema_update()
        lr_scheduler.step()

        # Logging
        info["loss"] = loss.item()
        info_list.append(info)

        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([item[key] for item in info_list])
                except Exception:
                    pass
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            metrics.update(evaluate_bpv2(config, envs, dataset, agent, logger))

            # Save Best
            primary_metric_key = "mean_success"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in best_metrics
                    or metrics[primary_metric_key]
                    > best_metrics.get(primary_metric_key, -1.0)
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(f"New best model! {primary_metric_key} = {success_rate:.4f}")
                    logger.save_agent(agent=agent, identifier="best")
                    best_metrics[primary_metric_key] = success_rate

            logger.log(metrics, category="eval")
            agent.train()

        if (n_gradient_step + 1) % config.log.save_freq == 0:
             logger.save_agent(agent=agent, identifier="latest")


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Standalone training function.

    Args:
        config: Configuration for training
        envs: Environment
        dataset: Training dataset
        agent: Agent to train
        logger: Logger for metrics
        resume_state: Optional dict with training state to resume from
    """

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4 if config.task.obs_type in ["state", "keypoint"] else 8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True,
    )
    loop_loader = loop_dataloader(dataloader)

    # lr scheduler
    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=config.optimization.gradient_steps
    )

    # warmup scheduler (mainly for flow map learning)
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    # Resume from checkpoint if available
    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        loguru.logger.info(f"Resuming training from step {start_step}")
        loguru.logger.info(f"Restored best metrics: {best_metrics}")

        # Fast-forward the lr_scheduler to the correct step
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # get batch from dataloader
        batch = next(loop_loader)

        # preprocess data
        if config.task.obs_type == "image":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs_data = obs_batch[k].to(config.optimization.device)
                if config.optimization.loss_type in ["prcp_v2"]:
                    obs[k] = obs_data[:, : config.task.obs_steps + 1, :]
                else:
                    obs[k] = obs_data[:, : config.task.obs_steps, :]
        elif config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device)
            if config.optimization.loss_type in ["prcp_v2"]:
                obs = obs[:, : config.task.obs_steps + 1, :]
            else:
                obs = obs[:, : config.task.obs_steps, :]  # (B, obs_horizon, obs_dim)
        elif config.task.obs_type == "keypoint":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs_data = obs_batch[k].to(config.optimization.device)
                if config.optimization.loss_type in ["prcp_v2"]:
                    obs[k] = obs_data[:, : config.task.obs_steps + 1, :]
                else:
                    obs[k] = obs_data[:, : config.task.obs_steps, :]  # (B, obs_horizon, obs_dim)
        else:
            raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

        # update diffusion
        if config.optimization.loss_type == "bridge_v2":
            delta_t_scalar = float(n_gradient_step)
        else:
            delta_t_scalar = warmup_scheduler(n_gradient_step)
        batch_size = act.shape[0]
        delta_t = torch.full(
            (batch_size,), delta_t_scalar, device=config.optimization.device
        )
        info = agent.update(act, obs, delta_t)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.item()
        lr_scheduler.step()
        info_list.append(info)

        # log metrics
        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except (KeyError, TypeError, ValueError) as e:
                    loguru.logger.error(f"Error calculating {key}: {e}")
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % config.log.save_freq == 0:
            loguru.logger.info("Save model...")
            logger.save_agent(agent=agent, identifier="latest")

        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                if config.optimization.loss_type in ["bridge_v3", "bridge_v2", "prcp_v1", "prcp_v2"]:
                    metrics.update(
                        evaluate_bpv3(config, envs, dataset, agent, logger, num_steps)
                    )
                elif config.optimization.loss_type == "rp_v1":
                    metrics.update(
                        evaluate_rpv1(config, envs, dataset, agent, logger, num_steps)
                    )
                else:
                    metrics.update(
                        evaluate(config, envs, dataset, agent, logger, num_steps)
                    )

                # metrics.update(
                #     evaluate_bpv3(config, envs, dataset, agent, logger, num_steps)
                # )

            # Update best metrics and average metrics
            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            # Check if this is a new best model based on success rate
            # Use the first num_steps in the list as the primary metric
            primary_metric_key = f"mean_success_{num_steps_list[0]}"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in old_best_metrics
                    or metrics[primary_metric_key]
                    > old_best_metrics[primary_metric_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    # Save to local models directory
                    logger.save_agent(agent=agent, identifier="best")

                    # Save to global checkpoints directory with success rate comparison
                    # Include training state for resuming
                    checkpoint_base_name = (
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
                    )
                    if config.optimization.loss_type == "bridge_v2":
                        checkpoint_base_name += f"_BPv2"
                    elif config.optimization.loss_type == "bridge_v3":
                        checkpoint_base_name += f"_BPv3"
                    elif config.optimization.loss_type == "prcp_v1":
                        checkpoint_base_name += f"_PRCPv1"
                    elif config.optimization.loss_type == "prcp_v2":
                        checkpoint_base_name += f"_PRCPv2"
                    elif config.optimization.loss_type == "rp_v1":
                        checkpoint_base_name += f"_RPv1"

                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                    }
                    logger.save_global_checkpoint(
                        agent,
                        checkpoint_base_name,
                        success_rate,
                        training_state=training_state,
                    )

            # Add best and average metrics to current metrics for logging
            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value

            # Print best and average metrics
            loguru.logger.info("Best metrics so far:")
            for key, value in best_metrics.items():
                loguru.logger.info(f"  {key}: {value:.4f}")
            if avg_metrics:
                loguru.logger.info("Average metrics (last 5 evals):")
                for key, value in avg_metrics.items():
                    loguru.logger.info(f"  {key}: {value:.4f}")

            logger.log(metrics, category="eval")
            agent.train()

        if (n_gradient_step + 1) % config.log.save_freq == 0:
             logger.save_agent(agent=agent, identifier="latest")


def evaluate(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        config: Configuration object containing evaluation parameters
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        num_steps: Number of steps for sampling

    Returns:
        dict: Metrics including mean step, reward, and success rate
    """
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * config.task.num_envs
        # NOTE: update env seed, the original envs is update seed so reset is broken
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * config.task.num_envs + j)
        obs, _ = envs.reset()
        t = 0

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < config.task.max_episode_steps:
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs = torch.tensor(
                    obs, device=config.optimization.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)
                obs = {"state": obs}
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)  # (num_envs, obs_steps, 20)
                # Split into keypoint and agent_pos
                keypoint_obs = obs_raw[:, :, :18]  # (num_envs, obs_steps, 18)
                agent_pos_obs = obs_raw[:, :, 18:20]  # (num_envs, obs_steps, 2)

                # Normalize
                nkeypoint = (
                    dataset.normalizer["obs"]["keypoint"]
                    .normalize(keypoint_obs.reshape(-1, 2))
                    .reshape(config.task.num_envs, config.task.obs_steps, 18)
                )
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(
                    agent_pos_obs
                )

                obs = {
                    "keypoint": torch.tensor(
                        nkeypoint,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                    "agent_pos": torch.tensor(
                        nagent_pos,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                }
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs = {}
                for k in obs_raw:
                    obs[k] = obs_raw[k].astype(
                        np.float32
                    )  # (num_envs, obs_steps, obs_dim)
                    obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
                    obs[k] = torch.tensor(
                        obs[k], device=config.optimization.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, obs_dim)
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            act_0 = torch.randn(
                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                device=config.optimization.device,
            )
            # run sampling (num_envs, horizon, action_dim)
            act_normed = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
            )

            # unnormalize prediction
            act_normed = (
                act_normed.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            act = dataset.normalizer["action"].unnormalize(act_normed)

            # get action by slicing from start to end
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, _ = envs.step(act)
            _ = terminated | truncated  # Track done status
            ep_reward += reward
            step_reward.append(reward)
            t += config.task.act_steps

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics


###!! 写一个专门针对 Bridge Policy 的 eval 函数吧
def evaluate_bp(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Bridge Policy Streaming Inference."""
    # Constants
    T_pred = config.optimization.prediction_offset
    H = config.task.horizon - T_pred
    # Bridge dt (flow step)
    dtau = 1.0 / T_pred

    episode_rewards = []
    episode_steps = []
    episode_success = []

    bs = config.task.num_envs
    device = config.optimization.device
    act_dim = config.task.act_dim

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * bs

        # Seed and Reset
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * bs + j)
        obs, _ = envs.reset()
        t_step = 0

        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))

        # === 1. Cold Start ===
        # Initialize State (X_curr, tau)
        # Stationary Padding: X_curr = [Pos, Pos, ..., Pos]

        # Initialize with Zeros first, will be filled by Stationary Padding logic
        X_curr = torch.zeros((bs, H, act_dim), device=device)
        tau = 0.0

        # Flag for first step
        is_first_step = True

        while t_step < config.task.max_episode_steps:
            # 1. Process Observation
            # Convert Env Obs to Tensor Batch
            if config.task.obs_type == "image":
                obs_raw = obs
                obs_batch = {}
                # Handle Image: (B, H, W, C) -> (B, C, H, W)
                img = obs_raw["image"].astype(np.float32) / 255.0
                img = np.moveaxis(img, -1, 1)
                obs_batch["image"] = dataset.normalizer["obs"]["image"].normalize(img)
                obs_batch["image"] = torch.tensor(obs_batch["image"], device=device, dtype=torch.float32)

                # Handle Agent Pos
                pos = obs_raw["agent_pos"].astype(np.float32)
                obs_batch["agent_pos"] = dataset.normalizer["obs"]["agent_pos"].normalize(pos)
                obs_batch["agent_pos"] = torch.tensor(obs_batch["agent_pos"], device=device, dtype=torch.float32)

                current_pos = obs_batch["agent_pos"][:, -1, :] # (B, D)

            elif config.task.obs_type == "state":
                state = obs.astype(np.float32)
                state = dataset.normalizer["obs"]["state"].normalize(state)
                obs_batch = {"state": torch.tensor(state, device=device, dtype=torch.float32)}
                current_pos = obs_batch["state"][:, -1, :2] # Assuming first 2 are pos

            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)
                # Split keypoint and agent pos
                keypoint = obs_raw[:, :, :18]
                agent_pos = obs_raw[:, :, 18:20]

                nkeypoint = dataset.normalizer["obs"]["keypoint"].normalize(keypoint.reshape(-1, 2)).reshape(bs, config.task.obs_steps, 18)
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos)

                obs_batch = {
                    "keypoint": torch.tensor(nkeypoint, device=device, dtype=torch.float32),
                    "agent_pos": torch.tensor(nagent_pos, device=device, dtype=torch.float32)
                }
                current_pos = obs_batch["agent_pos"][:, -1, :]

            # 2. Stationary Padding (Cold Start)
            if is_first_step:
                # X_curr = Repeat(Current_Pos)
                X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
                is_first_step = False

            # 3. Network Prediction
            # Input: X_curr, tau, obs_batch
            # We use agent.model directly
            tau_tensor = torch.full((bs,), tau, device=device)

            # Use agent.sample interface if possible, or direct model call
            # Since we need X_next, let's use direct model call + sampler helper
            # But sampler helper is not imported here. Let's call model directly.

            # Use EMA for evaluation if available
            encoder = agent.encoder_ema if config.optimization.ema_rate < 1 else agent.encoder
            flow_map = agent.flow_map_ema if config.optimization.ema_rate < 1 else agent.flow_map

            obs_emb = encoder(obs_batch, None)

            with torch.no_grad():
                v_pred = flow_map.get_velocity(tau_tensor, X_curr, obs_emb)

            # 4. Micro-step Integration
            X_next = X_curr + v_pred * dtau

            # 5. Execute Action
            # Execute first action of X_next
            act_normed = X_next[:, 0, :] # (B, D)

            # Unnormalize
            act_np = act_normed.detach().cpu().numpy()
            act_un = dataset.normalizer["action"].unnormalize(act_np)

            # Env Step
            # Reshape for env if needed (env expects (B, act_steps, D))
            if act_un.ndim == 2:
                act_un = act_un[:, None, :]

            obs, reward, terminated, truncated, _ = envs.step(act_un)

            # 6. Update State
            X_curr = X_next
            tau += dtau

            # 7. Handover (Rolling)
            if tau >= 1.0 - 1e-5: # Float tolerance
                tau = 0.0
                # Ideally we keep X_curr as is (Seamless Handover)
                # It is now the "Start Chunk" for the next Bridge
                pass

            # Stats
            ep_reward += reward
            step_reward.append(reward)
            t_step += 1

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t_step)
        episode_success.append(success)

    # loguru.logger.info(
    #     f"BP Eval | Mean success: {np.nanmean(episode_success)}"
    # )
    loguru.logger.info(
        f"BP Eval | Mean step: {np.nanmean(episode_steps):.2f} "
        f"Mean reward: {np.nanmean(episode_rewards):.4f} "
        f"Mean success: {np.nanmean(episode_success):.4f}"
    )

    return {
        "mean_step": np.nanmean(episode_steps),
        "mean_reward": np.nanmean(episode_rewards),
        "mean_success": np.nanmean(episode_success),
    }

def evaluate_bpv2(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Bridge Policy V2 Inference (Recurrent Refinement)."""
    # H = Horizon
    # 自动减 1，适配训练时的切片逻辑
    H = config.task.horizon - 1

    episode_rewards = []
    episode_steps = []
    episode_success = []

    bs = config.task.num_envs
    device = config.optimization.device
    act_dim = config.task.act_dim

    # Get BPv2 Sampler
    from mip.samplers import bridge_v2_sampler

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * bs

        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * bs + j)
        obs, _ = envs.reset()
        t_step = 0

        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))

        # === 1. Cold Start ===
        # Initialize X_curr with Zeros, will be filled by Stationary Padding logic
        X_curr = torch.zeros((bs, H, act_dim), device=device)
        is_first_step = True

        while t_step < config.task.max_episode_steps:
            # 1. Process Observation
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs_batch = {"state": torch.tensor(obs, device=device, dtype=torch.float32)}
                current_pos = obs_batch["state"][:, -1, :2]
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs_batch = {}
                for k in obs_raw:
                    # 使用局部变量 val，不修改原 obs 字典
                    val = obs_raw[k].astype(np.float32)
                    val = dataset.normalizer["obs"][k].normalize(val)
                    obs_batch[k] = torch.tensor(val, device=device, dtype=torch.float32)

                # Special handling for Current Pos needed for Stationary Padding
                current_pos = obs_batch["agent_pos"][:, -1, :]
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)
                keypoint_obs = obs_raw[:, :, :18]
                agent_pos_obs = obs_raw[:, :, 18:20]

                nkeypoint = dataset.normalizer["obs"]["keypoint"].normalize(keypoint_obs.reshape(-1, 2)).reshape(bs, config.task.obs_steps, 18)
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos_obs)

                obs_batch = {
                    "keypoint": torch.tensor(nkeypoint, device=device, dtype=torch.float32),
                    "agent_pos": torch.tensor(nagent_pos, device=device, dtype=torch.float32)
                }
                current_pos = obs_batch["agent_pos"][:, -1, :]
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            # 2. Stationary Padding (Cold Start)
            if is_first_step:
                # X_curr = Repeat(Current_Pos)
                X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
                is_first_step = False

            # 3. Network Prediction (One-step Refinement)
            with torch.no_grad():
                # Use agent.sample interface for consistency
                # Note: agent.sample expects act_0, obs, num_steps
                # For BPv2, we pass X_curr as act_0
                X_next = agent.sample(
                    act_0=X_curr,
                    obs=obs_batch,
                    num_steps=1,
                    use_ema=True # Explicitly use EMA
                )

            # 4. Execute Action
            # Execute first action of X_next
            act_normed = X_next[:, 0, :]
            act_np = act_normed.detach().cpu().numpy()
            act_un = dataset.normalizer["action"].unnormalize(act_np)

            if act_un.ndim == 2:
                act_un = act_un[:, None, :]

            obs, reward, terminated, truncated, _ = envs.step(act_un)

            # 5. Update State
            # Simply carry over the prediction as the next state
            X_curr = X_next

            # Stats
            ep_reward += reward
            step_reward.append(reward)
            t_step += 1

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t_step)
        episode_success.append(success)

    # loguru.logger.info(f"BPv2 Eval | Mean success: {np.nanmean(episode_success)}")
    loguru.logger.info(
        f"BPv2 Eval | Nstep: {num_steps} Mean step: {np.nanmean(episode_steps):.2f} "
        f"Mean reward: {np.nanmean(episode_rewards):.4f} "
        f"Mean success: {np.nanmean(episode_success):.4f}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics


def evaluate_bp_old(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        config: Configuration object containing evaluation parameters
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        num_steps: Number of steps for sampling

    Returns:
        dict: Metrics including mean step, reward, and success rate
    """
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * config.task.num_envs
        # NOTE: update env seed, the original envs is update seed so reset is broken
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * config.task.num_envs + j)
        obs, _ = envs.reset()
        t = 0

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        ##*** 第一步 act_0 就不是随机噪声了，这里还没进入到下面的循环，最好直接在这里复制！
        bs = config.task.num_envs
        act_dim = config.task.act_dim
        device = config.optimization.device
        # Get current agent position (point)
        if isinstance(obs, dict):
            if "agent_pos" in obs:
                curr_pos = obs["agent_pos"][:, -1, :]  # (B, D)
            elif "state" in obs:
                curr_pos = obs["state"][:, -1, :2]  # (B, D) assuming first 2 are pos
            else:
                curr_pos = torch.zeros((bs, act_dim), device=device)
        else:
            curr_pos = torch.zeros((bs, act_dim), device=device)

        # Repeat Pad to construct initial Window (B, Ta, D)
        ##!! 不是 Ta，Ta 现在变成 W 了，这里得是 H！得是 16！
        K = config.optimization.prediction_offset
        H = config.task.horizon - K  # 或者直接读 config.task.horizon
        act_0 = curr_pos.unsqueeze(1).expand(-1, H, -1)

        while t < config.task.max_episode_steps:
            # ## 采样，把 for 循环放这里？
            # K = getattr(config, "prediction_offset", 4)
            # for k in range(K):
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs = torch.tensor(
                    obs, device=config.optimization.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)
                obs = {"state": obs}
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)  # (num_envs, obs_steps, 20)
                # Split into keypoint and agent_pos
                keypoint_obs = obs_raw[:, :, :18]  # (num_envs, obs_steps, 18)
                agent_pos_obs = obs_raw[:, :, 18:20]  # (num_envs, obs_steps, 2)

                # Normalize
                nkeypoint = (
                    dataset.normalizer["obs"]["keypoint"]
                    .normalize(keypoint_obs.reshape(-1, 2))
                    .reshape(config.task.num_envs, config.task.obs_steps, 18)
                )
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(
                    agent_pos_obs
                )

                obs = {
                    "keypoint": torch.tensor(
                        nkeypoint,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                    "agent_pos": torch.tensor(
                        nagent_pos,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                }
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs = {}
                for k in obs_raw:
                    obs[k] = obs_raw[k].astype(
                        np.float32
                    )  # (num_envs, obs_steps, obs_dim)
                    obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
                    obs[k] = torch.tensor(
                        obs[k], device=config.optimization.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, obs_dim)
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            # run sampling (num_envs, horizon, action_dim)
            act_normed = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
            )

            ##!! 新的 act_0 就是网络自己输出的 act_normed
            act_0 = act_normed

            # unnormalize prediction
            act_normed = (
                act_normed.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            act = dataset.normalizer["action"].unnormalize(act_normed)

            # get action by slicing from start to end
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, _ = envs.step(act)
            _ = terminated | truncated  # Track done status
            ep_reward += reward
            step_reward.append(reward)
            t += config.task.act_steps

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics


# def evaluate_bpv3(config: Config, envs, dataset, agent, logger, num_steps=1):
#     """Bridge Policy V3 Inference (Rolling Buffer Denoising)."""

#     # BPv3 直接预测整个 Horizon，不需要像 BPv2 那样 H-1
#     H = config.task.horizon
#     bs = config.task.num_envs
#     device = config.optimization.device
#     act_dim = config.task.act_dim

#     episode_rewards = []
#     episode_steps = []
#     episode_success = []

#     # 预计算目标噪声表，用于冷启动加噪
#     sigma_min = 1e-4
#     sigma_max = 0.01
#     rho=0.9
#     R = sigma_max / sigma_min
#     start_idx = config.task.obs_steps - 1
#     L_eff = max(1.0, float(H - 1 - start_idx))

#     # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     # prog_target = torch.clamp((i_indices - start_idx) / L_eff, 0.0, 1.0)
#     # sigma_target = sigma_min * (R ** prog_target) # (bs, H)

#     for i in range(config.log.eval_episodes // bs):
#         step_reward = []
#         ep_reward = [0.0] * bs
#         # NOTE: update env seed, the original envs is update seed so reset is broken
#         for j in range(len(envs.envs)):
#             envs.envs[j].seed(config.optimization.seed + i * bs + j)
#         obs, _ = envs.reset()
#         t_step = 0

#         # initialize video stream
#         if config.log.save_video:
#             logger.video_init(envs.envs[0], enable=True, video_id=str(i))

#         # Initialize X_curr with Zeros, will be filled by Stationary Padding logic
#         X_curr = torch.zeros((bs, H, act_dim), device=device)
#         is_first_step = True

#         while t_step < config.task.max_episode_steps:
#             # 假设执行后，获得了 obs_batch 和 current_pos: (bs, act_dim)
#             # 1. Process Observation
#             if config.task.obs_type == "state":
#                 obs = obs.astype(np.float32)
#                 obs = dataset.normalizer["obs"]["state"].normalize(obs)
#                 obs_batch = {"state": torch.tensor(obs, device=device, dtype=torch.float32)}
#                 current_pos = obs_batch["state"][:, -1, :2]
#             elif config.task.obs_type == "image":
#                 obs_raw = obs
#                 obs_batch = {}
#                 for k in obs_raw:
#                     # 使用局部变量 val，不修改原 obs 字典
#                     val = obs_raw[k].astype(np.float32)
#                     val = dataset.normalizer["obs"][k].normalize(val)
#                     obs_batch[k] = torch.tensor(val, device=device, dtype=torch.float32)

#                 # Special handling for Current Pos needed for Stationary Padding
#                 current_pos = obs_batch["agent_pos"][:, -1, :]
#             elif config.task.obs_type == "keypoint":
#                 obs_raw = obs.astype(np.float32)
#                 keypoint_obs = obs_raw[:, :, :18]
#                 agent_pos_obs = obs_raw[:, :, 18:20]

#                 nkeypoint = dataset.normalizer["obs"]["keypoint"].normalize(keypoint_obs.reshape(-1, 2)).reshape(bs, config.task.obs_steps, 18)
#                 nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos_obs)

#                 obs_batch = {
#                     "keypoint": torch.tensor(nkeypoint, device=device, dtype=torch.float32),
#                     "agent_pos": torch.tensor(nagent_pos, device=device, dtype=torch.float32)
#                 }
#                 current_pos = obs_batch["agent_pos"][:, -1, :]
#             else:
#                 raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

#             # === 2. 冷启动初始化 (Cold Start) ===
#             if is_first_step:
#                 # # 初始动作：全部填充当前机器人的真实位置
#                 # X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
#                 # # 极其关键：按照合法的目标噪声表，赋予初始 AR 噪声，使系统一开始就在流形附近
#                 # eps = generate_ou_noise(bs, H, act_dim, device, rho=rho)
#                 # X_curr = X_curr + sigma_target.unsqueeze(-1) * eps
#                 # is_first_step = False

#                 # 用真实的机器状态 Padding (稳妥先验)
#                 X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)

#                 # # 按照稳态目标 (delta=0) 的 tau 比例混入 AR(1) 噪声
#                 # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#                 # tau_target = torch.clamp(1.0 - i_indices / (H - 1), 0.0, 1.0) # 左端1, 右端0

#                 # 按照合法的目标噪声表加入局部扰动
#                 i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#                 prog_target = torch.clamp((i_indices - start_idx) / L_eff, 0.0, 1.0)
#                 sigma_target = sigma_min * (R ** prog_target)

#                 eps = generate_ou_noise(bs, H, act_dim, device, rho=0.9)

#                 # 完美插值：左侧保留 current_pos，右侧渐变成纯 eps 噪声
#                 # X_curr = tau_target.unsqueeze(-1) * X_curr + (1.0 - tau_target.unsqueeze(-1)) * eps
#                 X_curr = X_curr + sigma_target.unsqueeze(-1) * eps

#                 is_first_step = False

#             # === 3. 网络预测 (BPv3 Sampler 内部包含了 Shift -> ODE) ===
#             with torch.no_grad():
#                 X_next = agent.sample(
#                     act_0=X_curr,
#                     obs=obs_batch,
#                     num_steps=config.task.act_steps,   #!! 反正是单步，这里就直接传 act_steps 得了
#                     use_ema=True
#                 )

#             # # === 4. 执行最左侧极净动作 ===
#             # act_normed = X_next[:, 0, :] # 提取 i=0 的动作
#             # act_np = act_normed.detach().cpu().numpy()
#             # act_un = dataset.normalizer["action"].unnormalize(act_np)

#             # if act_un.ndim == 2:
#             #     act_un = act_un[:, None, :]

#             # obs, reward, terminated, truncated, _ = envs.step(act_un)

#             # === 4. 标准 DP Slicing 取动作 ===
#             # 严格按照观测的时间窗口，取出属于当前 t 的有效动作
#             start = config.task.obs_steps - 1
#             end = start + config.task.act_steps
#             act_normed = X_next[:, start:end, :]

#             act_np = act_normed.detach().cpu().numpy()
#             act_un = dataset.normalizer["action"].unnormalize(act_np)

#             obs, reward, terminated, truncated, _ = envs.step(act_un)

#             # === 5. 状态轮转 ===
#             # 将打磨完的 Buffer 交给下一步去 shift
#             X_curr = X_next

#             ep_reward += reward
#             step_reward.append(reward)
#             t_step += 1

#         success = np.around(np.max(np.array(step_reward), axis=0), 2)
#         episode_rewards.append(ep_reward)
#         episode_steps.append(t_step)
#         episode_success.append(success)

#     loguru.logger.info(
#         f"BPv3 Eval | Nstep: {num_steps} Mean step: {np.nanmean(episode_steps):.2f} "
#         f"Mean reward: {np.nanmean(episode_rewards):.4f} "
#         f"Mean success: {np.nanmean(episode_success):.4f}"
#     )

#     metrics = {
#         f"mean_step_{num_steps}": np.nanmean(episode_steps),
#         f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
#         f"mean_success_{num_steps}": np.nanmean(episode_success),
#     }

#     return metrics


# def evaluate_bpv3(config: Config, envs, dataset, agent, logger, num_steps=1):
#     """Bridge Policy V3 Inference (Rolling Buffer Denoising)."""

#     # BPv3 直接预测整个 Horizon，也需要 -1
#     # 因为训练时裁掉了一步 (17 -> 16)，这里 H 必须等于 16
#     H = config.task.horizon
#     bs = config.task.num_envs
#     device = config.optimization.device
#     act_dim = config.task.act_dim

#     episode_rewards = []
#     episode_steps = []
#     episode_success = []

#     # # 预计算目标噪声表，用于冷启动加噪
#     # sigma_min = 1e-4
#     # sigma_max = 0.01
#     # rho=0.9
#     # R = sigma_max / sigma_min
#     # start_idx = config.task.obs_steps - 1
#     # L_eff = max(1.0, float(H - 1 - start_idx))

#     # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     # prog_target = torch.clamp((i_indices - start_idx) / L_eff, 0.0, 1.0)
#     # sigma_target = sigma_min * (R ** prog_target) # (bs, H)

#     # # 预计算冷启动时的稳态噪声分布 (delta = 0)
#     # sigma_min = 1e-3
#     # sigma_max = 0.05
#     # R = sigma_max / sigma_min
#     # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     # sigma_target = sigma_min * (R ** (i_indices / (H - 1))) # (bs, H)


#     # 预计算冷启动用的插值比例
#     sigma_min = 1.5e-4
#     sigma_max = 0.05
#     R = sigma_max / sigma_min

#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     tau_target = torch.clamp(1.0 - i_indices / (H - 1), 0.0, 1.0)

#     for i in range(config.log.eval_episodes // bs):
#         step_reward = []
#         ep_reward = [0.0] * bs
#         # NOTE: update env seed, the original envs is update seed so reset is broken
#         for j in range(len(envs.envs)):
#             envs.envs[j].seed(config.optimization.seed + i * bs + j)
#         obs, _ = envs.reset()
#         t_step = 0

#         # initialize video stream
#         if config.log.save_video:
#             logger.video_init(envs.envs[0], enable=True, video_id=str(i))

#         # Initialize X_curr with Zeros, will be filled by Stationary Padding logic
#         X_curr = torch.zeros((bs, H, act_dim), device=device)
#         is_first_step = True

#         while t_step < config.task.max_episode_steps:
#             # 假设执行后，获得了 obs_batch 和 current_pos: (bs, act_dim)
#             # 1. Process Observation
#             if config.task.obs_type == "state":
#                 obs = obs.astype(np.float32)
#                 obs = dataset.normalizer["obs"]["state"].normalize(obs)
#                 obs_batch = {"state": torch.tensor(obs, device=device, dtype=torch.float32)}
#                 current_pos = obs_batch["state"][:, -1, :2]
#             elif config.task.obs_type == "image":
#                 obs_raw = obs
#                 obs_batch = {}
#                 for k in obs_raw:
#                     # 使用局部变量 val，不修改原 obs 字典
#                     val = obs_raw[k].astype(np.float32)
#                     val = dataset.normalizer["obs"][k].normalize(val)
#                     obs_batch[k] = torch.tensor(val, device=device, dtype=torch.float32)

#                 # Special handling for Current Pos needed for Stationary Padding
#                 current_pos = obs_batch["agent_pos"][:, -1, :]
#             elif config.task.obs_type == "keypoint":
#                 obs_raw = obs.astype(np.float32)
#                 keypoint_obs = obs_raw[:, :, :18]
#                 agent_pos_obs = obs_raw[:, :, 18:20]

#                 nkeypoint = dataset.normalizer["obs"]["keypoint"].normalize(keypoint_obs.reshape(-1, 2)).reshape(bs, config.task.obs_steps, 18)
#                 nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos_obs)

#                 obs_batch = {
#                     "keypoint": torch.tensor(nkeypoint, device=device, dtype=torch.float32),
#                     "agent_pos": torch.tensor(nagent_pos, device=device, dtype=torch.float32)
#                 }
#                 current_pos = obs_batch["agent_pos"][:, -1, :]
#             else:
#                 raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

#             # === 2. 冷启动初始化 (Cold Start) ===
#             if is_first_step:
#                 # # 初始动作：全部填充当前机器人的真实位置
#                 # X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
#                 # # 极其关键：按照合法的目标噪声表，赋予初始 AR 噪声，使系统一开始就在流形附近
#                 # eps = generate_ou_noise(bs, H, act_dim, device, rho=rho)
#                 # X_curr = X_curr + sigma_target.unsqueeze(-1) * eps
#                 # is_first_step = False

#                 # # 用真实的机器状态 Padding (稳妥先验)
#                 # X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)

#                 # # 按照稳态目标 (delta=0) 的 tau 比例混入 AR(1) 噪声
#                 # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#                 # tau_target = torch.clamp(1.0 - i_indices / (H - 1), 0.0, 1.0) # 左端1, 右端0

#                 # # 按照合法的目标噪声表加入局部扰动
#                 # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#                 # prog_target = torch.clamp((i_indices - start_idx) / L_eff, 0.0, 1.0)
#                 # sigma_target = sigma_min * (R ** prog_target)

#                 # eps = generate_ou_noise(bs, H, act_dim, device, rho=0.9)

#                 # 完美插值：左侧保留 current_pos，右侧渐变成纯 eps 噪声
#                 # X_curr = tau_target.unsqueeze(-1) * X_curr + (1.0 - tau_target.unsqueeze(-1)) * eps
#                 # X_curr = X_curr + tau_target.unsqueeze(-1) * eps

#                 # X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
#                 # # 按照稳态目标 tau 比例混入噪声
#                 # i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#                 # tau_target = torch.clamp(1.0 - i_indices / (H - 1), 0.0, 1.0)
#                 # eps = generate_ou_noise(bs, H, act_dim, device, rho=0.9)

#                 # # 左端保留 current_pos，右端渐变为纯噪声
#                 # X_curr = tau_target.unsqueeze(-1) * X_curr + (1.0 - tau_target.unsqueeze(-1)) * eps

#                 # is_first_step = False


#                 # # 均值：用真实的机器当前状态填满整个 Buffer
#                 # X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)

#                 # # 加噪：加入符合稳态目标的微小 OU 噪声
#                 # eps = generate_ou_noise(bs, H, act_dim, device, rho=0.9)
#                 # X_curr = X_curr + sigma_target.unsqueeze(-1) * eps

#                 # is_first_step = False

#                 X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
#                 eps_pure = torch.randn(bs, H, act_dim, device=device) * sigma_max
#                 # 左实右虚，渐变过渡
#                 X_curr = tau_target.unsqueeze(-1) * X_curr + (1.0 - tau_target.unsqueeze(-1)) * eps_pure
#                 is_first_step = False

#             # === 3. 网络预测 (BPv3 Sampler 内部包含了 Shift -> ODE) ===
#             with torch.no_grad():
#                 X_next = agent.sample(
#                     act_0=X_curr,
#                     obs=obs_batch,
#                     num_steps=config.task.act_steps,   #!! 单步
#                     use_ema=True
#                 )

#             # === 4. 执行最左侧极净动作 ===
#             act_normed = X_next[:, 0, :] # 提取 i=0 的动作
#             act_np = act_normed.detach().cpu().numpy()
#             act_un = dataset.normalizer["action"].unnormalize(act_np)

#             if act_un.ndim == 2:
#                 act_un = act_un[:, None, :]

#             obs, reward, terminated, truncated, _ = envs.step(act_un)

#             # # === 4. 标准 DP Slicing 取动作 ===
#             # # 严格按照观测的时间窗口，取出属于当前 t 的有效动作
#             # start = config.task.obs_steps - 1
#             # end = start + config.task.act_steps
#             # act_normed = X_next[:, start:end, :]

#             # act_np = act_normed.detach().cpu().numpy()
#             # act_un = dataset.normalizer["action"].unnormalize(act_np)

#             # obs, reward, terminated, truncated, _ = envs.step(act_un)

#             # === 5. 状态轮转 ===
#             # 将打磨完的 Buffer 交给下一步去 shift
#             X_curr = X_next

#             ep_reward += reward
#             step_reward.append(reward)
#             t_step += 1

#         success = np.around(np.max(np.array(step_reward), axis=0), 2)
#         episode_rewards.append(ep_reward)
#         episode_steps.append(t_step)
#         episode_success.append(success)

#     loguru.logger.info(
#         f"BPv3 Eval | Nstep: {num_steps} Mean step: {np.nanmean(episode_steps):.2f} "
#         f"Mean reward: {np.nanmean(episode_rewards):.4f} "
#         f"Mean success: {np.nanmean(episode_success):.4f}"
#     )

#     metrics = {
#         f"mean_step_{num_steps}": np.nanmean(episode_steps),
#         f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
#         f"mean_success_{num_steps}": np.nanmean(episode_success),
#     }

#     return metrics


def evaluate_bpv3(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Bridge Policy V3 Inference (Rolling Buffer Denoising)."""

    # BPv3 直接预测整个 Horizon，也需要 -1
    # 因为训练时裁掉了一步 (17 -> 16)，这里 H 必须等于 16
    if config.optimization.loss_type == "prcp_v2":
        H = config.task.horizon - 2
    else:
        H = config.task.horizon - 1

    bs = config.task.num_envs
    device = config.optimization.device
    act_dim = config.task.act_dim

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for i in range(config.log.eval_episodes // bs):
        step_reward = []
        ep_reward = [0.0] * bs
        # NOTE: update env seed, the original envs is update seed so reset is broken
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * bs + j)
        obs, _ = envs.reset()
        t_step = 0

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))

        # Initialize X_curr with Zeros, will be filled by Stationary Padding logic
        X_curr = torch.zeros((bs, H, act_dim), device=device)
        is_first_step = True

        while t_step < config.task.max_episode_steps:
            # 假设执行后，获得了 obs_batch 和 current_pos: (bs, act_dim)
            # 1. Process Observation
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs_batch = {"state": torch.tensor(obs, device=device, dtype=torch.float32)}
                current_pos = obs_batch["state"][:, -1, :act_dim]
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs_batch = {}
                for k in obs_raw:
                    # 使用局部变量 val，不修改原 obs 字典
                    val = obs_raw[k].astype(np.float32)
                    val = dataset.normalizer["obs"][k].normalize(val)
                    obs_batch[k] = torch.tensor(val, device=device, dtype=torch.float32)

                # Special handling for Current Pos needed for Stationary Padding
                current_pos = obs_batch["agent_pos"][:, -1, :act_dim]
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)
                keypoint_obs = obs_raw[:, :, :18]
                agent_pos_obs = obs_raw[:, :, 18:20]

                nkeypoint = dataset.normalizer["obs"]["keypoint"].normalize(keypoint_obs.reshape(-1, 2)).reshape(bs, config.task.obs_steps, 18)
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos_obs)

                obs_batch = {
                    "keypoint": torch.tensor(nkeypoint, device=device, dtype=torch.float32),
                    "agent_pos": torch.tensor(nagent_pos, device=device, dtype=torch.float32)
                }
                current_pos = obs_batch["agent_pos"][:, -1, :act_dim]
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            # === 2. 冷启动初始化 (Cold Start) ===
            if is_first_step:
                # # # 初始动作：全部填充当前机器人的真实位置
                # X_curr = current_pos.unsqueeze(1).expand(-1, H, -1)
                # eps_pure = torch.randn(bs, H, act_dim, device=device) * sigma_max
                # # 左实右虚，渐变过渡
                # X_curr = tau_target.unsqueeze(-1) * X_curr + (1.0 - tau_target.unsqueeze(-1)) * eps_pure
                # is_first_step = False

                # simpler and cleaner cold start:
                # initialize the whole chunk from current state/action proxy
                X_curr = current_pos.unsqueeze(1).expand(-1, H, -1).contiguous()

                cold_sigma = getattr(config, "bridge_cold_sigma", 0.0)
                if cold_sigma > 0.0:
                    X_curr = X_curr + cold_sigma * torch.randn_like(X_curr)

                is_first_step = False

            # === 3. 网络预测 (BPv3 Sampler 内部包含了 Shift -> ODE) ===
            with torch.no_grad():
                X_next = agent.sample(
                    act_0=X_curr,
                    obs=obs_batch,
                    num_steps=config.task.act_steps,   #!! 单步
                    use_ema=True
                )

            # === 4. 执行最左侧极净动作 ===
            act_normed = X_next[:, 0, :] # 提取 i=0 的动作
            act_np = act_normed.detach().cpu().numpy()
            act_un = dataset.normalizer["action"].unnormalize(act_np)

            if act_un.ndim == 2:
                act_un = act_un[:, None, :]

            obs, reward, terminated, truncated, _ = envs.step(act_un)

            # # === 4. 标准 DP Slicing 取动作 ===
            # # 严格按照观测的时间窗口，取出属于当前 t 的有效动作
            # start = config.task.obs_steps - 1
            # end = start + config.task.act_steps
            # act_normed = X_next[:, start:end, :]

            # act_np = act_normed.detach().cpu().numpy()
            # act_un = dataset.normalizer["action"].unnormalize(act_np)

            # obs, reward, terminated, truncated, _ = envs.step(act_un)

            # === 5. 状态轮转 ===
            # 将打磨完的 Buffer 交给下一步去 shift
            X_curr = X_next

            ep_reward += reward
            step_reward.append(reward)
            t_step += 1

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t_step)
        episode_success.append(success)

    loguru.logger.info(
        f"BPv3 Eval | Nstep: {num_steps} Mean step: {np.nanmean(episode_steps):.2f} "
        f"Mean reward: {np.nanmean(episode_rewards):.4f} "
        f"Mean success: {np.nanmean(episode_success):.4f}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics


def evaluate_rpv1(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Rolling Policy V1 Inference (Rolling Buffer Denoising)."""

    H = config.task.horizon
    bs = config.task.num_envs
    device = config.optimization.device
    act_dim = config.task.act_dim

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for i in range(config.log.eval_episodes // bs):
        step_reward = []
        ep_reward = [0.0] * bs
        # NOTE: update env seed, the original envs is update seed so reset is broken
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * bs + j)
        obs, _ = envs.reset()
        t_step = 0

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))

        # Initialize X_curr with Zeros, will be filled by Stationary Padding logic
        X_curr = torch.zeros((bs, H, act_dim), device=device)
        is_first_step = True

        while t_step < config.task.max_episode_steps:
            # 假设执行后，获得了 obs_batch 和 current_pos: (bs, act_dim)
            # 1. Process Observation
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs_batch = {"state": torch.tensor(obs, device=device, dtype=torch.float32)}
                current_pos = obs_batch["state"][:, -1, :act_dim]
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs_batch = {}
                for k in obs_raw:
                    # 使用局部变量 val，不修改原 obs 字典
                    val = obs_raw[k].astype(np.float32)
                    val = dataset.normalizer["obs"][k].normalize(val)
                    obs_batch[k] = torch.tensor(val, device=device, dtype=torch.float32)

                # Special handling for Current Pos needed for Stationary Padding
                current_pos = obs_batch["agent_pos"][:, -1, :act_dim]
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)
                keypoint_obs = obs_raw[:, :, :18]
                agent_pos_obs = obs_raw[:, :, 18:20]

                nkeypoint = dataset.normalizer["obs"]["keypoint"].normalize(keypoint_obs.reshape(-1, 2)).reshape(bs, config.task.obs_steps, 18)
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(agent_pos_obs)

                obs_batch = {
                    "keypoint": torch.tensor(nkeypoint, device=device, dtype=torch.float32),
                    "agent_pos": torch.tensor(nagent_pos, device=device, dtype=torch.float32)
                }
                current_pos = obs_batch["agent_pos"][:, -1, :act_dim]
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            # === 2. 冷启动初始化 (Cold Start) ===
            if is_first_step:
                # # # 初始动作：全部填充当前机器人的真实位置
                # simpler and cleaner cold start:
                # initialize the whole chunk from current state/action proxy
                X_curr = current_pos.unsqueeze(1).expand(-1, H, -1).contiguous()

                tau_vec = _build_tau_ladder(
                    T=H,
                    device=device,
                    dtype=torch.float32,
                    tau_min=config.optimization.rolling_tau_min,
                    tau_max=config.optimization.rolling_tau_max,
                    mode=config.optimization.rolling_tau_mode,
                    beta=config.optimization.rolling_tau_beta,
                ).squeeze(0)  # (T,)

                eps = torch.randn(bs, H, act_dim, device=device)
                X_curr = (1.0 - tau_vec.view(1, H, 1)) * X_curr + tau_vec.view(1, H, 1) * eps

                is_first_step = False

            # === 3. 网络预测 (BPv3 Sampler 内部包含了 Shift -> ODE) ===
            with torch.no_grad():
                Y = agent.sample(
                    act_0=X_curr,
                    obs=obs_batch,
                    num_steps=config.task.act_steps,   #!! 单步
                    use_ema=True
                )

            # === 4. 执行最左侧极净动作 ===
            # act_normed = X_next[:, 0, :] # 提取 i=0 的动作

            act_normed = Y[:, 0, :]   # execute clean action
            # X_curr = Y[:, 1:, :]      # next noisy rolling window

            act_np = act_normed.detach().cpu().numpy()
            act_un = dataset.normalizer["action"].unnormalize(act_np)

            if act_un.ndim == 2:
                act_un = act_un[:, None, :]

            obs, reward, terminated, truncated, _ = envs.step(act_un)

            # # === 4. 标准 DP Slicing 取动作 ===
            # # 严格按照观测的时间窗口，取出属于当前 t 的有效动作
            # start = config.task.obs_steps - 1
            # end = start + config.task.act_steps
            # act_normed = X_next[:, start:end, :]

            # act_np = act_normed.detach().cpu().numpy()
            # act_un = dataset.normalizer["action"].unnormalize(act_np)

            # obs, reward, terminated, truncated, _ = envs.step(act_un)

            # === 5. 状态轮转 ===
            # 将打磨完的 Buffer 交给下一步去 shift
            X_curr = Y[:, 1:, :]

            ep_reward += reward
            step_reward.append(reward)
            t_step += 1

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t_step)
        episode_success.append(success)

    loguru.logger.info(
        f"RPv1 Eval | Nstep: {num_steps} Mean step: {np.nanmean(episode_steps):.2f} "
        f"Mean reward: {np.nanmean(episode_rewards):.4f} "
        f"Mean success: {np.nanmean(episode_success):.4f}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics

def sync_runtime_config(config):
    """
    Copy task-level static fields into optimization config so that
    loss/sampler functions (which only receive OptimizationConfig)
    can access them.
    """
    # config.task.loss_type = config.optimization.loss_type

    config.optimization.policy_horizon = config.task.horizon
    config.optimization.roll_chunk_horizon = config.task.horizon - 1
    config.optimization.obs_steps = config.task.obs_steps
    config.optimization.act_steps = config.task.act_steps
    config.optimization.act_dim = config.task.act_dim

    return config

@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    # general config setup
    set_seed(config.optimization.seed)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # sync task fields into optimization config
    config = sync_runtime_config(config)

    # env setup
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    loguru.logger.info("Finished setting up env")

    # dataset setup
    dataset = make_dataset(config.task)
    loguru.logger.info("Finished setting up dataset")

    agent = TrainingAgent(config)
    resume_state = None

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.mode == "train" and config.optimization.auto_resume:
        # Automatically look for checkpoint to resume from
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )

        ## 如果用 BridgePolicy 的时候考虑加入别的参数
        if config.optimization.loss_type == "bridge":
            K = config.optimization.prediction_offset
            H = config.task.horizon - K  # 或者直接读 config.task.horizon
            checkpoint_base_name += f"_H{H}_K{K}"
        elif config.optimization.loss_type == "bridge_v2":
            # Horizon is effectively H+1 for training, but model sees H
            # User sets this to H+1 manually in command line or yaml
            checkpoint_base_name += f"_BPv2"
        elif config.optimization.loss_type == "bridge_v3":
            checkpoint_base_name += f"_BPv3"
        elif config.optimization.loss_type == "prcp_v1":
            checkpoint_base_name += f"_PRCPv1"
        elif config.optimization.loss_type == "prcp_v2":
            checkpoint_base_name += f"_PRCPv2"
        elif config.optimization.loss_type == "rp_v1":
            checkpoint_base_name += f"_RPv1"

        checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
        if checkpoint_path:
            loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            loguru.logger.info("Loading checkpoint with optimizer state...")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        loguru.logger.info("Auto-resume disabled, starting training from scratch")

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()

        # Check for BPv2
        if config.optimization.loss_type == "bridge_v2":
            metrics = {"step": 0}
            metrics.update(evaluate_bpv2(config, envs, dataset, agent, logger))
            logger.log(metrics, category="eval")
            # print result in easy to read format
            for key, val in metrics.items():
                if "mean_success" in key:
                    loguru.logger.info(f"{key} - {val}")
            return # Exit after eval

        num_steps_list = get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(evaluate_bp(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")

        # print result in easy to read format
        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
