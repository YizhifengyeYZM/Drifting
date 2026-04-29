"""Sampler for different training objectives.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import numpy as np
import torch

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.torch_utils import at_least_ndim

##!! BPv3
import math
from mip.losses import (
    dp_ddpm_step,
    dp_get_schedule,
    dp_model_timestep,
    generate_ou_noise,
    mp1_interval_velocity,
)


def get_default_step_list(loss_type: str):
    if loss_type in ["flow", "ctm", "lmd", "psd", "lsd", "esd", "mf"]:
        return 3 ** np.arange(2, -1, -1)        ## [9,3,1]
    elif loss_type == "bridge":
        return [16]
    elif loss_type in ["regression", "mip", "tsd", "straight_flow"]:
        return [1]
    elif loss_type == "mp1":
        return [1]
    elif loss_type in ["dp", "ddpm"]:
        return [100]
    elif loss_type == "naive_drift":
        return [1]
    elif loss_type == "bridge_v2":
        return [1] # BPv2 is one-step inference
    elif loss_type == "bridge_v3":
        return [1]
    elif loss_type == "prcp_v1":
        return [1]
    elif loss_type == "prcp_v2":
        return [1]
    elif loss_type == "rp_v1":
        return [1]
    elif loss_type == "drifting":
        return [1]
    elif loss_type == "drifting2":
        return [1]
    elif loss_type == "drifting3":
        return [1]
    elif loss_type == "drifting4":
        return [1]
    elif loss_type == "drifting5":
        return [1]
    elif loss_type == "drifting6":
        return [1]
    elif loss_type == "globaldiag":
        return [1]
    elif loss_type == "arl":
        return [1]
    elif loss_type == "geofuse":
        return [1]
    elif loss_type == "geofuse_noise":
        return [1]
    elif loss_type == "geofuse_align":
        return [1]
    elif loss_type == "drifting7":
        return [1]
    elif loss_type == "drifting8":
        return [1]
    elif loss_type == "drifting9":
        return [1]
    elif loss_type == "drifting10":
        return [1]
    elif loss_type == "drifting11":
        return [1]
    elif loss_type == "drifting12":
        return [1]
    elif loss_type == "drifting13":
        return [1]
    elif loss_type == "drifting14":
        return [1]
    elif loss_type == "drift6min":
        return [1]
    elif loss_type == "drift6matrix":
        return [1]
    elif loss_type == "drift6min_awshort":
        return [1]
    elif loss_type == "drift6min_rebal":
        return [1]
    elif loss_type == "idp":
        return [1]
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def get_sampler(loss_type: str):
    if loss_type == "flow":
        return ode_sampler
    elif loss_type in ["regression", "straight_flow"]:
        return regression_sampler
    elif loss_type in ["tsd", "mip"]:
        return mip_sampler
    elif loss_type in ["dp", "ddpm"]:
        return dp_sampler
    elif loss_type == "mp1":
        return mp1_sampler
    elif loss_type == "naive_drift":
        return naive_drift_sampler
    elif loss_type in ["lmd", "ctm", "psd", "lsd", "esd", "mf"]:
        return flow_map_sampler
    elif loss_type == "bridge":
        return bridge_sampler
    elif loss_type == "bridge_v2":
        return bridge_v2_sampler
    elif loss_type == "bridge_v3":
        return bridge_v3_sampler
    elif loss_type == "prcp_v1":
        return prcp_sampler
    elif loss_type == "prcp_v2":
        return prcp_sampler
    elif loss_type == "rp_v1":
        return rolling_policy_sampler
    elif loss_type == "drifting":
        return drifting_policy_sampler
    elif loss_type == "drifting2":
        return drifting_policy_sampler2
    elif loss_type == "drifting3":
        return drifting_policy_sampler
    elif loss_type == "drifting4":
        return drifting_policy_sampler4
    elif loss_type == "drifting5":
        return drifting_policy_sampler4
    elif loss_type == "drifting6":
        return drifting_policy_sampler4
    elif loss_type == "globaldiag":
        return drifting_policy_sampler4
    elif loss_type == "arl":
        return drifting_policy_sampler4
    elif loss_type == "geofuse":
        return drifting_policy_sampler4
    elif loss_type == "geofuse_noise":
        return geofuse_noise_sampler
    elif loss_type == "geofuse_align":
        return geofuse_align_sampler
    elif loss_type == "drifting7":
        return drifting_policy_sampler7
    elif loss_type == "drifting8":
        return drifting_policy_sampler8
    elif loss_type == "drifting9":
        return drifting_policy_sampler9
    elif loss_type == "drifting10":
        return drifting_policy_sampler10
    elif loss_type == "drifting11":
        return drifting_policy_sampler11
    elif loss_type == "drifting12":
        return drifting_policy_sampler12
    elif loss_type == "drifting13":
        return drifting_policy_sampler13
    elif loss_type == "drifting14":
        return drifting_policy_sampler14
    elif loss_type == "drift6min":
        return drifting_policy_sampler4
    elif loss_type == "drift6matrix":
        return drifting_policy_sampler4
    elif loss_type == "drift6min_awshort":
        return drifting_policy_sampler4
    elif loss_type == "drift6min_rebal":
        return drifting_policy_sampler4
    elif loss_type == "idp":
        return implicit_drifting_sampler
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def implicit_drifting_sampler(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drifting Policy sampler.
    Deployment is strictly one-step.
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)

    obs_emb = encoder(obs, None)
    act_in = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_in, obs_emb)
    return act_pred_0

def ode_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    num_steps = config.num_steps
    sample_mode = config.sample_mode
    t_schedule = np.linspace(0, 1, num_steps + 1)
    if sample_mode == "stochastic":
        act_s = torch.randn_like(act_0, device=act_0.device)
    else:
        act_s = torch.zeros_like(act_0, device=act_0.device)
    obs_emb = encoder(obs, None)
    bs = act_0.shape[0]
    for i in range(num_steps):
        s_val = t_schedule[i]
        t_val = t_schedule[i + 1]
        s = torch.full((bs,), s_val, device=act_0.device)
        t = torch.full((bs,), t_val, device=act_0.device)
        b_s = flow_map.get_velocity(s, act_s, obs_emb)
        s_expanded = at_least_ndim(s, act_s.dim())
        t_expanded = at_least_ndim(t, act_s.dim())
        act_s = act_s + b_s * (t_expanded - s_expanded)
    act = act_s
    return act


def flow_map_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """This function is designed for flow map sampler, i.e. for the distilled shortcut model.

    Args:
        config (OptimizationConfig): the configuration
        flow_map (FlowMap): the flow map
        encoder (BaseEncoder): the encoder
        act_0 (torch.Tensor): the initial action
        obs (torch.Tensor): the observation

    Returns:
        torch.Tensor: the sampled action
    """
    num_steps = config.num_steps
    sample_mode = config.sample_mode
    t_schedule = np.linspace(0, 1, num_steps + 1)
    if sample_mode == "stochastic":
        act_s = torch.randn_like(act_0, device=act_0.device)
    else:
        act_s = torch.zeros_like(act_0, device=act_0.device)
    obs_emb = encoder(obs, None)
    bs = act_0.shape[0]
    for i in range(num_steps):
        s_val = t_schedule[i]
        t_val = t_schedule[i + 1]
        s = torch.full((bs,), s_val, device=act_0.device)
        t = torch.full((bs,), t_val, device=act_0.device)
        act_s = flow_map(s, t, act_s, obs_emb)
    act = act_s
    return act


def regression_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    bs = act_0.shape[0]
    act_zeros = torch.zeros_like(act_0, device=act_0.device)
    t = torch.zeros(bs, device=act_0.device)
    obs_emb = encoder(obs, None)
    act = flow_map.get_velocity(t, act_zeros, obs_emb)
    return act


def dp_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Diffusion Policy DDPM sampler. Default eval NFE is 100."""
    obs_emb = encoder(obs, None)
    batch_size = act_0.shape[0]
    sample = torch.randn_like(act_0, device=act_0.device)

    schedule = dp_get_schedule(config, device=act_0.device, dtype=act_0.dtype)
    num_train_timesteps = int(getattr(config, "dp_num_train_timesteps", 100))
    configured_steps = int(getattr(config, "dp_num_inference_steps", 100))
    num_steps = int(getattr(config, "num_steps", configured_steps))
    if num_steps <= 1 and configured_steps > 1:
        num_steps = configured_steps
    num_steps = max(1, min(num_steps, num_train_timesteps))

    timesteps = torch.linspace(
        num_train_timesteps - 1,
        0,
        num_steps,
        device=act_0.device,
        dtype=torch.float32,
    ).round().long()

    for i, timestep_scalar in enumerate(timesteps):
        prev_scalar = timesteps[i + 1] if i + 1 < len(timesteps) else timestep_scalar.new_tensor(-1)
        timestep = torch.full(
            (batch_size,),
            int(timestep_scalar.item()),
            device=act_0.device,
            dtype=torch.long,
        )
        prev_timestep = torch.full(
            (batch_size,),
            int(prev_scalar.item()),
            device=act_0.device,
            dtype=torch.long,
        )
        t_model = dp_model_timestep(config, timestep, dtype=act_0.dtype)
        model_output = flow_map.get_velocity(t_model, sample, obs_emb)
        sample = dp_ddpm_step(
            config,
            model_output=model_output,
            timestep=timestep,
            prev_timestep=prev_timestep,
            sample=sample,
            schedule=schedule,
        )

    return sample


def mp1_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """MP1 one-step sampler: x_0 = x_1 - u_theta(x_1, r=0, t=1 | obs)."""
    batch_size = act_0.shape[0]
    obs_emb = encoder(obs, None)
    noise_scale = float(getattr(config, "mp1_noise_scale", 1.0))
    act_t = torch.randn_like(act_0, device=act_0.device) * noise_scale
    r = torch.zeros((batch_size,), device=act_0.device, dtype=act_0.dtype)
    t = torch.ones((batch_size,), device=act_0.device, dtype=act_0.dtype)
    u = mp1_interval_velocity(flow_map, act_t, r, t, obs_emb)
    return act_t - u


def naive_drift_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """One-forward Drifting sampler: standard noise to raw action chunk."""
    del config
    bs = act_0.shape[0]
    t0 = torch.zeros((bs,), device=act_0.device)
    obs_emb = encoder(obs, None)
    noise = torch.empty_like(act_0, device=act_0.device).normal_(0, 1)
    return flow_map.get_velocity(t0, noise, obs_emb)


def mip_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Simplified minimum iterative policy
    Note that now the first step prediction is not scaled by t_two_step,
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)
    t = torch.full((bs,), config.t_two_step, device=act_0.device)

    obs_emb = encoder(obs, None)

    act_0 = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    # NOTE: now the first step prediction is not scaled by t_two_step,
    # if you want the original form, you can use the mip_origin_sampler
    act_pred_1 = flow_map.get_velocity(t, act_pred_0, obs_emb)

    act = act_pred_1
    return act


def drifting_policy_sampler2(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Absolute Single-Step Inference!
    抛弃了任何迭代循环，直接从纯噪声映射出高质量 Action Chunk。
    """
    bs = act_0.shape[0]

    # 推断时直接固定起始时间 t = 0
    s = torch.zeros((bs,), device=act_0.device)

    # 提取观测条件
    obs_emb = encoder(obs, None)

    # 多模态的核心：输入标准高斯噪声！它作为“随机种子”在空间中选定模态
    noise = torch.empty_like(act_0, device=act_0.device).normal_(0, 1)

    # 仅需 1 次前向传播 (1-Step Flow Map)
    act_pred = flow_map.get_velocity(s, noise, obs_emb)
    # act_pred = noise + pred_velocity

    return act_pred

def drifting_policy_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Absolute Single-Step Inference!
    抛弃了任何迭代循环，直接从纯噪声映射出高质量 Action Chunk。
    """
    bs = act_0.shape[0]

    # 推断时直接固定起始时间 t = 0
    s = torch.zeros((bs,), device=act_0.device)

    # 提取观测条件
    obs_emb = encoder(obs, None)

    # 多模态的核心：输入标准高斯噪声！它作为“随机种子”在空间中选定模态
    noise = torch.empty_like(act_0, device=act_0.device).normal_(0, 1)

    # 仅需 1 次前向传播 (1-Step Flow Map)
    pred_velocity = flow_map.get_velocity(s, noise, obs_emb)
    act_pred = noise + pred_velocity

    return act_pred


def drifting_policy_sampler4(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drifting Policy sampler.
    Deployment is strictly one-step.
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)

    obs_emb = encoder(obs, None)
    act_in = torch.zeros_like(act_0, device=act_0.device)
    # act_in = torch.empty_like(act_0, device=act_0.device).normal_(0, 1)   ## 4.26：random start

    act_pred_0 = flow_map.get_velocity(s, act_in, obs_emb)
    return act_pred_0


def drifting_policy_sampler7(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift7 sampler:
    strict one-step, zero-start coarse deployment.
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)

    obs_emb = encoder(obs, None)
    act_in = torch.zeros_like(act_0, device=act_0.device)

    act_pred_0 = flow_map.get_velocity(s, act_in, obs_emb)
    return act_pred_0


def drifting_policy_sampler8(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift8 sampler:
    strict one-step, zero-start coarse deployment.
    """
    return drifting_policy_sampler7(config, flow_map, encoder, act_0, obs)


def drifting_policy_sampler9(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift9 sampler:
    strict one-step, zero-start coarse deployment.
    """
    return drifting_policy_sampler7(config, flow_map, encoder, act_0, obs)


def drifting_policy_sampler10(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift10 sampler:
    strict one-step, zero-start coarse deployment.
    """
    return drifting_policy_sampler7(config, flow_map, encoder, act_0, obs)


def drifting_policy_sampler11(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift11 sampler:
    strict one-step, zero-start coarse deployment.
    """
    return drifting_policy_sampler7(config, flow_map, encoder, act_0, obs)


def drifting_policy_sampler12(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift12 sampler:
    strict one-step, zero-start coarse deployment.
    """
    return drifting_policy_sampler7(config, flow_map, encoder, act_0, obs)


def drifting_policy_sampler13(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift13 sampler:
    strict one-step generation from a long-step field.
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)

    obs_emb = encoder(obs, None)
    act_in = torch.zeros_like(act_0, device=act_0.device)
    field_pred_0 = flow_map.get_velocity(s, act_in, obs_emb)
    act_pred_0 = act_in + config.t_two_step * field_pred_0
    return act_pred_0


def drifting_policy_sampler14(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Drift14 sampler:
    strict one-step deployment from the explicit proposal head.
    """
    return drifting_policy_sampler7(config, flow_map, encoder, act_0, obs)

def geofuse_align_sampler(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    GeoFuse-Align sampler:
    strict one-step, zero-start
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)

    obs_emb = encoder(obs, None)
    act_in = torch.zeros_like(act_0, device=act_0.device)

    act_pred_0 = flow_map.get_velocity(s, act_in, obs_emb)
    return act_pred_0


def geofuse_noise_sampler(
    config,
    flow_map,
    encoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    GeoFuse-Noise sampler:
    strict one-step, noise-start
    train/infer are matched: both start from Gaussian noise
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)

    obs_emb = encoder(obs, None)
    act_in = torch.empty_like(act_0, device=act_0.device).normal_(0, 1)

    act_pred_0 = flow_map.get_velocity(s, act_in, obs_emb)
    return act_pred_0

def mip_origin_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Original minimum iterative policy"""
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)
    t = torch.full((bs,), config.t_two_step, device=act_0.device)
    obs_emb = encoder(obs, None)

    act_0 = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    # this is the original form in the paper
    act_pred_1 = flow_map.get_velocity(t, act_pred_0 * config.t_two_step, obs_emb)

    act = act_pred_1
    return act


def bridge_sampler_old(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Generalized Bridge Policy Sampler (Streaming & Chunking).

    Args:
        act_0: Dummy input for shape and device.
    """
    # 1. Initialize from Observation (Construct Initial Window)
    bs = act_0.shape[0]
    Ta = act_0.shape[1]
    act_dim = act_0.shape[-1]
    device = act_0.device

    # # Get current agent position (point)
    # if isinstance(obs, dict):
    #     if "agent_pos" in obs:
    #         curr_pos = obs["agent_pos"][:, -1, :]  # (B, D)
    #     elif "state" in obs:
    #         curr_pos = obs["state"][:, -1, :2]  # (B, D) assuming first 2 are pos
    #     else:
    #         curr_pos = torch.zeros((bs, act_dim), device=device)
    # else:
    #     curr_pos = torch.zeros((bs, act_dim), device=device)

    # # Repeat Pad to construct initial Window (B, Ta, D)
    # ##!! 不是 Ta，Ta 现在变成 W 了，这里得是 H！得是 16！
    # H = 16
    # x_curr = curr_pos.unsqueeze(1).expand(-1, H, -1)

    obs_emb = encoder(obs, None)

    # 2. Inference Mode
    if config.sample_mode == "streaming":
        # === Closed-loop Streaming (Mode A) ===
        # Single step planning at t=0

        # Predict velocity field v(x_curr, t=0)
        t = torch.zeros((bs,), device=device)
        v_seq = flow_map.get_velocity(t, act_0, obs_emb)

        # We need the sliding velocity.
        # Since x_curr is constant across time dim (repeated),
        # v_seq should also be roughly constant or we just take the first component.
        # Logic: v_seq represents the displacement field for the whole window.
        # v = v_seq[:, 0, :]  # (B, D)

        # Integrate: Slide the window forward by 1 physics step
        # v corresponds to displacement over K steps (prediction_offset)
        # So for 1 step, we scale by 1/K
        K = getattr(config, "prediction_offset", 4)
        dt = 1.0 / K

        act = act_0 + v_seq * dt

        # Return format: (B, Ta, D) filled with x_next_point
        # This allows the evaluator to slice it however it wants
        # act = x_next_point.unsqueeze(1).expand(-1, H, -1)

    else:
        # === Open-loop Chunking (Mode B) ===
        # Integrate from t=0 to t=1 to generate a full K-step transition?
        # Or standard flow matching integration?

        # In Generalized Bridge, we learned the flow from Window_t to Window_{t+K}
        # If we integrate t from 0 to 1, we transform the Current Window to the Future Window.
        # But for "Chunking" execution, we typically want the trajectory points *within* this window?
        # Actually, if we view the window as a trajectory, transforming it gives us the trajectory K steps later.

        # Let's stick to the standard ODE integration logic which matches the training objective.
        # We integrate the flow to get x_1 (The future window).

        H = config.num_steps
        t_schedule = np.linspace(0, 1, H + 1)
        x_t = act_0

        for i in range(H):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((bs,), s_val, device=device)
            t = torch.full((bs,), t_val, device=device)

            # Predict velocity
            v_seq = flow_map.get_velocity(s, x_t, obs_emb)

            # Euler step (or others)
            dt = t_val - s_val
            x_t = x_t + v_seq * dt

        act = x_t

    return act


def bridge_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    tau: float = 0.0,
    **kwargs
):
    """Bridge Policy One-Step Integrator (for custom evaluation loop).

    In BP, the sampling loop is handled externally (Streaming).
    This function performs the micro-step integration:
    X_next = X_curr + v_theta(X_curr, tau, obs) * dtau

    Args:
        X_curr: Current action chunk state (B, H, D)
        obs: Current observation
        tau: Current flow time [0, 1]
    """
    # Prediction Horizon N
    T_pred = config.prediction_offset if hasattr(config, "prediction_offset") else 16

    # Flow step size dtau = 1 / N
    dtau = 1.0 / T_pred

    bs = X_curr.shape[0]
    device = X_curr.device

    # Prepare tau tensor
    tau_tensor = torch.full((bs,), tau, device=device)

    # 1. Encode Observation
    obs_emb = encoder(obs, None)

    # 2. Predict Velocity
    # v_pred = v_theta(X_curr, tau, h_curr)
    v_pred = flow_map.get_velocity(tau_tensor, X_curr, obs_emb)

    # 3. Micro-step Integration (Euler)
    X_next = X_curr + v_pred * dtau

    return X_next


def bridge_v2_sampler_old(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float = 1.0, # 1.0 = Baseline (Visual only)
    **kwargs
):
    """Bridge Policy V2 Sampler with CFG support.

    Refinement: X_{t+1} = X_curr + (Prediction - X_curr) ?
    No, BPv2 predicts X_{t+1} directly.
    """
    bs = X_curr.shape[0]
    device = X_curr.device

    # Encode Obs
    obs_emb = encoder(obs, None)

    # 1. Conditional Forward (Visual) -> t=1
    t_cond = torch.ones((bs,), device=device)
    X_cond = flow_map.get_velocity(t_cond, X_curr, obs_emb)

    if guidance_scale == 1.0:
        return X_cond

    # 2. Unconditional Forward (Blind) -> t=0
    t_uncond = torch.zeros((bs,), device=device)
    # Zero out obs
    obs_emb_null = torch.zeros_like(obs_emb)
    X_uncond = flow_map.get_velocity(t_uncond, X_curr, obs_emb_null)

    # 3. Combine
    # Formula: Uncond + w * (Cond - Uncond)
    X_next = X_uncond + guidance_scale * (X_cond - X_uncond)

    return X_next

def bridge_v2_sampler_old2(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float = 1.0,
    **kwargs
):
    """Bridge Policy V2 Sampler (Residual Mode)."""
    bs = X_curr.shape[0]
    device = X_curr.device

    # Encode Obs
    obs_emb = encoder(obs, None)

    # 1. Conditional Forward (Visual) -> t=1
    t_cond = torch.ones((bs,), device=device)
    # Network predicts Residual: (X_curr - X_next)
    res_cond = flow_map.get_velocity(t_cond, X_curr, obs_emb)

    if guidance_scale == 1.0:
        # X_next = X_in - Residual
        return X_curr - res_cond

    # 2. Unconditional Forward (Blind) -> t=0
    t_uncond = torch.zeros((bs,), device=device)
    obs_emb_null = torch.zeros_like(obs_emb)
    res_uncond = flow_map.get_velocity(t_uncond, X_curr, obs_emb_null)

    # 3. Combine Residuals (CFG on Residual Space)
    # res_final = res_uncond + w * (res_cond - res_uncond)
    res_final = res_uncond + guidance_scale * (res_cond - res_uncond)

    # 4. Update
    X_next = X_curr - res_final

    return X_next


def bridge_sampler_old2(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Generalized Bridge Policy Sampler (Streaming & Chunking).

    Args:
        act_0: Dummy input for shape and device.
    """
    # 1. Initialize from Observation (Construct Initial Window)
    bs = act_0.shape[0]
    Ta = act_0.shape[1]
    act_dim = act_0.shape[-1]
    device = act_0.device

    # Get current agent position (point)
    # NOTE: In Streaming mode, act_0 is the previous prediction (Warm Start)
    # But we still need the current observation to construct the "Flat Anchor"
    if isinstance(obs, dict):
        if "agent_pos" in obs:
            curr_pos = obs["agent_pos"][:, -1, :]  # (B, D)
        elif "state" in obs:
            curr_pos = obs["state"][:, -1, :2]  # (B, D) assuming first 2 are pos
        else:
            curr_pos = torch.zeros((bs, act_dim), device=device)
    else:
        curr_pos = torch.zeros((bs, act_dim), device=device)

    # Repeat Pad to construct Flat Anchor Window (B, Ta, D)
    # H = Ta (which is usually 16)
    x_curr = curr_pos.unsqueeze(1).expand(-1, Ta, -1)

    obs_emb = encoder(obs, None)

    # 2. Inference Mode
    if config.sample_mode == "streaming":
        # === Closed-loop Streaming (Leaky Warm Start) ===

        # [关键] 我们需要拿到上一步的预测结果 (act_0)
        # 这里的 act_0 是 evaluate_bp 传进来的，代表 X_prev

        # 1. Leaky Update (构造输入)
        # alpha = 0.8 (保留 80% 历史，20% 修正为当前观测)
        # 这个 alpha 应该和训练时的分布匹配
        alpha = 0.8

        # x_curr 是纯平的 Flat Anchor (从 obs 构造的)
        # act_0 是上一步的预测 (Warm Start)
        x_in = alpha * act_0 + (1 - alpha) * x_curr

        # 2. Predict (t=0)
        t = torch.zeros((bs,), device=device)
        v_seq = flow_map.get_velocity(t, x_in, obs_emb)

        # 3. Integrate
        K = getattr(config, "prediction_offset", 4)
        dt = 1.0 / K

        # 这里的积分基准应该是 x_in
        # 因为网络预测的是从 x_in 出发怎么走
        act = x_in + v_seq * dt

    else:
        # === Open-loop Chunking (Mode B) ===
        # Integrate from t=0 to t=1 to generate a full K-step transition?
        # Or standard flow matching integration?

        # In Generalized Bridge, we learned the flow from Window_t to Window_{t+K}
        # If we integrate t from 0 to 1, we transform the Current Window to the Future Window.
        # But for "Chunking" execution, we typically want the trajectory points *within* this window?
        # Actually, if we view the window as a trajectory, transforming it gives us the trajectory K steps later.

        # Let's stick to the standard ODE integration logic which matches the training objective.
        # We integrate the flow to get x_1 (The future window).

        H = config.num_steps
        t_schedule = np.linspace(0, 1, H + 1)
        x_t = act_0

        for i in range(H):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((bs,), s_val, device=device)
            t = torch.full((bs,), t_val, device=device)

            # Predict velocity
            v_seq = flow_map.get_velocity(s, x_t, obs_emb)

            # Euler step (or others)
            dt = t_val - s_val
            x_t = x_t + v_seq * dt

        act = x_t

    return act


def bridge_v2_sampler_0221(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float = 1.0, # 1.0 = Baseline (Visual only)
    **kwargs
):
    """Bridge Policy V2 Sampler with CFG support.

    Refinement: X_{t+1} = X_curr + (Prediction - X_curr) ?
    No, BPv2 predicts X_{t+1} directly.
    """
    bs = X_curr.shape[0]
    device = X_curr.device

    # Encode Obs
    obs_emb = encoder(obs, None)

    sigma_inf = torch.tensor(0.5, device=device)

    # Calculate Coefficients for Inference
    c_in = 1 / (sigma_inf ** 2 + 1).sqrt()
    c_skip = 1 / (sigma_inf ** 2 + 1)
    c_out = sigma_inf / (sigma_inf ** 2 + 1).sqrt()

    # Prepare Input
    X_net_in = X_curr * c_in

    # 1. Conditional Forward (Visual) -> t=1
    t_cond = torch.ones((bs,), device=device)
    F_cond = flow_map.get_velocity(t_cond, X_net_in, obs_emb)

    if guidance_scale == 1.0:
        X_cond = c_skip * X_curr + c_out * F_cond
        return X_cond

    # 2. Unconditional Forward (Blind) -> t=0
    t_uncond = torch.zeros((bs,), device=device)
    # Zero out obs
    obs_emb_null = torch.zeros_like(obs_emb)
    F_uncond = flow_map.get_velocity(t_uncond, X_net_in, obs_emb_null)

    # 3. Combine
    # Formula: Uncond + w * (Cond - Uncond)
    F_final = F_uncond + guidance_scale * (F_cond - F_uncond)

    # 4. Reconstruct
    X_next = c_skip * X_curr + c_out * F_final

    return X_next


def bridge_v2_sampler0309(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float = 1.0, # 1.0 = Baseline (Visual only)
    **kwargs
):
    """Bridge Policy V2 Sampler with CFG support.

    Refinement: X_{t+1} = X_curr + (Prediction - X_curr) ?
    No, BPv2 predicts X_{t+1} directly.
    """
    bs = X_curr.shape[0]
    device = X_curr.device

    # X_shifted = torch.cat([X_curr[:, 1:], X_curr[:, -1:]], dim=1)

    # ##!! Plan2: RTC 的梯度引导法
    # X_shifted.requires_grad_(True)

    # Encode Obs
    obs_emb = encoder(obs, None)

    # 1. Conditional Forward (Visual) -> t=1
    t_cond = torch.ones((bs,), device=device)

    ##** 原方案
    X_cond = flow_map.get_velocity(t_cond, X_curr, obs_emb)

    ##** 采两遍
    # X_cond1 = flow_map.get_velocity(t_cond, X_curr, obs_emb)
    # X_cond = flow_map.get_velocity(t_cond, X_cond1, obs_emb)

    if guidance_scale == 1.0:
        # ##!! Plan 1：尝试 EMA 结果
        # alpha = 0.8
        # X_cond = alpha * X_cond + (1 - alpha) * X_shifted

        # ##!! Plan 2: RTC 梯度引导
        # loss = torch.mean((X_cond[:, 0, :] - X_shifted[:, 0, :].detach()) ** 2)
        # # Compute Gradient w.r.t Input (X_shifted)
        # grads = torch.autograd.grad(loss, X_shifted)[0]

        # # Refine Input: Move X_shifted to minimize jump
        # # eta = 0.1 (Step size)
        # X_cond = X_shifted - 0.1 * grads


        return X_cond

    # 2. Unconditional Forward (Blind) -> t=0
    t_uncond = torch.zeros((bs,), device=device)
    # Zero out obs
    obs_emb_null = torch.zeros_like(obs_emb)

    ##** 原方案
    X_uncond = flow_map.get_velocity(t_uncond, X_curr, obs_emb_null)

    ##** 采两遍
    # X_uncond1 = flow_map.get_velocity(t_uncond, X_curr, obs_emb_null)
    # X_uncond = flow_map.get_velocity(t_uncond, X_uncond1, obs_emb_null)


    # 3. Combine
    # Formula: Uncond + w * (Cond - Uncond)
    X_next = X_uncond + guidance_scale * (X_cond - X_uncond)

    return X_next

def bridge_v2_sampler03092(
    config, # OptimizationConfig
    flow_map, # FlowMap
    encoder, # BaseEncoder
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float = 1.0, # 1.0 = Baseline (Visual only)
    **kwargs
):
    """Bridge Policy V2 Sampler with CFG support.

    Refinement: X_{t+1} = X_curr + V_pred
    (Self-Forcing: The network predicts the residual/jump to the next state)
    """
    bs = X_curr.shape[0]
    device = X_curr.device

    # Encode Obs
    obs_emb = encoder(obs, None)

    # 1. Conditional Forward (Visual) -> t=1
    t_cond = torch.ones((bs,), device=device)

    # 网络现在预测的是 Residual (Velocity)
    V_cond = flow_map.get_velocity(t_cond, X_curr, obs_emb)

    if guidance_scale == 1.0:
        V_pred = V_cond
    else:
        # 2. Unconditional Forward (Blind) -> t=0
        t_uncond = torch.zeros((bs,), device=device)
        obs_emb_null = torch.zeros_like(obs_emb)
        V_uncond = flow_map.get_velocity(t_uncond, X_curr, obs_emb_null)

        # 3. Combine CFG
        # Formula: Uncond + w * (Cond - Uncond)
        V_pred = V_uncond + guidance_scale * (V_cond - V_uncond)

    # ==========================
    # 4. State Update (核心修改区)
    # ==========================
    # 因为我们在 Loss 中学的是 V_target = X_next - X_in
    # 所以在推理时，X_next = X_curr + V_pred
    # 这个加法同时完成了 "去噪" 和 "Chunk 平移" 的动作！
    X_next = X_curr + V_pred

    return X_next

def bridge_v2_sampler(
    config, flow_map, encoder,
    X_curr: torch.Tensor,
    obs: torch.Tensor,
    # guidance_scale: float = 1.0,
    **kwargs
):
    """Bridge Policy V2 Sampler — CDP version."""
    bs = X_curr.shape[0]
    device = X_curr.device
    obs_emb = encoder(obs, None)
    guidance_scale = getattr(config, "guidance_scale", 1.0)

    # Conditional
    t_cond = torch.ones((bs,), device=device)

    # sigma_inf = 0
    # t_cond = torch.full((bs,), sigma_inf, device=device)


    out_cond = flow_map.get_velocity(t_cond, X_curr, obs_emb)

    if guidance_scale == 1.0:
        out = out_cond
    else:
        # Unconditional
        t_uncond = torch.zeros((bs,), device=device)
        obs_null = torch.zeros_like(obs_emb)
        out_uncond = flow_map.get_velocity(t_uncond, X_curr, obs_null)
        out = out_uncond + guidance_scale * (out_cond - out_uncond)

    # ---------- MODE A: Direct Prediction ----------
    X_next = out

    # ---------- MODE B: Clean Velocity ----------
    # X_shifted = torch.cat([X_curr[:, 1:, :], X_curr[:, -1:, :]], dim=1)
    # X_next = X_shifted + out

    # blend_beta = 1.0

    # if blend_beta >= 1.0:
    #     X_next = X_raw
    # else:
    #     # 惯性预测：上一个 chunk 平移一步，末尾用网络预测填充
    #     X_shifted = torch.cat([X_curr[:, 1:, :], X_raw[:, -1:, :]], dim=1)
    #     X_next = (1 - blend_beta) * X_shifted + blend_beta * X_raw


    return X_next




# def bridge_v3_sampler(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     act_0: torch.Tensor,
#     obs: torch.Tensor,
# ):
#     """Bridge Policy V3 Sampler — Rolling & ODE Pull."""

#     X_curr = act_0  # 从评估环境传进来的 Rolling Buffer
#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     ##!! 这里不是直接读 yaml 或者启动命令，是外面 eval 的时候已经传进来了
#     ##!! 当时就是把 config.task.act_steps 传给 num_steps 的
#     act_steps = config.num_steps
#     obs_steps = 2
#     start_idx = obs_steps - 1  #!!! 极其关键：这是要执行的动作索引

#     # === BPv3 核心超参数 ===
#     sigma_min = 1e-4
#     sigma_max = 0.01
#     rho=0.9
#     R = sigma_max / sigma_min
#     L_eff = max(1.0, float(H - 1 - start_idx))

#     # BPv3-Lite 默认不加 Churning，可以通过 config 开启
#     gamma = getattr(config, "bpv3_gamma", 0.0)
#     use_heun = getattr(config, "bpv3_use_heun", False)

#     # ==========================
#     # 动作 1: 物理平移 (Shift)
#     # ==========================
#     X_shift = torch.zeros_like(X_curr)


#     # X_shift[:, :-1, :] = X_curr[:, 1:, :]
#     # # 最右侧填入巨大噪声，以最后一个有效动作为锚点
#     # X_shift[:, -1, :] = X_curr[:, -1, :] + sigma_max * torch.randn(bs, D, device=device)

#     if act_steps < H:
#         X_shift[:, :-act_steps, :] = X_curr[:, act_steps:, :]
#         # 稳妥的 Padding：用上一个动作延伸
#         X_shift[:, -act_steps:, :] = X_curr[:, -1:, :].expand(-1, act_steps, -1)
#     else:
#         X_shift = X_curr[:, -1:, :].expand(-1, H, -1)

#     # 物理平移在数学上，意味着所有槽位的相对噪声层级都退后了一档 (即 delta=1.0)
#     delta_curr_val = float(act_steps)

#     # ==========================
#     # 动作 2: Langevin Churning (可选)
#     # ==========================
#     if gamma > 0:
#         delta_churn = (H - 1) * math.log(1 + gamma) / math.log(R)
#         delta_curr_val += delta_churn

#         # 精确计算需要增加的物理噪声量
#         i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)


#         # sigma_base = sigma_min * (R ** ((i_indices + 1.0) / (H - 1)))
#         # sigma_hat = sigma_min * (R ** ((i_indices + delta_curr_val) / (H - 1)))

#         prog_base = torch.clamp((i_indices + 1.0 - start_idx) / L_eff, 0.0, 1.0)
#         sigma_base = sigma_min * (R ** prog_base)

#         prog_hat = torch.clamp((i_indices + delta_curr_val - start_idx) / L_eff, 0.0, 1.0)
#         sigma_hat = sigma_min * (R ** prog_hat)

#         noise_to_add = torch.sqrt(torch.clamp(sigma_hat**2 - sigma_base**2, min=0.0))
#         eps_churn = generate_ou_noise(bs, H, D, device, rho=rho)

#         X_churn = X_shift + noise_to_add.unsqueeze(-1) * eps_churn
#     else:
#         X_churn = X_shift

#     # ==========================
#     # 动作 3: 精确 ODE/SDE 拉回
#     # ==========================
#     obs_emb = encoder(obs, None)

#     # 告诉网络当前的混乱度：传标量 delta_curr_val
#     delta_tensor = torch.full((bs,), delta_curr_val, device=device)
#     X_clean = flow_map.get_velocity(delta_tensor, X_churn, obs_emb)

#     # 计算当前与目标的数学噪声水平
#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)


#     # sigma_curr = sigma_min * (R ** ((i_indices + delta_curr_val) / (H - 1)))
#     # sigma_target = sigma_min * (R ** (i_indices / (H - 1)))

#     prog_curr = torch.clamp((i_indices + delta_curr_val - start_idx) / L_eff, 0.0, 1.0)
#     sigma_curr = sigma_min * (R ** prog_curr)

#     prog_target = torch.clamp((i_indices + 0.0 - start_idx) / L_eff, 0.0, 1.0)
#     sigma_target = sigma_min * (R ** prog_target)

#     # Variance Exploding 概率流 ODE 步进公式: dx = d_sigma * (x - X0) / sigma
#     step_size = sigma_target - sigma_curr # MUST BE NEGATIVE
#     grad = (X_churn - X_clean) / sigma_curr.unsqueeze(-1)

#     if not use_heun:
#         # 防线3-Lite: 标准一阶 Euler
#         X_next = X_churn + step_size.unsqueeze(-1) * grad
#     else:
#         # 防线3-Pro: 二阶 Heun (2 NFE)
#         X_tmp = X_churn + step_size.unsqueeze(-1) * grad
#         delta_target_tensor = torch.zeros((bs,), device=device) # 目标是 delta=0
#         X_clean_2 = flow_map.get_velocity(delta_target_tensor, X_tmp, obs_emb)
#         grad_2 = (X_tmp - X_clean_2) / sigma_target.unsqueeze(-1)
#         X_next = X_churn + step_size.unsqueeze(-1) * (grad + grad_2) / 2.0

#     return X_next




# def bridge_v3_sampler(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     act_0: torch.Tensor,
#     obs: torch.Tensor,
# ):
#     """Bridge Policy V3 Sampler — Generic Rolling for Flow Matching."""

#     X_curr = act_0
#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     ##!! 这里不是直接读 yaml 或者启动命令，是外面 eval 的时候已经传进来了
#     ##!! 当时就是把 config.task.act_steps 传给 num_steps 的
#     act_steps = config.num_steps

#     # ==========================
#     # 动作 1: 物理平移 (Shift)
#     # ==========================
#     X_shift = torch.zeros_like(X_curr)
#     if act_steps < H:
#         X_shift[:, :-act_steps, :] = X_curr[:, act_steps:, :]
#         # 听你的求稳建议：用最后一帧兜底 Padding
#         X_shift[:, -act_steps:, :] = X_curr[:, -1:, :].expand(-1, act_steps, -1)
#         # 然后在 Padding 上加上对应的高斯噪声 (因为最右侧要求是纯噪声)
#         X_shift[:, -act_steps:, :] += torch.randn(bs, act_steps, D, device=device)
#     else:
#         X_shift = torch.randn_like(X_curr)

#     # 物理左移了 act_steps 格，系统的进度条 delta 增加了 act_steps
#     delta_curr_val = float(act_steps)

#     # ==========================
#     # 动作 2: 精确 ODE 直线拉回
#     # ==========================
#     obs_emb = encoder(obs, None)
#     delta_tensor = torch.full((bs,), delta_curr_val, device=device)

#     # 预测全场速度
#     V_pred = flow_map.get_velocity(delta_tensor, X_shift, obs_emb)

#     # 计算时间步长 d_tau
#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     tau_curr = torch.clamp(1.0 - (i_indices + delta_curr_val) / (H - 1), 0.0, 1.0)
#     tau_target = torch.clamp(1.0 - (i_indices + 0.0) / (H - 1), 0.0, 1.0)

#     step_size = tau_target - tau_curr # (bs, H), 值是正的，因为我们向 \tau=1 移动

#     # 直线跨步 (Rectified Flow 的原生欧拉步)
#     X_next = X_shift + step_size.unsqueeze(-1) * V_pred

#     return X_next



# def bridge_v3_sampler(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     act_0: torch.Tensor,
#     obs: torch.Tensor,
# ):
#     """Bridge Policy V3 Continuous Sampler — One-Step Shift Projection."""

#     X_curr = act_0
#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     # 提取 CFG scale
#     guidance_scale = getattr(config, "guidance_scale", 1.0)

#     # 将观测编码
#     obs_emb = encoder(obs, None)

#     # ==========================================
#     # 核心：通知网络当前的全局物理状态 (脏度)
#     # ==========================================
#     # 我们刚刚在外部环境里做了一次动作，系统向未来平移了一步。
#     # 相当于整体数据的噪声相对位置右移了一格，也就是 delta = 1.0。
#     # 告诉网络："帮我清理这 1.0 偏移量的物理毛刺"
#     delta_curr_val = 1.0
#     t_cond = torch.full((bs,), delta_curr_val, device=device)

#     # ==========================================
#     # Denoising Autoencoder 直接预测
#     # ==========================================
#     out_cond = flow_map.get_velocity(t_cond, X_curr, obs_emb)
#     X_next = out_cond

#     # if guidance_scale == 1.0:
#     #     X_next = out_cond
#     # else:
#     #     # CFG 无条件推断
#     #     t_uncond = torch.zeros((bs,), device=device) # 无条件时 t 传 0
#     #     obs_null = torch.zeros_like(obs_emb)

#     #     out_uncond = flow_map.get_velocity(t_uncond, X_curr, obs_null)
#     #     X_next = out_uncond + guidance_scale * (out_cond - out_uncond)

#     return X_next




# def bridge_v3_sampler(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     act_0: torch.Tensor,
#     obs: torch.Tensor,
# ):
#     """Bridge V3 Sampler — Shift, Re-noise, and Epsilon-Denoise."""

#     X_curr = act_0
#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     act_steps = config.num_steps
#     guidance_scale = getattr(config, "guidance_scale", 1.0)
#     obs_emb = encoder(obs, None)

#     sigma_min = 1.5e-4
#     sigma_max = 0.05
#     R = sigma_max / sigma_min

#     # ==========================
#     # 动作 1: 物理平移 (Shift)
#     # ==========================
#     X_shift = torch.zeros_like(X_curr)
#     if act_steps < H:
#         X_shift[:, :-act_steps, :] = X_curr[:, act_steps:, :]
#         # 【填坑4】：最右侧盲区，绝不 Padding！直接填入方差为 100 的纯白噪声！
#         # 强迫网络无中生有，幻觉出未来。
#         X_shift[:, -act_steps:, :] = torch.randn(bs, act_steps, D, device=device) * sigma_max
#     else:
#         X_shift = torch.randn_like(X_curr) * sigma_max

#     # ==========================
#     # 动作 2: 计算当前 Buffer 应该有的合法噪声水平
#     # ==========================
#     # 系统物理左移了 act_steps，相当于 delta = act_steps
#     delta_curr_val = float(act_steps)

#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     delta_scale = math.log(R) / (H - 1)

#     # 这就是 Shift 后，每个槽位"理应"具备的理论本底噪声
#     sigmas_target = sigma_min * (R ** (i_indices / (H - 1))) * math.exp(delta_curr_val * delta_scale)
#     sigmas_target = torch.clamp(sigmas_target, min=sigma_min, max=sigma_max)

#     # ==========================
#     # 动作 3: 显式重加噪 (洗涤物理误差!)
#     # ==========================
#     # 【填坑5】：X_shift 虽然平移了，但它身上带的是上一步的残余噪声和物理形变。
#     # 我们主动给它加上符合目标量级的纯正高斯白噪，洗掉那些结构性的毛刺！
#     noise_injection = torch.randn(bs, H, D, device=device)
#     # 注意：为了防止左侧执行端(sigma_min=1e-4)积累随机游走，我们加的噪声也极小
#     X_in = X_shift + sigmas_target.unsqueeze(-1) * noise_injection

#     # ==========================
#     # 动作 4: Epsilon 预测与一步还原
#     # ==========================
#     t_cond = torch.full((bs,), delta_curr_val, device=device)
#     Eps_pred = flow_map.get_velocity(t_cond, X_in, obs_emb)

#     if guidance_scale > 1.0:
#         t_uncond = torch.zeros((bs,), device=device)
#         obs_null = torch.zeros_like(obs_emb)
#         Eps_uncond = flow_map.get_velocity(t_uncond, X_in, obs_null)
#         Eps_pred = Eps_uncond + guidance_scale * (Eps_pred - Eps_uncond)

#     # X_0 = X_noisy - sigma * epsilon (一步到位的 DAE 去噪公式！)
#     X_clean = X_in - sigmas_target.unsqueeze(-1) * Eps_pred

#     return X_clean



def bridge_v3_sampler0318(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Bridge V3 Sampler -- Explicit Shift Then Residual Refinement

    IMPORTANT:
        act_0 must have shape (B, H, D)

    Steps:
        1) hard-coded shift
        2) optional small rollout-style perturbation
        3) predict residual refinement
        4) return refined next chunk
    """
    X_curr = act_0
    assert X_curr.ndim == 3, f"act_0 must be (B, H, D), got {X_curr.shape}"

    bs, H, D = X_curr.shape
    device = X_curr.device
    dtype = X_curr.dtype

    act_steps = getattr(config, "num_steps", 1)
    guidance_scale = getattr(config, "guidance_scale", 1.0)

    # --------------------------------------------------
    # 1. Hard-coded shift
    # --------------------------------------------------
    X_shift = torch.zeros_like(X_curr)

    if act_steps < H:
        X_shift[:, :-act_steps, :] = X_curr[:, act_steps:, :]
        # tail warm-start: copy last surviving action
        tail_seed = X_curr[:, -1:, :].expand(-1, act_steps, -1)
        X_shift[:, -act_steps:, :] = tail_seed
    else:
        # degenerate case: everything shifts out, fill with last action
        X_shift = X_curr[:, -1:, :].expand(-1, H, -1).contiguous()

    # --------------------------------------------------
    # 2. Optional tiny perturbation to simulate rollout mismatch
    # --------------------------------------------------
    sigma_min = getattr(config, "bridge_sigma_min_eval", 0.0)
    sigma_max = getattr(config, "bridge_sigma_max_eval", 0.0)
    noise_type = getattr(config, "bridge_noise_type_eval", "none")

    if sigma_max > 0.0:
        if sigma_min <= 0.0:
            i_idx = torch.arange(H, device=device, dtype=dtype).unsqueeze(0).expand(bs, H)
            if H > 1:
                sigmas = sigma_max * (i_idx / float(H - 1))
            else:
                sigmas = torch.full((bs, H), sigma_max, device=device, dtype=dtype)
        else:
            R_sigma = sigma_max / sigma_min
            i_idx = torch.arange(H, device=device, dtype=dtype).unsqueeze(0).expand(bs, H)
            sigmas = sigma_min * (R_sigma ** (i_idx / float(max(H - 1, 1))))

        if noise_type == "ou":
            noise = generate_ou_noise(bs, H, D, device, rho=getattr(config, "bridge_noise_rho_eval", 0.9))
        elif noise_type == "gaussian":
            noise = torch.randn(bs, H, D, device=device, dtype=dtype)
        else:
            noise = torch.zeros(bs, H, D, device=device, dtype=dtype)

        X_in = X_shift + sigmas.unsqueeze(-1) * noise
    else:
        X_in = X_shift

    # --------------------------------------------------
    # 3. Encode observations
    # --------------------------------------------------
    obs_emb = encoder(obs, None)

    # lightweight time token
    t_cond = torch.ones((bs,), device=device, dtype=dtype) * getattr(config, "bridge_t_value", 1.0)

    # --------------------------------------------------
    # 4. Predict residual refinement
    # --------------------------------------------------
    R_cond = flow_map.get_velocity(t_cond, X_in, obs_emb)

    if guidance_scale == 1.0:
        R_pred = R_cond
    else:
        obs_null = torch.zeros_like(obs_emb)
        t_uncond = torch.zeros((bs,), device=device, dtype=dtype)
        R_uncond = flow_map.get_velocity(t_uncond, X_in, obs_null)
        R_pred = R_uncond + guidance_scale * (R_cond - R_uncond)

    # refined next chunk
    X_next = X_in + R_pred
    return X_next




# def bridge_v3_sampler(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     act_0: torch.Tensor,
#     obs: torch.Tensor,
# ):
#     """
#     Bridge Policy V3 Sampler -- Explicit Shift + Full Next-Chunk Prediction

#     IMPORTANT:
#         act_0 must have shape (B, H, D)

#     Inference:
#         1) explicitly shift current chunk
#         2) optionally add tiny corruption
#         3) directly predict full next clean chunk
#     """
#     X_curr = act_0
#     assert X_curr.ndim == 3, f"act_0 must be (B, H, D), got {X_curr.shape}"

#     bs, H, D = X_curr.shape
#     device = X_curr.device
#     dtype = X_curr.dtype

#     act_steps = getattr(config, "num_steps", 1)
#     guidance_scale = getattr(config, "guidance_scale", 1.0)

#     # ==========================================================
#     # 1. Explicit shift
#     # ==========================================================
#     X_shift = torch.zeros_like(X_curr)

#     if act_steps < H:
#         X_shift[:, :-act_steps, :] = X_curr[:, act_steps:, :]
#         # tail warm-start: repeat last surviving action
#         X_shift[:, -act_steps:, :] = X_curr[:, -1:, :].expand(-1, act_steps, -1)
#     else:
#         # degenerate case
#         X_shift = X_curr[:, -1:, :].expand(-1, H, -1).contiguous()

#     # ==========================================================
#     # 2. Optional tiny inference corruption
#     # ==========================================================
#     sigma_min = getattr(config, "bridge_sigma_min_eval", 0.0)
#     sigma_max = getattr(config, "bridge_sigma_max_eval", 0.0)
#     noise_type = getattr(config, "bridge_noise_type_eval", "none")
#     noise_rho = getattr(config, "bridge_noise_rho_eval", 0.9)

#     if sigma_max > 0.0:
#         i_idx = torch.arange(H, device=device, dtype=dtype).unsqueeze(0).expand(bs, H)

#         if sigma_min > 0.0:
#             ratio = sigma_max / sigma_min
#             sigmas = sigma_min * (ratio ** (i_idx / float(max(H - 1, 1))))
#         else:
#             if H > 1:
#                 sigmas = sigma_max * (i_idx / float(H - 1))
#             else:
#                 sigmas = torch.full((bs, H), sigma_max, device=device, dtype=dtype)

#         if noise_type == "ou":
#             noise = generate_ou_noise(bs, H, D, device, rho=noise_rho).to(dtype)
#         elif noise_type == "gaussian":
#             noise = torch.randn(bs, H, D, device=device, dtype=dtype)
#         else:
#             noise = torch.zeros(bs, H, D, device=device, dtype=dtype)

#         X_in = X_shift + sigmas.unsqueeze(-1) * noise
#     else:
#         X_in = X_shift

#     # ==========================================================
#     # 3. Encode obs
#     # ==========================================================
#     obs_emb = encoder(obs, None)

#     # lightweight conditioning token
#     t_cond = torch.ones((bs,), device=device, dtype=dtype) * getattr(config, "bridge_t_value", 1.0)

#     # ==========================================================
#     # 4. Predict full next chunk
#     # ==========================================================
#     X_cond = flow_map.get_velocity(t_cond, X_in, obs_emb)

#     if guidance_scale == 1.0:
#         X_next = X_cond
#     else:
#         obs_null = torch.zeros_like(obs_emb)
#         t_uncond = torch.zeros((bs,), device=device, dtype=dtype)
#         X_uncond = flow_map.get_velocity(t_uncond, X_in, obs_null)
#         X_next = X_uncond + guidance_scale * (X_cond - X_uncond)

#     return X_next




def bridge_v3_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """
    Bridge Policy V5 Sampler
    Single-step Explicit Shift + Preconditioned Residual Prediction

    IMPORTANT:
        act_0 must have shape (B, H, D)

    Inference:
        1) explicit shift
        2) optional tiny perturbation
        3) predict normalized residual
        4) unscale and recover next chunk
    """
    X_curr = act_0
    assert X_curr.ndim == 3, f"act_0 must be (B, H, D), got {X_curr.shape}"

    bs, H, D = X_curr.shape
    device = X_curr.device
    dtype = X_curr.dtype

    act_steps = getattr(config, "num_steps", 1)
    guidance_scale = getattr(config, "guidance_scale", 1.0)

    # ==========================================================
    # 1. Explicit shift
    # ==========================================================
    X_shift = torch.zeros_like(X_curr)

    if act_steps < H:
        X_shift[:, :-act_steps, :] = X_curr[:, act_steps:, :]
        X_shift[:, -act_steps:, :] = X_curr[:, -1:, :].expand(-1, act_steps, -1)
    else:
        X_shift = X_curr[:, -1:, :].expand(-1, H, -1).contiguous()

    # ==========================================================
    # 2. Same preconditioning scale used at training
    # ==========================================================
    scale_min = getattr(config, "bridge_scale_min", 0.02)
    scale_max = getattr(config, "bridge_scale_max", 1.0)
    scale_eps = getattr(config, "bridge_scale_eps", 1e-6)

    i_idx = torch.arange(H, device=device, dtype=dtype).unsqueeze(0).expand(bs, H)

    if H > 1:
        if getattr(config, "bridge_scale_schedule", "exp") == "linear":
            scales = scale_min + (scale_max - scale_min) * (i_idx / float(H - 1))
        else:
            ratio = scale_max / scale_min
            scales = scale_min * (ratio ** (i_idx / float(H - 1)))
    else:
        scales = torch.full((bs, H), scale_max, device=device, dtype=dtype)

    # ==========================================================
    # 3. Optional tiny input perturbation at inference
    # ==========================================================
    noise_std = getattr(config, "bridge_input_noise_std_eval", 0.0)
    noise_type = getattr(config, "bridge_input_noise_type_eval", "none")
    noise_rho = getattr(config, "bridge_input_noise_rho_eval", 0.9)

    if noise_std > 0.0:
        if noise_type == "ou":
            noise = generate_ou_noise(bs, H, D, device, rho=noise_rho).to(dtype)
        elif noise_type == "gaussian":
            noise = torch.randn(bs, H, D, device=device, dtype=dtype)
        else:
            noise = torch.zeros(bs, H, D, device=device, dtype=dtype)

        X_in = X_shift + noise_std * noise
    else:
        X_in = X_shift

    # ==========================================================
    # 4. Encode obs
    # ==========================================================
    obs_emb = encoder(obs, None)

    t_cond = torch.ones((bs,), device=device, dtype=dtype) * getattr(config, "bridge_t_value", 1.0)

    # ==========================================================
    # 5. Predict normalized residual
    # ==========================================================
    Y_cond = flow_map.get_velocity(t_cond, X_in, obs_emb)

    if guidance_scale == 1.0:
        Y_pred = Y_cond
    else:
        obs_null = torch.zeros_like(obs_emb)
        t_uncond = torch.zeros((bs,), device=device, dtype=dtype)
        Y_uncond = flow_map.get_velocity(t_uncond, X_in, obs_null)
        Y_pred = Y_uncond + guidance_scale * (Y_cond - Y_uncond)

    # Unscale to raw residual, recover next chunk
    R_pred = Y_pred * (scales.unsqueeze(-1) + scale_eps)
    X_next = X_shift + R_pred

    return X_next



def _shift_chunk(x: torch.Tensor, act_steps: int = 1) -> torch.Tensor:
    """
    Shift chunk left by act_steps.
    Tail is padded by repeating the last surviving action.
    x: (B, H, D)
    """
    assert x.ndim == 3
    B, H, D = x.shape

    y = torch.zeros_like(x)
    if act_steps < H:
        y[:, :-act_steps, :] = x[:, act_steps:, :]
        y[:, -act_steps:, :] = x[:, -1:, :].expand(-1, act_steps, -1)
    else:
        y = x[:, -1:, :].expand(-1, H, -1).contiguous()
    return y

def prcp_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs,
):
    """
    PRCP sampler: single-step rolling chunk update.

    IMPORTANT:
        act_0 shape = (B, Hp, D)
        where Hp = config.task.horizon - 1

    Inference:
        1) explicit shift
        2) predict rollout-ready next chunk
    """
    X_curr = act_0
    assert X_curr.ndim == 3, f"act_0 must be (B, H, D), got {X_curr.shape}"

    bs, H, D = X_curr.shape
    device = X_curr.device
    dtype = X_curr.dtype

    act_steps = getattr(config, "num_steps", 1)
    guidance_scale = getattr(config, "guidance_scale", 1.0)

    # 1. explicit shift
    X_shift = _shift_chunk(X_curr, act_steps=act_steps)

    # 2. obs encoding
    obs_emb = encoder(obs, None)

    # 3. stage token
    # For PRCP-v1/v2 first round, use rolling stage only.
    t_roll = getattr(config, "prcp_t_roll", 1.0)
    t_cond = torch.full((bs,), t_roll, device=device, dtype=dtype)

    X_cond = flow_map.get_velocity(t_cond, X_shift, obs_emb)
    X_next = X_cond

    # if guidance_scale == 1.0:
    #     X_next = X_cond
    # else:
    #     obs_null = torch.zeros_like(obs_emb)
    #     t_uncond = torch.zeros((bs,), device=device, dtype=dtype)
    #     X_uncond = flow_map.get_velocity(t_uncond, X_shift, obs_null)
    #     X_next = X_uncond + guidance_scale * (X_cond - X_uncond)

    return X_next




def _build_tau_ladder(
    T: int,
    device,
    dtype,
    tau_min: float = 0.03,
    tau_max: float = 1.0,
    mode: str = "linear",
    beta: float = 2.0,
):
    if T == 1:
        return torch.full((1, 1), tau_max, device=device, dtype=dtype)

    idx = torch.arange(T, device=device, dtype=dtype)

    if mode == "exp":
        raw = torch.exp(beta * idx / float(T - 1))
        raw = (raw - raw[0]) / (raw[-1] - raw[0] + 1e-8)
        tau = tau_min + (tau_max - tau_min) * raw
    else:
        tau = tau_min + (tau_max - tau_min) * (idx / float(T - 1))

    return tau.view(1, T)


def _sample_noise(bs, T, D, device, dtype, noise_type="gaussian", rho=0.9):
    if noise_type == "ou":
        return generate_ou_noise(bs, T, D, device, rho=rho).to(dtype)
    return torch.randn(bs, T, D, device=device, dtype=dtype)


def _forward_noise(clean_x, tau_vec, noise):
    tau = tau_vec.unsqueeze(-1)  # (B or 1, T, 1)
    return (1.0 - tau) * clean_x + tau * noise


def rolling_policy_sampler0322(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs,
):
    """
    Returns:
        Y: (B, T+1, D)
            Y[:, 0, :]   = clean action to execute now
            Y[:, 1:, :]  = next noisy rolling window
    """
    X_window = act_0
    assert X_window.ndim == 3, f"act_0 must be (B, T, D), got {X_window.shape}"

    bs, T, D = X_window.shape
    device = X_window.device
    dtype = X_window.dtype

    tau_vec = _build_tau_ladder(
        T=T,
        device=device,
        dtype=dtype,
        tau_min=getattr(config, "rolling_tau_min", 0.03),
        tau_max=getattr(config, "rolling_tau_max", 1.0),
        mode=getattr(config, "rolling_tau_mode", "linear"),
        beta=getattr(config, "rolling_tau_beta", 2.0),
    )  # (1, T)

    obs_emb = encoder(obs, None)

    t_scalar = torch.full(
        (bs,),
        float(tau_vec.mean().detach().cpu()),
        device=device,
        dtype=dtype,
    )

    X_clean_pred = flow_map.get_velocity(
        t_scalar,
        X_window,
        obs_emb,
        slot_noise_levels=tau_vec.expand(bs, -1),
    )  # (B, T, D)

    # build next noisy rolling window
    X_next = torch.zeros_like(X_window)

    if T > 1:
        tau_next = tau_vec[:, :-1]  # (1, T-1)

        eps_next = _sample_noise(
            bs=bs,
            T=T - 1,
            D=D,
            device=device,
            dtype=dtype,
            noise_type=getattr(config, "rolling_noise_type_eval", "gaussian"),
            rho=getattr(config, "rolling_noise_rho_eval", 0.9),
        )

        X_next[:, :-1, :] = _forward_noise(
            X_clean_pred[:, 1:, :],
            tau_next,
            eps_next,
        )

    # tail fresh high-noise sample
    eps_tail = torch.randn(bs, 1, D, device=device, dtype=dtype)
    tau_tail = tau_vec[:, -1:].unsqueeze(-1)  # (1, 1, 1)
    X_next[:, -1:, :] = tau_tail * eps_tail

    # package:
    # first slot = executable clean action
    # remaining slots = next noisy window
    Y = torch.cat([X_clean_pred[:, 0:1, :], X_next], dim=1)
    return Y



def rolling_policy_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs,
):
    """
    Rolling Policy (Local-Step) sampler

    Input:
        act_0: current rolling noisy window, shape (B, T, D)
               interpreted as levels tau_1 ... tau_T

    Output:
        Y: (B, T+1, D)
           Y[:, 0, :]   = current clean action to execute
           Y[:, 1:, :]  = next rolling noisy window (again tau_1 ... tau_T)
    """
    X_window = act_0
    assert X_window.ndim == 3, f"act_0 must be (B, T, D), got {X_window.shape}"

    bs, T, D = X_window.shape
    device = X_window.device
    dtype = X_window.dtype

    tau_vec = _build_tau_ladder(
        T=T,
        device=device,
        dtype=dtype,
        tau_min=getattr(config, "rolling_tau_min", 1e-3),
        tau_max=getattr(config, "rolling_tau_max", 0.05),
        mode=getattr(config, "rolling_tau_mode", "linear"),
        beta=getattr(config, "rolling_tau_beta", 2.0),
    )  # (1, T)

    obs_emb = encoder(obs, None)

    t_scalar = torch.full(
        (bs,),
        float(tau_vec.mean().detach().cpu()),
        device=device,
        dtype=dtype,
    )

    # predict one-level-down window:
    # slot 0 -> clean
    # slot 1 -> tau_1
    # ...
    # slot T-1 -> tau_{T-1}
    X_down = flow_map.get_velocity(
        t_scalar,
        X_window,
        obs_emb,
        slot_noise_levels=tau_vec.expand(bs, -1),
    )  # (B, T, D)

    # build next rolling window of size T:
    # take slots 1..T-1 from X_down, append fresh tail at tau_T
    X_next = torch.zeros_like(X_window)

    if T > 1:
        X_next[:, :-1, :] = X_down[:, 1:, :]

    # append fresh highest-noise candidate
    eps_tail = torch.randn(bs, 1, D, device=device, dtype=dtype)
    # tau_tail = tau_vec[:, -1:].unsqueeze(-1)  # (1,1,1)
    # X_next[:, -1:, :] = tau_tail * eps_tail
    clean_ref = X_down[:, 0:1, :]
    # X_next[:, -1:, :] = _forward_noise(clean_ref, tau_tail, eps_tail)
    X_next[:, -1:, :] = _forward_noise(clean_ref, tau_vec[:, -1:], eps_tail)


    # package:
    # first element is executable clean action
    # remaining are next noisy window
    Y = torch.cat([X_down[:, 0:1, :], X_next], dim=1)
    return Y
