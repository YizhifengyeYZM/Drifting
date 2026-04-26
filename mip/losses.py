"""Losses for iterative policy training."""

from collections.abc import Callable

import torch
import numpy as np
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant


def get_norm(x: torch.Tensor, norm_type: str) -> torch.Tensor:
    if norm_type == "l2":
        # squared L2 (no sqrt)
        return torch.sum(x * x, dim=-1)
    elif norm_type == "l1":
        return torch.sum(torch.abs(x), dim=-1)
    elif norm_type == "smooth_l1":
        # per-element smooth L1, then sum over last dim
        return torch.sum(
            F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none"), dim=-1
        )
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


def _naive_drift_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """Official Drifting pairwise L2 distance, ported from the JAX release."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def _naive_drift_core_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.2),
):
    """Drifting loss on raw generated samples.

    Shapes follow the official implementation:
    gen=[B, C_g, S], fixed_pos=[B, C_p, S], fixed_neg=[B, C_n, S].
    """
    batch_size, num_gen, sample_dim = gen.shape
    num_pos = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(batch_size, 0, sample_dim)
    num_neg = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(batch_size, num_gen)
    if weight_pos is None:
        weight_pos = gen.new_ones(batch_size, num_pos)
    if weight_neg is None:
        weight_neg = gen.new_ones(batch_size, num_neg)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        info = {}
        dist = _naive_drift_cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        scale_inputs = torch.clamp(scale / (sample_dim ** 0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)

        diag_mask = torch.eye(num_gen, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, num_neg + num_pos)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * 100.0

        force_across_R = torch.zeros_like(old_gen_scaled)
        for temperature in temperatures:
            logits = -dist_normed / temperature
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            split_idx = num_gen + num_neg
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]
            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            r_coeff_pos = aff_pos * sum_neg
            r_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force = torch.einsum("biy,byx->bix", r_coeff, targets_scaled)
            total_coeffs = r_coeff.sum(dim=-1)
            total_force = total_force - total_coeffs.unsqueeze(-1) * old_gen_scaled

            force_norm = (total_force ** 2).mean()
            info[f"loss_R_{str(temperature).replace('.', '_')}"] = force_norm
            force_across_R = force_across_R + total_force / torch.sqrt(
                torch.clamp(force_norm, min=1e-8)
            )

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs.detach()
    loss = ((gen_scaled - goal_scaled.detach()) ** 2).mean(dim=(-1, -2))
    info = {key: value.mean() for key, value in info.items()}
    return loss, info


def _dp_cosine_betas(
    num_train_timesteps: int,
    device: torch.device,
    dtype: torch.dtype,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """Diffusers/DDPM squaredcos_cap_v2 beta schedule."""
    steps = torch.arange(num_train_timesteps + 1, device=device, dtype=dtype)
    t = steps / num_train_timesteps
    alphas_cumprod = torch.cos(((t + 0.008) / 1.008) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, min=1e-8, max=max_beta)


def dp_get_schedule(
    config: OptimizationConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build the small DDPM schedule used by DP loss and sampler."""
    num_train_timesteps = int(getattr(config, "dp_num_train_timesteps", 100))
    beta_schedule = str(getattr(config, "dp_beta_schedule", "squaredcos_cap_v2"))
    beta_start = float(getattr(config, "dp_beta_start", 0.0001))
    beta_end = float(getattr(config, "dp_beta_end", 0.02))

    if beta_schedule == "squaredcos_cap_v2":
        betas = _dp_cosine_betas(num_train_timesteps, device=device, dtype=dtype)
    elif beta_schedule == "linear":
        betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, device=device, dtype=dtype
        )
    else:
        raise NotImplementedError(f"DP beta schedule {beta_schedule} not implemented.")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
    }


def dp_model_timestep(
    config: OptimizationConfig,
    timesteps: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert integer DDPM timesteps to the scalar consumed by MIP networks."""
    mode = str(getattr(config, "dp_timestep_scale", "raw"))
    timesteps = timesteps.to(dtype=dtype)
    if mode == "raw":
        return timesteps
    if mode == "normalized":
        denom = max(int(getattr(config, "dp_num_train_timesteps", 100)) - 1, 1)
        return timesteps / denom
    raise NotImplementedError(f"DP timestep scale {mode} not implemented.")


def _dp_extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    sample_shape: torch.Size,
) -> torch.Tensor:
    out = values.gather(0, timesteps.clamp(min=0).long())
    return out.reshape(timesteps.shape[0], *((1,) * (len(sample_shape) - 1)))


def dp_add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    alphas_cumprod = schedule["alphas_cumprod"]
    sqrt_alpha_prod = torch.sqrt(_dp_extract(alphas_cumprod, timesteps, clean.shape))
    sqrt_one_minus_alpha_prod = torch.sqrt(
        1.0 - _dp_extract(alphas_cumprod, timesteps, clean.shape)
    )
    return sqrt_alpha_prod * clean + sqrt_one_minus_alpha_prod * noise


def dp_ddpm_step(
    config: OptimizationConfig,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    prev_timestep: torch.Tensor,
    sample: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """One DDPM reverse step for epsilon-prediction with fixed_small variance."""
    alphas_cumprod = schedule["alphas_cumprod"]
    alpha_prod_t = _dp_extract(alphas_cumprod, timestep, sample.shape)
    prev_index = prev_timestep.clamp(min=0).long()
    alpha_prod_t_prev_raw = _dp_extract(alphas_cumprod, prev_index, sample.shape)
    alpha_prod_t_prev = torch.where(
        (prev_timestep >= 0).reshape(-1, *((1,) * (sample.dim() - 1))),
        alpha_prod_t_prev_raw,
        torch.ones_like(alpha_prod_t_prev_raw),
    )
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1.0 - current_alpha_t

    pred_original_sample = (
        sample - torch.sqrt(torch.clamp(beta_prod_t, min=1e-20)) * model_output
    ) / torch.sqrt(torch.clamp(alpha_prod_t, min=1e-20))
    if bool(getattr(config, "dp_clip_sample", True)):
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

    pred_original_coeff = (
        torch.sqrt(alpha_prod_t_prev) * current_beta_t / torch.clamp(beta_prod_t, min=1e-20)
    )
    current_sample_coeff = (
        torch.sqrt(current_alpha_t) * beta_prod_t_prev / torch.clamp(beta_prod_t, min=1e-20)
    )
    pred_prev_sample = pred_original_coeff * pred_original_sample + current_sample_coeff * sample

    variance = (
        beta_prod_t_prev / torch.clamp(beta_prod_t, min=1e-20) * current_beta_t
    )
    noise = torch.randn_like(sample)
    nonzero_mask = (timestep > 0).reshape(-1, *((1,) * (sample.dim() - 1))).to(sample.dtype)
    return pred_prev_sample + nonzero_mask * torch.sqrt(torch.clamp(variance, min=1e-20)) * noise


def get_loss_fn(loss_type: str) -> Callable:
    if loss_type == "flow":
        return flow_loss
    elif loss_type == "regression":
        return regression_loss
    elif loss_type == "straight_flow":
        return straight_flow_loss
    elif loss_type == "tsd":
        return tsd_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "lmd":
        return lmd_loss
    elif loss_type == "ctm":
        return ctm_loss
    elif loss_type == "psd":
        return psd_loss
    elif loss_type == "lsd":
        return lsd_loss
    elif loss_type == "esd":
        return esd_loss
    elif loss_type == "mf":
        return mf_loss
    elif loss_type == "bridge":
        return bridge_loss
    elif loss_type == "bridge_v2":
        return bridge_v2_loss03092
    elif loss_type == "bridge_v3":
        return bridge_v3_loss
    elif loss_type == "prcp_v1":
        return prcp_v1_loss
    elif loss_type == "prcp_v2":
        return prcp_v2_loss
    elif loss_type == "rp_v1":
        return rolling_policy_v1_loss
    elif loss_type in ["dp", "ddpm"]:
        return dp_loss
    elif loss_type == "naive_drift":
        return naive_drift_loss
    elif loss_type == "drifting":
        return drifting_policy_loss
    elif loss_type == "drifting2":
        return drifting_policy_loss2
    elif loss_type == "drifting3":
        return drifting_policy_loss3
    elif loss_type == "drifting4":
        return drifting_policy_loss4
    elif loss_type == "drifting5":
        return drifting_policy_loss5
    elif loss_type == "drifting6":
        return drifting_policy_loss6
    elif loss_type == "globaldiag":
        return globaldiag_policy_loss
    elif loss_type == "arl":
        return arl_policy_loss
    elif loss_type == "geofuse":
        return geofuse_policy_loss
    elif loss_type == "geofuse_noise":
        return geofuse_noise_loss
    elif loss_type == "geofuse_align":
        return geofuse_align_loss
    elif loss_type == "drifting7":
        return drifting_policy_loss7
    elif loss_type == "drifting8":
        return drifting_policy_loss8
    elif loss_type == "drifting9":
        return drifting_policy_loss9
    elif loss_type == "drifting10":
        return drifting_policy_loss10
    elif loss_type == "drifting11":
        return drifting_policy_loss11
    elif loss_type == "drifting12":
        return drifting_policy_loss12
    elif loss_type == "drifting13":
        return drifting_policy_loss13
    elif loss_type == "drifting14":
        return drifting_policy_loss14
    elif loss_type == "drift6min":
        return drift6min_loss
    elif loss_type == "drift6matrix":
        return drift6matrix_loss
    elif loss_type == "drift6min_awshort":
        return drift6min_awshort_loss
    elif loss_type == "drift6min_rebal":
        return drift6min_rebal_loss
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Standard regression loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}

def straight_flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Straight flow loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)

    # Major difference compared to regression: use random noise instead of zeros
    act_0 = torch.randn_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def dp_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Diffusion Policy DDPM loss on normalized action chunks."""
    del interp, delta_t
    obs_emb = encoder(obs, None)
    batch_size = act.shape[0]
    num_train_timesteps = int(getattr(config, "dp_num_train_timesteps", 100))
    prediction_type = str(getattr(config, "dp_prediction_type", "epsilon"))
    loss_scale = float(getattr(config, "dp_loss_scale", 1.0))

    schedule = dp_get_schedule(config, device=act.device, dtype=act.dtype)
    timesteps = torch.randint(
        low=0,
        high=num_train_timesteps,
        size=(batch_size,),
        device=act.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(act)
    noisy_act = dp_add_noise(act, noise, timesteps, schedule)

    t_model = dp_model_timestep(config, timesteps, dtype=act.dtype)
    pred = flow_map.get_velocity(t_model, noisy_act, obs_emb)

    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "sample":
        target = act
    else:
        raise NotImplementedError(f"DP prediction type {prediction_type} not implemented.")

    raw_loss = F.mse_loss(pred, target)
    loss = loss_scale * raw_loss
    return loss, {
        "dp_mse": raw_loss.detach(),
        "dp_timestep_mean": timesteps.float().mean().detach(),
        "dp_loss_scale": torch.as_tensor(loss_scale, device=act.device, dtype=act.dtype),
    }


def naive_drift_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Naive Drifting baseline on raw action chunks.

    This matches the released Drifting robotics adaptation: draw G generator
    samples per label, use a single t=0 forward pass, and apply Drifting loss
    directly in action space.
    """
    del interp
    batch_size, horizon, act_dim = act.shape
    gen_per_label = int(getattr(config, "naive_drift_gen_per_label", 8))
    if gen_per_label < 1:
        raise ValueError("naive_drift_gen_per_label must be >= 1.")

    temperatures = tuple(
        float(v)
        for v in getattr(config, "naive_drift_temperatures", [0.02, 0.05, 0.2])
    )
    per_timestep_loss = bool(getattr(config, "naive_drift_per_timestep_loss", True))
    loss_scale = float(getattr(config, "naive_drift_loss_scale", 1.0))

    obs_emb = encoder(obs, None)
    obs_emb_rep = obs_emb.repeat_interleave(gen_per_label, dim=0)
    noise = torch.empty(
        batch_size * gen_per_label,
        horizon,
        act_dim,
        device=act.device,
        dtype=act.dtype,
    ).normal_(0, 1)
    t0 = torch.zeros(
        batch_size * gen_per_label,
        device=delta_t.device,
        dtype=delta_t.dtype,
    )

    pred_all = flow_map.get_velocity(t0, noise, obs_emb_rep)
    pred_actions = pred_all.reshape(batch_size, gen_per_label, horizon, act_dim)

    if per_timestep_loss:
        loss = act.new_zeros(())
        metrics = {}
        for step in range(horizon):
            gen_step = pred_actions[:, :, step, :]
            pos_step = act[:, step, :].unsqueeze(1)
            step_loss, step_info = _naive_drift_core_loss(
                gen_step,
                pos_step,
                temperatures=temperatures,
            )
            loss = loss + step_loss.mean()
            for key, value in step_info.items():
                metrics[key] = metrics.get(key, act.new_zeros(())) + value / horizon
        loss = loss / horizon
    else:
        gen = pred_actions.reshape(batch_size, gen_per_label, horizon * act_dim)
        pos = act.reshape(batch_size, 1, horizon * act_dim)
        raw_loss, metrics = _naive_drift_core_loss(
            gen,
            pos,
            temperatures=temperatures,
        )
        loss = raw_loss.mean()

    loss = loss * loss_scale
    metrics = {f"naive_drift_{key}": value.detach() for key, value in metrics.items()}
    metrics["naive_drift_loss_scale"] = torch.as_tensor(
        loss_scale, device=loss.device, dtype=loss.dtype
    )
    return loss, metrics


def tsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Two step denoising loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.empty_like(act).normal_(0, 1)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss0 = get_norm((act_pred_0 - act_t) / config.t_two_step, config.norm_type)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def mip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # NOTE: in paper, we use
    # act_t = config.t_two_step * act + (1 - config.t_two_step) * noise
    # but we found that this is not necessary when config.t_two_step close to 1.
    # feel free to use the original form if you want to, you can refer to mip_origin_loss

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}

def drifting_policy_loss2(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Single-step Drifting Policy Loss preserving multimodal capability."""
    bs = act.shape[0]
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step

    # 1. 提取观测特征
    obs_emb = encoder(obs, None)

    # 2. 学生视角的单步跨越起点 (Train-Test Alignment)
    # 必须输入标准高斯噪声，保证与推断时完全一致，激活多模态路由能力
    noise_s = torch.empty_like(act).normal_(0, 1)

    # 3. 老师视角的局部扰动起点 (保留 MIP 的 C2 噪声注入)
    # 在目标流形附近注入噪声，强制网络学习局部纠错场
    noise_t = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise_t

    # 网络前向传播
    act_pred_0 = flow_map.get_velocity(s, noise_s, obs_emb) # 学生单步预测
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)   # 老师局部修正

    # =========================================================
    # Drifting Joint Kernel (极致压缩计算版)
    # =========================================================
    tau_obs = getattr(config, 'tau_obs', 0.05)
    epsilon = getattr(config, 'epsilon', 0.1)

    # 1. 计算观测特征引力 k_obs
    # [B, ...] -> [B, -1] 展平，计算余弦相似度
    obs_flat = F.normalize(obs_emb.reshape(bs, -1), p=2, dim=-1)
    k_obs = torch.exp((torch.matmul(obs_flat, obs_flat.t()) - 1.0) / tau_obs)

    # 2. 将动作特征展平计算距离 (使用 reshape 应对非连续内存)
    act_flat = act.reshape(bs, -1)
    act_p1_flat = act_pred_1.reshape(bs, -1)

    # 3. 计算 V+ (向真实动作靠拢的引力场)
    dist_sq_gt = torch.cdist(act_p1_flat, act_flat, p=2).pow(2)
    k_plus = k_obs * torch.exp(-dist_sq_gt / epsilon)
    k_plus = k_plus / (k_plus.sum(dim=1, keepdim=True) + 1e-8)
    # 用 BxB 权重矩阵乘以 BxD 的动作矩阵，再恢复形状
    V_plus = torch.matmul(k_plus, act_flat).reshape_as(act) - act_pred_1

    # 4. 计算 V- (推开自身假样本的斥力场)
    dist_sq_pred = torch.cdist(act_p1_flat, act_p1_flat, p=2).pow(2)
    k_minus = k_obs * torch.exp(-dist_sq_pred / epsilon)
    k_minus = k_minus / (k_minus.sum(dim=1, keepdim=True) + 1e-8)
    V_minus = torch.matmul(k_minus, act_p1_flat).reshape_as(act) - act_pred_1

    # 5. 构造完美目标并阻断梯度
    target = (act_pred_1 + 0.5 * V_plus - 0.5 * V_minus).detach()

    # =========================================================
    # Loss 计算 (老师局部拟合 + 学生单步蒸馏)
    # =========================================================
    loss1 = get_norm((act_pred_1 - target) / (1 - config.t_two_step), config.norm_type)
    loss0 = get_norm((act_pred_0 - target) / config.t_two_step, config.norm_type)

    loss = config.loss_scale * torch.mean(loss0 + loss1)

    return loss, {}


def drifting_policy_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    # act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = config.t_two_step * act + (1 - config.t_two_step) * noise

    # NOTE: in paper, we use
    # act_t = config.t_two_step * act + (1 - config.t_two_step) * noise
    # but we found that this is not necessary when config.t_two_step close to 1.
    # feel free to use the original form if you want to, you can refer to mip_origin_loss

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    pred_velocity_0 = flow_map.get_velocity(s, noise, obs_emb)
    pred_velocity_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((pred_velocity_0 - (act - noise)), config.norm_type)
    loss1 = get_norm((pred_velocity_1 - (act - noise)), config.norm_type)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}

def sinkhorn(cost_matrix: torch.Tensor, epsilon: float = 0.5, n_iters: int = 30):
    # 1. 归一化代价矩阵，防止 exp(-很大的数) 直接下溢出变成全 0
    C = cost_matrix / (cost_matrix.max() + 1e-8)

    # 2. 正常的 Sinkhorn
    K = torch.exp(-C / epsilon)
    B = cost_matrix.shape[0]
    u = torch.ones(B, device=cost_matrix.device)
    v = torch.ones(B, device=cost_matrix.device)
    for _ in range(n_iters):
        v = 1.0 / (K.t() @ u + 1e-8)
        u = 1.0 / (K @ v + 1e-8)

    # 3. 绝对不乘 B！返回真正的概率分配矩阵
    return u.unsqueeze(1) * K * v.unsqueeze(0)

# def drifting_policy_loss3(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ) -> float:
#     """Minimum iterative policy loss."""
#     # sample
#     bs = act.shape[0]
#     s = torch.zeros_like(delta_t, device=delta_t.device)
#     t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step

#     obs_emb = encoder(obs, None)

#     # =========================================================
#     # 核心：联合特征 Sinkhorn 配对
#     # =========================================================
#     with torch.no_grad():
#         raw_noise = torch.empty_like(act).normal_(0, 1)

#         # ##! 拼接 Action 和 Obs 特征 (天然隔离条件，无需额外超参！)
#         # act_flat = act.reshape(bs, -1)
#         # noise_flat = raw_noise.reshape(bs, -1)
#         # obs_flat = obs_emb.reshape(bs, -1)

#         # joint_act = torch.cat([act_flat, obs_flat], dim=1)
#         # joint_noise = torch.cat([noise_flat, obs_flat], dim=1)

#         # # 计算联合代价并求解软分配
#         # P = sinkhorn(torch.cdist(joint_noise, joint_act, p=2).pow(2), epsilon=1.0)


#         ##** 拼接 obs embedding，哪怕是 embedding，维度都太高了（相较于 action 以及 noise）
#         # 所以，先尝试一种【容易导致模型坍缩，但维度可接受，不会爆炸的】action 单独和 noise 做 cost matrix 的形式
#         act_flat = act.reshape(bs, -1)
#         noise_flat = raw_noise.reshape(bs, -1)
#         P = sinkhorn(torch.cdist(noise_flat, act_flat, p=2).pow(2), epsilon=1.0)

#         # 生成配对噪声：paired_noise[i] 是针对 act[i] 的最优噪声期望
#         paired_noise = (P @ noise_flat).reshape_as(act)

#     v_target = act - paired_noise
#     act_t = config.t_two_step * act + (1 - config.t_two_step) * paired_noise

#     # 1. 老师网络：在 t=0.9 处学习纠错速度
#     # act_t = interp.calc_It(t, paired_noise, act)
#     # v_target_t = interp.calc_It_dot(t, paired_noise, act) # 自动算出精确的 (act - paired_noise)
#     pred_v_1 = flow_map.get_velocity(t, act_t, obs_emb)
#     loss1 = get_norm(pred_v_1 - v_target, config.norm_type)

#     # 2. 学生网络：从 t=0 跨越 (推断时的唯一路径)
#     # 起点就是 paired_noise
#     # v_target_s = interp.calc_It_dot(s, paired_noise, act)
#     pred_v_0 = flow_map.get_velocity(s, paired_noise, obs_emb)
#     loss0 = get_norm(pred_v_0 - v_target, config.norm_type)

#     loss = config.loss_scale * torch.mean(loss0 + loss1)

#     return loss, {}


def drifting_policy_loss3(
    config, flow_map, encoder, interp, act: torch.Tensor, obs: torch.Tensor, delta_t: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    bs = act.shape[0]
    s = torch.zeros_like(delta_t)
    t = torch.full_like(delta_t, config.t_two_step)
    obs_emb = encoder(obs, None)

    # =========================================================
    # 绝对安全的硬匹配 (Optimal Transport Pairing)
    # 没有任何 epsilon，没有任何数值溢出，绝对 1 对 1 拉直轨迹
    # =========================================================
    with torch.no_grad():
        raw_noise = torch.empty_like(act).normal_(0, 1)

        # 展平计算距离
        act_flat = act.reshape(bs, -1)
        noise_flat = raw_noise.reshape(bs, -1)

        # 计算纯净的动作欧氏距离 (如果你想加回 Obs 隔离墙，也绝对不会爆 0)
        cost_matrix = torch.cdist(noise_flat, act_flat, p=2).pow(2)

        # 使用 Scipy 算最优匹配 (100% 精准，无超参)
        # row_ind 对应 noise，col_ind 对应 act
        cost_matrix_np = cost_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

        # 极其关键：按正确的顺序重排噪声！
        # 让 paired_noise 的第 i 个元素，恰好对应 act 的第 i 个元素
        paired_noise = torch.empty_like(raw_noise)
        paired_noise[col_ind] = raw_noise[row_ind]

    # =========================================================
    # 以下代码无需任何更改！它们原本的逻辑就是完美的
    # =========================================================
    v_target = act - paired_noise
    act_t = config.t_two_step * act + (1 - config.t_two_step) * paired_noise

    pred_v_1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss1 = get_norm(pred_v_1 - v_target, config.norm_type)

    pred_v_0 = flow_map.get_velocity(s, paired_noise, obs_emb)
    loss0 = get_norm(pred_v_0 - v_target, config.norm_type)

    loss = config.loss_scale * torch.mean(loss0 + loss1)
    return loss, {}

def drifting_policy_loss4(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Minimal Drifting Policy loss for state experiments.

    Assumptions for this clean version:
    - encoder(obs, None) returns a tensor of shape [B, H] or [B, ...]
    - use current action representation directly
    - no top-k, no warmup, no memory bank
    - deployment remains one-step
    """

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    num_particles = getattr(config, "drifting_num_particles", 4)
    lambda_neg = getattr(config, "drifting_lambda_neg", 0.25)
    eps_cov = getattr(config, "drifting_cov_eps", 1e-4)

    B = act.shape[0]
    act_shape = act.shape[1:]
    D = act[0].numel()
    device = act.device
    dtype = act.dtype
    eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)

    # ------------------------------------------------------------------
    # 1) one-step branch (the actual deployment branch)
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)

    obs_emb = encoder(obs, None)
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=1e-6)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)          # [B, ...]
    act_pred_0_flat = act_pred_0.reshape(B, D)
    act_flat = act.reshape(B, D)

    # ------------------------------------------------------------------
    # 2) short-step branch with M particles per observation
    # ------------------------------------------------------------------
    noise = torch.randn((B, num_particles) + act_shape, device=device, dtype=dtype)
    act_t = act.unsqueeze(1) + (1.0 - t_star) * noise              # [B, M, ...]

    t_rep = torch.full((B * num_particles,), t_star, device=device, dtype=delta_t.dtype)
    obs_emb_rep = obs_emb.repeat_interleave(num_particles, dim=0)

    act_pred_1 = flow_map.get_velocity(
        t_rep,
        act_t.reshape(B * num_particles, *act_shape),
        obs_emb_rep,
    ).reshape(B, num_particles, *act_shape)                        # [B, M, ...]

    act_pred_1_flat = act_pred_1.reshape(B, num_particles, D)
    short_det = act_pred_1_flat.detach()

    # ------------------------------------------------------------------
    # 3) batch-wise soft conditional weights w_ij (self included)
    # ------------------------------------------------------------------
    sim = obs_feat @ obs_feat.transpose(0, 1)                      # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    logits = (sim - sim_mean) / sim_std
    weights = logits.softmax(dim=1)                               # [B, B]

    # ------------------------------------------------------------------
    # 4) positive local geometry from expert actions
    # ------------------------------------------------------------------
    mu_pos = weights @ act_flat.detach()                           # [B, D]

    centered_pos = act_flat.detach().unsqueeze(0) - mu_pos.unsqueeze(1)   # [B, B, D]
    cov_pos = torch.einsum("ijd,ije,ij->ide", centered_pos, centered_pos, weights)

    n_eff = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)      # [B]
    tau_pos = cov_pos.diagonal(dim1=-2, dim2=-1).mean(dim=-1)     # [B]
    rho_pos = float(D) / (float(D) + n_eff)                       # [B]

    sigma_pos = (
        (1.0 - rho_pos).view(B, 1, 1) * cov_pos
        + rho_pos.view(B, 1, 1) * tau_pos.view(B, 1, 1) * eye
        + eps_cov * eye
    )

    chol_pos, info_pos = torch.linalg.cholesky_ex(sigma_pos)
    if torch.any(info_pos > 0):
        sigma_pos = sigma_pos + 1e-4 * eye
        chol_pos = torch.linalg.cholesky(sigma_pos)

    # score_pos(y) = - Sigma_pos^{-1} (y - mu_pos)
    delta_pos = short_det - mu_pos.unsqueeze(1)                   # [B, M, D]
    rhs_pos = delta_pos.transpose(1, 2)                           # [B, D, M]
    solve_pos = torch.cholesky_solve(rhs_pos, chol_pos).transpose(1, 2)
    score_pos = -solve_pos                                        # [B, M, D]

    # ------------------------------------------------------------------
    # 5) weak self cloud from short-step particles
    # ------------------------------------------------------------------
    mu_neg = short_det.mean(dim=1)                                # [B, D]
    centered_neg = short_det - mu_neg.unsqueeze(1)                # [B, M, D]
    cov_neg = torch.einsum("bmd,bme->bde", centered_neg, centered_neg) / float(num_particles)

    tau_neg = cov_neg.diagonal(dim1=-2, dim2=-1).mean(dim=-1)     # [B]
    rho_neg = float(D) / (float(D) + float(num_particles))

    sigma_neg = (
        (1.0 - rho_neg) * cov_neg
        + rho_neg * tau_neg.view(B, 1, 1) * eye
        + eps_cov * eye
    )

    chol_neg, info_neg = torch.linalg.cholesky_ex(sigma_neg)
    if torch.any(info_neg > 0):
        sigma_neg = sigma_neg + 1e-4 * eye
        chol_neg = torch.linalg.cholesky(sigma_neg)

    # score_neg(y) = - Sigma_neg^{-1} (y - mu_neg)
    delta_neg = short_det - mu_neg.unsqueeze(1)                   # [B, M, D]
    rhs_neg = delta_neg.transpose(1, 2)                           # [B, D, M]
    solve_neg = torch.cholesky_solve(rhs_neg, chol_neg).transpose(1, 2)
    score_neg = -solve_neg                                        # [B, M, D]

    # ------------------------------------------------------------------
    # 6) training-time drifting field
    # ------------------------------------------------------------------
    field = score_pos - lambda_neg * score_neg                    # [B, M, D]

    # clip correction to unit norm in the current action scale
    correction = field / field.norm(dim=-1, keepdim=True).clamp_min(1.0)
    corrected = (short_det + correction).detach()                 # [B, M, D]
    corrected_mean = corrected.mean(dim=1).reshape(B, *act_shape)

    # ------------------------------------------------------------------
    # 7) losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))

    act_rep = act.unsqueeze(1).expand_as(act_pred_1)
    loss_short = torch.mean(
        get_norm(
            (act_pred_1.reshape(B * num_particles, *act_shape) - act_rep.reshape(B * num_particles, *act_shape))
            / (1.0 - t_star),
            norm_type,
        )
    )

    loss_drift = torch.mean((act_pred_1_flat - corrected).pow(2).mean(dim=-1))
    loss_distill = torch.mean(get_norm((act_pred_0 - corrected_mean) / t_star, norm_type))

    loss = loss_coarse + 0.5 * loss_short + loss_drift + loss_distill
    loss = loss_scale * loss

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_drift": float(loss_drift.detach()),
        "loss_distill": float(loss_distill.detach()),
        "w_self_mean": float(weights.diagonal().mean().detach()),
        "field_norm_mean": float(field.norm(dim=-1).mean().detach()),
        "tau_pos_mean": float(tau_pos.mean().detach()),
        "tau_neg_mean": float(tau_neg.mean().detach()),
    }

    return loss, aux

def drifting_policy_loss5(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Aggressive v2:
    MIP skeleton + geometry-weighted short-step loss

    Key changes from previous drifting version:
    - no self term
    - no particle cloud
    - no corrected target
    - no loss_drift / loss_distill
    - local geometry only reweights supervised residuals
    """

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    # Optional knobs with safe defaults; no config change required
    geo_weight = getattr(config, "drifting_geo_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()

    # ------------------------------------------------------------------
    # 1) same two branches as MIP
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # 2) batch-wise soft conditional weights from observation features
    # ------------------------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=1e-6)

    sim = obs_feat @ obs_feat.transpose(0, 1)                  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    logits = (sim - sim_mean) / sim_std
    weights = logits.softmax(dim=1)                           # [B, B], self included

    # ------------------------------------------------------------------
    # 3) paired-centered local diagonal geometry
    #    v_i[d] = sum_j w_ij * (a_j^*[d] - a_i^*[d])^2
    # ------------------------------------------------------------------
    act_flat = act.reshape(B, D).detach()
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)     # [B, B, D], a_j - a_i
    local_var = (weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1)  # [B, D]

    # Turn local variance into per-dim geometry weights:
    # - low local variance  -> larger weight
    # - high local variance -> smaller weight
    # Then renormalize so the average weight per sample is ~1
    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    geo_scale = torch.sqrt(local_var_mean / (local_var + geo_eps))
    geo_scale = geo_scale / geo_scale.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    geo_scale = geo_scale.detach()                             # [B, D]

    # ------------------------------------------------------------------
    # 4) losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    short_res_flat = (act_pred_1 - act).reshape(B, D)
    weighted_short_res = (geo_scale * short_res_flat).reshape_as(act_pred_1)
    loss_geo_short = torch.mean(
        get_norm(weighted_short_res / (1 - t_star), norm_type)
    )

    # MIP base + geometry-aware auxiliary on the short branch
    loss = loss_coarse + loss_short + geo_weight * loss_geo_short
    loss = loss_scale * loss

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "w_self_mean": float(weights.diagonal().mean().detach()),
        "local_var_mean": float(local_var.mean().detach()),
        "geo_scale_mean": float(geo_scale.mean().detach()),
        "geo_scale_max_mean": float(geo_scale.max(dim=-1).values.mean().detach()),
        "geo_scale_min_mean": float(geo_scale.min(dim=-1).values.mean().detach()),
    }

    return loss, aux


def drifting_policy_loss6(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift6:
    MIP skeleton + anisotropy-gated geometry on both short and coarse branches

    Compared with Drift5:
    - geometry only penalizes dimensions with geo_scale > 1
    - geometry is applied to both short branch and coarse/deployment branch
    - adds retrieval/locality diagnostics to aux
    """

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    # Safe defaults so you can run without editing config first
    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    # ------------------------------------------------------------------
    # 1) Same two-branch MIP skeleton
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # 2) Batch-wise soft conditional weights from observation features
    # ------------------------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=1e-6)

    sim = obs_feat @ obs_feat.transpose(0, 1)                  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    logits = (sim - sim_mean) / sim_std
    weights = logits.softmax(dim=1)                           # [B, B], self included

    # ------------------------------------------------------------------
    # 3) Paired-centered local diagonal geometry
    #    local_var_i[d] = sum_j w_ij * (a_j^*[d] - a_i^*[d])^2
    # ------------------------------------------------------------------
    act_flat = act.reshape(B, D).detach()
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)      # [B, B, D], a_j - a_i

    local_var = (weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1)   # [B, D]

    # geo_scale has per-sample mean ~1 by construction, so geo_scale_mean is not informative
    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    geo_scale = torch.sqrt(local_var_mean / (local_var + geo_eps))
    geo_scale = geo_scale / geo_scale.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    geo_scale = geo_scale.detach()

    # Only keep the anisotropic "excess" above isotropic baseline
    geo_excess = F.relu(geo_scale - 1.0).detach()                  # [B, D]

    # ------------------------------------------------------------------
    # 4) Base MIP losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    # ------------------------------------------------------------------
    # 5) Geometry-augmented losses on both branches
    # ------------------------------------------------------------------
    coarse_res_flat = (act_pred_0 - act).reshape(B, D)
    short_res_flat = (act_pred_1 - act).reshape(B, D)

    weighted_coarse_res = (geo_excess * coarse_res_flat).reshape_as(act_pred_0)
    weighted_short_res = (geo_excess * short_res_flat).reshape_as(act_pred_1)

    loss_geo_coarse = torch.mean(
        get_norm(weighted_coarse_res / t_star, norm_type)
    )
    loss_geo_short = torch.mean(
        get_norm(weighted_short_res / (1 - t_star), norm_type)
    )

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_geo_coarse
    )
    loss = loss_scale * loss

    # ------------------------------------------------------------------
    # 6) Retrieval / locality diagnostics
    # ------------------------------------------------------------------
    # effective support size of each row of weights
    n_eff = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)

    # top-k mass concentration
    top1_mass = weights.max(dim=1).values
    k5 = min(5, B)
    k10 = min(10, B)
    top5_mass = weights.topk(k5, dim=1).values.sum(dim=1)
    top10_mass = weights.topk(k10, dim=1).values.sum(dim=1)

    # self vs nearest non-self similarity margin
    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    sim_nonself = sim.masked_fill(eye_mask, -1e9)
    max_nonself = sim_nonself.max(dim=1).values
    self_next_margin = sim.diagonal() - max_nonself

    # weighted action-space neighborhood radius
    pair_l2 = pair_diff.norm(dim=-1)                             # [B, B]
    neighbor_radius_l2 = (weights * pair_l2).sum(dim=1)

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_geo_coarse": float(loss_geo_coarse.detach()),
        # "loss_drift": 0.0,
        # "loss_distill": 0.0,

        "w_self_mean": float(weights.diagonal().mean().detach()),
        "n_eff_mean": float(n_eff.mean().detach()),
        "top1_mass_mean": float(top1_mass.mean().detach()),
        "top5_mass_mean": float(top5_mass.mean().detach()),
        "top10_mass_mean": float(top10_mass.mean().detach()),
        "self_next_margin_mean": float(self_next_margin.mean().detach()),
        "neighbor_radius_l2_mean": float(neighbor_radius_l2.mean().detach()),

        "local_var_mean": float(local_var.mean().detach()),
        "geo_scale_mean": float(geo_scale.mean().detach()),   # expected to stay ~1
        "geo_scale_min_mean": float(geo_scale.min(dim=-1).values.mean().detach()),
        "geo_scale_max_mean": float(geo_scale.max(dim=-1).values.mean().detach()),

        "geo_excess_mean": float(geo_excess.mean().detach()),
        "geo_excess_max_mean": float(geo_excess.max(dim=-1).values.mean().detach()),
        "geo_excess_active_frac": float((geo_excess > 0).float().mean().detach()),
    }

    return loss, aux


def _metricdrift_build_conditional_weights(
    obs_emb: torch.Tensor,
    act_flat: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Build the same observation-conditioned soft neighborhood used by Drift6,
    but compute diagnostics without materializing B x B x D tensors.
    """
    B = act_flat.shape[0]
    device = act_flat.device

    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=eps)

    sim = obs_feat @ obs_feat.transpose(0, 1)
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    logits = (sim - sim_mean) / sim_std
    weights = logits.softmax(dim=1)

    n_eff = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)
    top1_mass = weights.max(dim=1).values
    top5_mass = weights.topk(min(5, B), dim=1).values.sum(dim=1)
    top10_mass = weights.topk(min(10, B), dim=1).values.sum(dim=1)

    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    sim_nonself = sim.masked_fill(eye_mask, -1e9)
    max_nonself = sim_nonself.max(dim=1).values
    self_next_margin = sim.diagonal() - max_nonself

    act_sq = act_flat.pow(2).sum(dim=-1, keepdim=True)
    pair_sqdist = (act_sq + act_sq.transpose(0, 1) - 2.0 * (act_flat @ act_flat.transpose(0, 1))).clamp_min(0.0)
    pair_l2 = pair_sqdist.sqrt()
    neighbor_radius_l2 = (weights * pair_l2).sum(dim=1)

    stats = {
        "w_self_mean": weights.diagonal().mean().detach(),
        "n_eff_mean": n_eff.mean().detach(),
        "top1_mass_mean": top1_mass.mean().detach(),
        "top5_mass_mean": top5_mass.mean().detach(),
        "top10_mass_mean": top10_mass.mean().detach(),
        "self_next_margin_mean": self_next_margin.mean().detach(),
        "neighbor_radius_l2_mean": neighbor_radius_l2.mean().detach(),
    }
    return weights, stats


def _metricdrift_refine_weights_with_gt_locality(
    base_weights: torch.Tensor,
    act_flat: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Refine the observation-conditioned neighborhood using GT-action locality.

    This keeps obs similarity as the prior support, but asks the local geometry
    estimator to focus more tightly on samples whose expert actions are truly
    close to the current sample's expert action.
    """
    act_sq = act_flat.pow(2).sum(dim=-1, keepdim=True)
    pair_sqdist = (
        act_sq + act_sq.transpose(0, 1) - 2.0 * (act_flat @ act_flat.transpose(0, 1))
    ).clamp_min(0.0)
    pair_l2 = pair_sqdist.sqrt()

    alpha = (base_weights * pair_sqdist).sum(dim=1, keepdim=True).clamp_min(eps)
    refine = torch.exp(-pair_sqdist / alpha)
    ref_weights = base_weights * refine
    ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(eps)

    ref_n_eff = 1.0 / ref_weights.pow(2).sum(dim=1).clamp_min(1e-12)
    ref_top1 = ref_weights.max(dim=1).values
    ref_top5 = ref_weights.topk(min(5, ref_weights.shape[0]), dim=1).values.sum(dim=1)
    ref_top10 = ref_weights.topk(min(10, ref_weights.shape[0]), dim=1).values.sum(dim=1)
    ref_radius = (ref_weights * pair_l2).sum(dim=1)

    stats = {
        "alpha_mean": alpha.mean().detach(),
        "ref_n_eff_mean": ref_n_eff.mean().detach(),
        "ref_top1_mass_mean": ref_top1.mean().detach(),
        "ref_top5_mass_mean": ref_top5.mean().detach(),
        "ref_top10_mass_mean": ref_top10.mean().detach(),
        "ref_neighbor_radius_l2_mean": ref_radius.mean().detach(),
    }
    return ref_weights, stats


def _metricdrift_build_diag_metric(
    weights: torch.Tensor,
    act_flat: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Minimal version:
    sample-specific local diagonal precision minus task-global diagonal baseline.
    """
    second_diag = weights @ act_flat.pow(2)
    local_mean = weights @ act_flat
    local_var = (second_diag - 2.0 * local_mean * act_flat + act_flat.pow(2)).clamp_min(eps)

    global_mean = act_flat.mean(dim=0, keepdim=True)
    global_var = (act_flat - global_mean).pow(2).mean(dim=0).clamp_min(eps)

    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(eps)
    global_var_mean = global_var.mean().clamp_min(eps)

    local_scale = torch.sqrt(local_var_mean / (local_var + eps))
    local_scale = local_scale / local_scale.mean(dim=-1, keepdim=True).clamp_min(eps)

    global_scale = torch.sqrt(global_var_mean / (global_var + eps))
    global_scale = global_scale / global_scale.mean().clamp_min(eps)

    metric_excess = F.relu(
        local_scale / global_scale.unsqueeze(0).clamp_min(eps) - 1.0
    ).detach()
    metric_diag = (1.0 + metric_excess).detach()

    stats = {
        "local_var_mean": local_var.mean().detach(),
        "global_var_mean": global_var.mean().detach(),
        "local_scale_min_mean": local_scale.min(dim=-1).values.mean().detach(),
        "local_scale_max_mean": local_scale.max(dim=-1).values.mean().detach(),
        "global_scale_min": global_scale.min().detach(),
        "global_scale_max": global_scale.max().detach(),
        "metric_excess_mean": metric_excess.mean().detach(),
        "metric_excess_max_mean": metric_excess.max(dim=-1).values.mean().detach(),
        "metric_excess_active_frac": (metric_excess > 0).float().mean().detach(),
    }
    return metric_diag, stats


def _metricdrift_stable_cholesky(
    cov: torch.Tensor,
    eye: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Robust Cholesky with adaptive diagonal jitter.
    Keeps the matrix semantics unchanged unless numerical issues appear.
    """
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    chol, info = torch.linalg.cholesky_ex(cov)
    if torch.all(info == 0):
        return cov, chol

    jitter = max(eps, 1e-6)
    for _ in range(6):
        cov_try = cov + jitter * eye
        chol, info = torch.linalg.cholesky_ex(cov_try)
        if torch.all(info == 0):
            return cov_try, chol
        jitter *= 10.0

    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(jitter)
    cov_psd = torch.einsum("...ij,...j,...kj->...ik", eigvecs, eigvals, eigvecs)
    cov_psd = 0.5 * (cov_psd + cov_psd.transpose(-1, -2))
    chol = torch.linalg.cholesky(cov_psd)
    return cov_psd, chol


def _metricdrift_build_matrix_metric(
    weights: torch.Tensor,
    act_flat: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Matrix version:
    local PSD precision excess over the task-global covariance baseline.
    """
    B, D = act_flat.shape
    device = act_flat.device
    dtype = act_flat.dtype
    eye = torch.eye(D, device=device, dtype=dtype)

    local_mean = weights @ act_flat
    act_outer = act_flat.unsqueeze(-1) * act_flat.unsqueeze(-2)
    local_second = torch.einsum("ij,jde->ide", weights, act_outer)

    ref_outer = act_flat.unsqueeze(-1) * act_flat.unsqueeze(-2)
    local_cov = (
        local_second
        - local_mean.unsqueeze(-1) * act_flat.unsqueeze(-2)
        - act_flat.unsqueeze(-1) * local_mean.unsqueeze(-2)
        + ref_outer
    )
    local_cov = 0.5 * (local_cov + local_cov.transpose(-1, -2))
    local_cov = local_cov + eps * eye.unsqueeze(0)

    global_mean = act_flat.mean(dim=0, keepdim=True)
    centered = act_flat - global_mean
    global_cov = centered.transpose(0, 1) @ centered / act_flat.shape[0]
    global_cov = 0.5 * (global_cov + global_cov.transpose(0, 1))
    global_cov = global_cov + eps * eye

    local_cov, local_chol = _metricdrift_stable_cholesky(
        local_cov, eye.unsqueeze(0), eps=eps
    )
    local_inv = torch.cholesky_inverse(local_chol)

    global_cov, global_chol = _metricdrift_stable_cholesky(
        global_cov, eye, eps=eps
    )
    global_inv = torch.cholesky_inverse(global_chol)

    local_trace = local_cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(eps)
    global_trace = global_cov.diagonal().mean().clamp_min(eps)

    local_prec = local_trace.view(B, 1, 1) * local_inv
    global_prec = global_trace * global_inv

    delta_prec = 0.5 * (
        (local_prec - global_prec.unsqueeze(0))
        + (local_prec - global_prec.unsqueeze(0)).transpose(-1, -2)
    )
    eigvals, eigvecs = torch.linalg.eigh(delta_prec)
    pos_eigvals = F.relu(eigvals)
    metric_excess = torch.einsum("bij,bj,bkj->bik", eigvecs, pos_eigvals, eigvecs)
    metric_matrix = eye.unsqueeze(0) + metric_excess

    local_var = local_cov.diagonal(dim1=-2, dim2=-1)
    global_var = global_cov.diagonal()
    diag_cov = torch.diag_embed(local_var)
    offdiag_norm = (local_cov - diag_cov).pow(2).sum(dim=(-1, -2)).sqrt()
    full_norm = local_cov.pow(2).sum(dim=(-1, -2)).sqrt().clamp_min(eps)
    cov_offdiag_ratio = offdiag_norm / full_norm

    stats = {
        "local_var_mean": local_var.mean().detach(),
        "global_var_mean": global_var.mean().detach(),
        "cov_offdiag_ratio_mean": cov_offdiag_ratio.mean().detach(),
        "matrix_excess_trace_mean": metric_excess.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().detach(),
        "matrix_excess_max_eig_mean": pos_eigvals.max(dim=-1).values.mean().detach(),
        "matrix_excess_rank_mean": (pos_eigvals > 1e-6).float().sum(dim=-1).mean().detach(),
        "matrix_excess_active_frac": (pos_eigvals > 1e-6).float().mean().detach(),
    }
    return metric_matrix.detach(), stats


def _metricdrift_l2_quadratic(
    residual_flat: torch.Tensor,
    metric_matrix: torch.Tensor,
):
    return torch.einsum("bi,bij,bj->b", residual_flat, metric_matrix, residual_flat)


def _drift6min_build_batch(
    config,
    flow_map,
    encoder,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    if config.norm_type != "l2":
        raise NotImplementedError("drift6min variants currently require norm_type='l2'.")

    t_star = config.t_two_step
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=device)
    noise = torch.empty_like(act).normal_(0, 1)


    obs_emb = encoder(obs, None)
    # act_pred_0 = flow_map.get_velocity(s, noise, obs_emb)       ## 4.26：random start
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)       ## 正常情况

    # act_t = act + (1 - t_star) * noise      ## 正常情况
    act_t = act_pred_0 + (1 - t_star) * noise   ##**4.26：在输出附近构造

    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    act_flat = act.reshape(B, D).detach()
    weights, weight_stats = _metricdrift_build_conditional_weights(
        obs_emb, act_flat, eps=eps
    )
    metric_diag, metric_stats = _metricdrift_build_diag_metric(weights, act_flat, eps=eps)

    coarse_res_flat = (act_pred_0 - act).reshape(B, D)
    short_res_flat = (act_pred_1 - act).reshape(B, D)

    coarse_loss_i = (metric_diag * coarse_res_flat.pow(2)).sum(dim=-1) / (t_star * t_star)
    short_loss_i = (metric_diag * short_res_flat.pow(2)).sum(dim=-1) / (
        (1 - t_star) * (1 - t_star)
    )

    coarse_flat = act_pred_0.reshape(B, D).detach()
    short_flat = act_pred_1.reshape(B, D).detach()
    proposal_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    correction_step = short_flat - coarse_flat
    direct_gt = act_flat - coarse_flat
    correction_progress = (
        proposal_to_gt_l2 - short_to_gt_l2
    ) / proposal_to_gt_l2.clamp_min(eps)
    correction_align = F.cosine_similarity(
        correction_step, direct_gt, dim=-1, eps=eps
    )

    aux = {
        "proposal_to_gt_l2_mean": float(proposal_to_gt_l2.mean().detach()),
        "coarse_to_gt_l2_mean": float(proposal_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_align_cos_mean": float(correction_align.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < proposal_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in weight_stats.items()})
    aux.update({k: float(v) for k, v in metric_stats.items()})

    return {
        "loss_scale": loss_scale,
        "coarse_loss_i": coarse_loss_i,
        "short_loss_i": short_loss_i,
        "correction_progress": correction_progress.detach(),
        "aux": aux,
    }


def drift6min_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Implicit Drift6 minimal version:
    Drift6 rewritten as a conditional diagonal contraction metric over the
    one-step proposal, with task-global diagonal anisotropy removed first.
    """
    del interp
    batch = _drift6min_build_batch(
        config, flow_map, encoder, act, obs, delta_t
    )
    loss_coarse = torch.mean(batch["coarse_loss_i"])
    loss_short = torch.mean(batch["short_loss_i"])
    loss = batch["loss_scale"] * (loss_coarse + loss_short)

    aux = batch["aux"].copy()
    aux.update(
        {
            "loss_coarse": float(loss_coarse.detach()),
            "loss_short": float(loss_short.detach()),
        }
    )
    return loss, aux


def drift6min_awshort_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift6min + advantage-weighted short supervision.

    Short loss gets more weight on samples where the short branch already shows
    a detached advantage over the coarse one-step proposal.
    """
    del interp
    batch = _drift6min_build_batch(
        config, flow_map, encoder, act, obs, delta_t
    )

    loss_coarse = torch.mean(batch["coarse_loss_i"])
    loss_short = torch.mean(batch["short_loss_i"])

    short_adv = F.relu(batch["correction_progress"])
    short_weight = 1.0 + short_adv
    short_weight = short_weight / short_weight.mean().clamp_min(1e-6)
    loss_short_aw = torch.mean(short_weight * batch["short_loss_i"])

    loss = batch["loss_scale"] * (loss_coarse + loss_short_aw)

    aux = batch["aux"].copy()
    aux.update(
        {
            "loss_coarse": float(loss_coarse.detach()),
            "loss_short": float(loss_short.detach()),
            "loss_short_aw": float(loss_short_aw.detach()),
            "short_adv_mean": float(short_adv.mean().detach()),
            "short_adv_max_mean": float(short_adv.max().detach()),
            "short_adv_active_frac": float(
                (short_adv > 0).float().mean().detach()
            ),
            "aw_weight_max_mean": float(short_weight.max().detach()),
        }
    )
    return loss, aux


def drift6min_rebal_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift6min + global branch rebalancing.

    Reweights the short branch globally according to its detached scale relative
    to the coarse branch, without changing per-sample neighborhood geometry.
    """
    del interp
    batch = _drift6min_build_batch(
        config, flow_map, encoder, act, obs, delta_t
    )

    loss_coarse = torch.mean(batch["coarse_loss_i"])
    loss_short = torch.mean(batch["short_loss_i"])
    short_scale = torch.sqrt(
        loss_coarse.detach() / loss_short.detach().clamp_min(1e-6)
    )
    loss_short_rebal = short_scale * loss_short

    loss = batch["loss_scale"] * (loss_coarse + loss_short_rebal)

    aux = batch["aux"].copy()
    aux.update(
        {
            "loss_coarse": float(loss_coarse.detach()),
            "loss_short": float(loss_short.detach()),
            "loss_short_rebal": float(loss_short_rebal.detach()),
            "short_rebal_scale": float(short_scale.detach()),
        }
    )
    return loss, aux


def drift6matrix_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Implicit Drift6 matrix version:
    the same contraction idea as drift6min, but with a PSD matrix-valued local
    excess metric over a task-global covariance baseline.
    """
    del interp
    if config.norm_type != "l2":
        raise NotImplementedError("drift6matrix currently requires norm_type='l2'.")

    t_star = config.t_two_step
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    act_flat = act.reshape(B, D).detach()
    weights, weight_stats = _metricdrift_build_conditional_weights(obs_emb, act_flat, eps=eps)
    metric_matrix, metric_stats = _metricdrift_build_matrix_metric(weights, act_flat, eps=eps)

    coarse_res_flat = (act_pred_0 - act).reshape(B, D) / t_star
    short_res_flat = (act_pred_1 - act).reshape(B, D) / (1 - t_star)

    loss_coarse = torch.mean(_metricdrift_l2_quadratic(coarse_res_flat, metric_matrix))
    loss_short = torch.mean(_metricdrift_l2_quadratic(short_res_flat, metric_matrix))
    loss = loss_scale * (loss_coarse + loss_short)

    coarse_flat = act_pred_0.reshape(B, D).detach()
    short_flat = act_pred_1.reshape(B, D).detach()
    proposal_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    correction_step = short_flat - coarse_flat
    direct_gt = act_flat - coarse_flat
    correction_progress = (
        proposal_to_gt_l2 - short_to_gt_l2
    ) / proposal_to_gt_l2.clamp_min(eps)
    correction_align = F.cosine_similarity(
        correction_step, direct_gt, dim=-1, eps=eps
    )

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "proposal_to_gt_l2_mean": float(proposal_to_gt_l2.mean().detach()),
        "coarse_to_gt_l2_mean": float(proposal_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_align_cos_mean": float(correction_align.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < proposal_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in weight_stats.items()})
    aux.update({k: float(v) for k, v in metric_stats.items()})
    return loss, aux


def globaldiag_policy_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    GlobalDiag:
    MIP skeleton + global diagonal anisotropy prior.
    Retrieval metrics are logged only for diagnosis and do not affect loss.
    """

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    # ------------------------------------------------------------------
    # 1) MIP two-branch skeleton
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # 2) Global diagonal geometry from batch expert actions
    # ------------------------------------------------------------------
    act_flat = act.reshape(B, D).detach()

    global_var = act_flat.var(dim=0, unbiased=False).clamp_min(geo_eps)   # [D]
    global_var_mean = global_var.mean().clamp_min(geo_eps)

    geo_scale = torch.sqrt(global_var_mean / global_var)                  # [D]
    geo_scale = geo_scale / geo_scale.mean().clamp_min(geo_eps)
    geo_scale = geo_scale.detach()

    geo_excess = F.relu(geo_scale - 1.0).detach()                         # [D]

    # ------------------------------------------------------------------
    # 3) Base MIP losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    # ------------------------------------------------------------------
    # 4) Geometry losses on both branches
    # ------------------------------------------------------------------
    coarse_res_flat = (act_pred_0 - act).reshape(B, D)
    short_res_flat = (act_pred_1 - act).reshape(B, D)

    weighted_coarse_res = (geo_excess.unsqueeze(0) * coarse_res_flat).reshape_as(act_pred_0)
    weighted_short_res = (geo_excess.unsqueeze(0) * short_res_flat).reshape_as(act_pred_1)

    loss_geo_coarse = torch.mean(get_norm(weighted_coarse_res / t_star, norm_type))
    loss_geo_short = torch.mean(get_norm(weighted_short_res / (1 - t_star), norm_type))

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_geo_coarse
    )
    loss = loss_scale * loss

    # ------------------------------------------------------------------
    # 5) Retrieval/locality diagnostics only
    # ------------------------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=1e-6)

    sim = obs_feat @ obs_feat.transpose(0, 1)                  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    logits = (sim - sim_mean) / sim_std
    weights = logits.softmax(dim=1)                           # [B, B], diag included

    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)  # [B, B, D], a_j - a_i
    pair_l2 = pair_diff.norm(dim=-1)                           # [B, B]

    n_eff = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)
    top1_mass = weights.max(dim=1).values
    top5_mass = weights.topk(min(5, B), dim=1).values.sum(dim=1)
    top10_mass = weights.topk(min(10, B), dim=1).values.sum(dim=1)

    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    sim_nonself = sim.masked_fill(eye_mask, -1e9)
    max_nonself = sim_nonself.max(dim=1).values
    self_next_margin = sim.diagonal() - max_nonself

    neighbor_radius_l2 = (weights * pair_l2).sum(dim=1)

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_geo_coarse": float(loss_geo_coarse.detach()),
        "loss_drift": 0.0,
        "loss_distill": 0.0,

        # same keys as ARL for easy dashboard comparison
        "base_w_self_mean": float(weights.diagonal().mean().detach()),
        "ref_w_self_mean": float(weights.diagonal().mean().detach()),
        "base_n_eff_mean": float(n_eff.mean().detach()),
        "ref_n_eff_mean": float(n_eff.mean().detach()),
        "base_top1_mass_mean": float(top1_mass.mean().detach()),
        "ref_top1_mass_mean": float(top1_mass.mean().detach()),
        "base_top5_mass_mean": float(top5_mass.mean().detach()),
        "ref_top5_mass_mean": float(top5_mass.mean().detach()),
        "base_top10_mass_mean": float(top10_mass.mean().detach()),
        "ref_top10_mass_mean": float(top10_mass.mean().detach()),
        "self_next_margin_mean": float(self_next_margin.mean().detach()),
        "base_neighbor_radius_l2_mean": float(neighbor_radius_l2.mean().detach()),
        "ref_neighbor_radius_l2_mean": float(neighbor_radius_l2.mean().detach()),

        "local_var_mean": float(global_var.mean().detach()),
        "geo_scale_mean": float(geo_scale.mean().detach()),
        "geo_scale_min_mean": float(geo_scale.min().detach()),
        "geo_scale_max_mean": float(geo_scale.max().detach()),
        "geo_excess_mean": float(geo_excess.mean().detach()),
        "geo_excess_max_mean": float(geo_excess.max().detach()),
        "geo_excess_active_frac": float((geo_excess > 0).float().mean().detach()),
    }

    return loss, aux


def arl_policy_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    ARL = Action-Refined Local geometry

    Steps:
    1) observation weights u_ij from encoder features
    2) refine with paired expert-action distances to get w_ij^ref
    3) build diagonal local geometry from refined weights
    4) use V6-style geo_excess losses on both short and coarse branches
    """

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    # ------------------------------------------------------------------
    # 1) MIP two-branch skeleton
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # 2) Base observation weights u_ij
    # ------------------------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=1e-6)

    sim = obs_feat @ obs_feat.transpose(0, 1)                  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    logits = (sim - sim_mean) / sim_std
    base_weights = logits.softmax(dim=1)                       # [B, B], self included

    # ------------------------------------------------------------------
    # 3) Action-refined local weights
    # ------------------------------------------------------------------
    act_flat = act.reshape(B, D).detach()
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)          # [B, B, D], a_j - a_i
    pair_sqdist = pair_diff.pow(2).mean(dim=-1)                        # [B, B]
    pair_l2 = pair_diff.norm(dim=-1)                                   # [B, B]

    # self-calibrated action-local scale alpha_i
    alpha = (base_weights * pair_sqdist).sum(dim=1, keepdim=True).clamp_min(geo_eps)  # [B, 1]

    refine = torch.exp(-pair_sqdist / alpha)                           # [B, B]
    ref_weights = base_weights * refine
    ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(geo_eps)

    # ------------------------------------------------------------------
    # 4) Refined diagonal local geometry
    # ------------------------------------------------------------------
    local_var = (ref_weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1)   # [B, D]

    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    geo_scale = torch.sqrt(local_var_mean / (local_var + geo_eps))
    geo_scale = geo_scale / geo_scale.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    geo_scale = geo_scale.detach()

    geo_excess = F.relu(geo_scale - 1.0).detach()                      # [B, D]

    # ------------------------------------------------------------------
    # 5) Base MIP losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    # ------------------------------------------------------------------
    # 6) Geometry losses on both branches
    # ------------------------------------------------------------------
    coarse_res_flat = (act_pred_0 - act).reshape(B, D)
    short_res_flat = (act_pred_1 - act).reshape(B, D)

    weighted_coarse_res = (geo_excess * coarse_res_flat).reshape_as(act_pred_0)
    weighted_short_res = (geo_excess * short_res_flat).reshape_as(act_pred_1)

    loss_geo_coarse = torch.mean(get_norm(weighted_coarse_res / t_star, norm_type))
    loss_geo_short = torch.mean(get_norm(weighted_short_res / (1 - t_star), norm_type))

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_geo_coarse
    )
    loss = loss_scale * loss

    # ------------------------------------------------------------------
    # 7) Diagnostics: before vs after refinement
    # ------------------------------------------------------------------
    def _weight_stats(weights_):
        n_eff_ = 1.0 / weights_.pow(2).sum(dim=1).clamp_min(1e-12)
        top1_ = weights_.max(dim=1).values
        top5_ = weights_.topk(min(5, B), dim=1).values.sum(dim=1)
        top10_ = weights_.topk(min(10, B), dim=1).values.sum(dim=1)
        radius_ = (weights_ * pair_l2).sum(dim=1)
        return n_eff_, top1_, top5_, top10_, radius_

    base_n_eff, base_top1, base_top5, base_top10, base_radius = _weight_stats(base_weights)
    ref_n_eff, ref_top1, ref_top5, ref_top10, ref_radius = _weight_stats(ref_weights)

    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    sim_nonself = sim.masked_fill(eye_mask, -1e9)
    max_nonself = sim_nonself.max(dim=1).values
    self_next_margin = sim.diagonal() - max_nonself

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_geo_coarse": float(loss_geo_coarse.detach()),
        "loss_drift": 0.0,
        "loss_distill": 0.0,

        "base_w_self_mean": float(base_weights.diagonal().mean().detach()),
        "ref_w_self_mean": float(ref_weights.diagonal().mean().detach()),
        "base_n_eff_mean": float(base_n_eff.mean().detach()),
        "ref_n_eff_mean": float(ref_n_eff.mean().detach()),
        "base_top1_mass_mean": float(base_top1.mean().detach()),
        "ref_top1_mass_mean": float(ref_top1.mean().detach()),
        "base_top5_mass_mean": float(base_top5.mean().detach()),
        "ref_top5_mass_mean": float(ref_top5.mean().detach()),
        "base_top10_mass_mean": float(base_top10.mean().detach()),
        "ref_top10_mass_mean": float(ref_top10.mean().detach()),
        "self_next_margin_mean": float(self_next_margin.mean().detach()),
        "base_neighbor_radius_l2_mean": float(base_radius.mean().detach()),
        "ref_neighbor_radius_l2_mean": float(ref_radius.mean().detach()),

        "alpha_mean": float(alpha.mean().detach()),
        "local_var_mean": float(local_var.mean().detach()),
        "geo_scale_mean": float(geo_scale.mean().detach()),  # expected ~1
        "geo_scale_min_mean": float(geo_scale.min(dim=-1).values.mean().detach()),
        "geo_scale_max_mean": float(geo_scale.max(dim=-1).values.mean().detach()),
        "geo_excess_mean": float(geo_excess.mean().detach()),
        "geo_excess_max_mean": float(geo_excess.max(dim=-1).values.mean().detach()),
        "geo_excess_active_frac": float((geo_excess > 0).float().mean().detach()),
    }

    return loss, aux


def geofuse_policy_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept only for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    GeoFuse:
    global diagonal anisotropy  x  action-refined local geometry

    Intuition:
    - global branch supplies task-level anisotropy prior
    - local branch supplies sample-specific action-local refinement
    - fuse both, then keep only the anisotropic excess above isotropic baseline
    """

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    # ------------------------------------------------------------------
    # 1) MIP skeleton
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # 2) Observation similarity weights (base)
    # ------------------------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=1e-6)

    sim = obs_feat @ obs_feat.transpose(0, 1)                  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    logits = (sim - sim_mean) / sim_std
    base_weights = logits.softmax(dim=1)                       # [B, B], self included

    # ------------------------------------------------------------------
    # 3) Pairwise expert-action geometry
    # ------------------------------------------------------------------
    act_flat = act.reshape(B, D).detach()
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)      # [B, B, D]
    pair_sqdist = pair_diff.pow(2).mean(dim=-1)                    # [B, B]
    pair_l2 = pair_diff.norm(dim=-1)                               # [B, B]

    # ------------------------------------------------------------------
    # 4) Local refinement (ARL)
    # ------------------------------------------------------------------
    alpha = (base_weights * pair_sqdist).sum(dim=1, keepdim=True).clamp_min(geo_eps)   # [B, 1]
    refine = torch.exp(-pair_sqdist / alpha)                                              # [B, B]
    ref_weights = base_weights * refine
    ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(geo_eps)

    # ------------------------------------------------------------------
    # 5) Global diagonal geometry
    # ------------------------------------------------------------------
    global_var = act_flat.var(dim=0, unbiased=False).clamp_min(geo_eps)   # [D]
    global_var_mean = global_var.mean().clamp_min(geo_eps)

    global_geo_scale = torch.sqrt(global_var_mean / global_var)            # [D]
    global_geo_scale = global_geo_scale / global_geo_scale.mean().clamp_min(geo_eps)
    global_geo_scale = global_geo_scale.detach()

    # ------------------------------------------------------------------
    # 6) Local diagonal geometry from refined weights
    # ------------------------------------------------------------------
    local_var = (ref_weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1)   # [B, D]
    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(geo_eps)

    local_geo_scale = torch.sqrt(local_var_mean / (local_var + geo_eps))    # [B, D]
    local_geo_scale = local_geo_scale / local_geo_scale.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    local_geo_scale = local_geo_scale.detach()

    # ------------------------------------------------------------------
    # 7) Fuse global and local geometry
    # ------------------------------------------------------------------
    fused_geo_scale = local_geo_scale * global_geo_scale.unsqueeze(0)        # [B, D]
    fused_geo_scale = fused_geo_scale / fused_geo_scale.mean(dim=-1, keepdim=True).clamp_min(geo_eps)
    fused_geo_scale = fused_geo_scale.detach()

    fused_geo_excess = F.relu(fused_geo_scale - 1.0).detach()                # [B, D]

    # ------------------------------------------------------------------
    # 8) Base MIP losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    # ------------------------------------------------------------------
    # 9) GeoFuse losses on both branches
    # ------------------------------------------------------------------
    coarse_res_flat = (act_pred_0 - act).reshape(B, D)
    short_res_flat = (act_pred_1 - act).reshape(B, D)

    weighted_coarse_res = (fused_geo_excess * coarse_res_flat).reshape_as(act_pred_0)
    weighted_short_res = (fused_geo_excess * short_res_flat).reshape_as(act_pred_1)

    loss_geo_coarse = torch.mean(get_norm(weighted_coarse_res / t_star, norm_type))
    loss_geo_short = torch.mean(get_norm(weighted_short_res / (1 - t_star), norm_type))

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_geo_coarse
    )
    loss = loss_scale * loss

    # ------------------------------------------------------------------
    # 10) Diagnostics
    # ------------------------------------------------------------------
    def _weight_stats(weights_):
        n_eff_ = 1.0 / weights_.pow(2).sum(dim=1).clamp_min(1e-12)
        top1_ = weights_.max(dim=1).values
        top5_ = weights_.topk(min(5, B), dim=1).values.sum(dim=1)
        top10_ = weights_.topk(min(10, B), dim=1).values.sum(dim=1)
        radius_ = (weights_ * pair_l2).sum(dim=1)
        return n_eff_, top1_, top5_, top10_, radius_

    base_n_eff, base_top1, base_top5, base_top10, base_radius = _weight_stats(base_weights)
    ref_n_eff, ref_top1, ref_top5, ref_top10, ref_radius = _weight_stats(ref_weights)

    eye_mask = torch.eye(B, device=device, dtype=torch.bool)
    sim_nonself = sim.masked_fill(eye_mask, -1e9)
    max_nonself = sim_nonself.max(dim=1).values
    self_next_margin = sim.diagonal() - max_nonself

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_geo_coarse": float(loss_geo_coarse.detach()),
        "loss_drift": 0.0,
        "loss_distill": 0.0,

        # locality diagnostics
        "base_w_self_mean": float(base_weights.diagonal().mean().detach()),
        "ref_w_self_mean": float(ref_weights.diagonal().mean().detach()),
        "base_n_eff_mean": float(base_n_eff.mean().detach()),
        "ref_n_eff_mean": float(ref_n_eff.mean().detach()),
        "base_top1_mass_mean": float(base_top1.mean().detach()),
        "ref_top1_mass_mean": float(ref_top1.mean().detach()),
        "base_top5_mass_mean": float(base_top5.mean().detach()),
        "ref_top5_mass_mean": float(ref_top5.mean().detach()),
        "base_top10_mass_mean": float(base_top10.mean().detach()),
        "ref_top10_mass_mean": float(ref_top10.mean().detach()),
        "self_next_margin_mean": float(self_next_margin.mean().detach()),
        "base_neighbor_radius_l2_mean": float(base_radius.mean().detach()),
        "ref_neighbor_radius_l2_mean": float(ref_radius.mean().detach()),

        # geometry diagnostics
        "alpha_mean": float(alpha.mean().detach()),
        "global_var_mean": float(global_var.mean().detach()),
        "local_var_mean": float(local_var.mean().detach()),

        "global_geo_scale_mean": float(global_geo_scale.mean().detach()),   # expected ~1
        "global_geo_scale_min_mean": float(global_geo_scale.min().detach()),
        "global_geo_scale_max_mean": float(global_geo_scale.max().detach()),

        "local_geo_scale_mean": float(local_geo_scale.mean().detach()),     # expected ~1
        "local_geo_scale_min_mean": float(local_geo_scale.min(dim=-1).values.mean().detach()),
        "local_geo_scale_max_mean": float(local_geo_scale.max(dim=-1).values.mean().detach()),

        "fused_geo_scale_mean": float(fused_geo_scale.mean().detach()),     # expected ~1
        "fused_geo_scale_min_mean": float(fused_geo_scale.min(dim=-1).values.mean().detach()),
        "fused_geo_scale_max_mean": float(fused_geo_scale.max(dim=-1).values.mean().detach()),

        "fused_geo_excess_mean": float(fused_geo_excess.mean().detach()),
        "fused_geo_excess_max_mean": float(fused_geo_excess.max(dim=-1).values.mean().detach()),
        "fused_geo_excess_active_frac": float((fused_geo_excess > 0).float().mean().detach()),
    }

    return loss, aux


def _geofuse_shared_geometry(
    obs_emb: torch.Tensor,
    act: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Shared GeoFuse geometry builder.

    Returns:
        geom: dict containing
            - base_weights
            - ref_weights
            - global_geo_scale
            - local_geo_scale
            - fused_geo_scale
            - fused_geo_excess
            - global_var
            - local_var
            - alpha
        stats: dict of tensor stats for logging
    """
    B = act.shape[0]
    act_flat = act.reshape(B, -1).detach()
    D = act_flat.shape[1]
    device = act.device

    # ------------------------------------------------------------------
    # Observation-space base weights
    # ------------------------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=eps)

    sim = obs_feat @ obs_feat.transpose(0, 1)  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    logits = (sim - sim_mean) / sim_std
    base_weights = logits.softmax(dim=1)  # [B, B], self included

    # ------------------------------------------------------------------
    # Pairwise expert-action geometry
    # pair_diff[i, j] = a_j - a_i
    # ------------------------------------------------------------------
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)   # [B, B, D]
    pair_sqdist = pair_diff.pow(2).mean(dim=-1)                 # [B, B]
    pair_l2 = pair_diff.norm(dim=-1)                            # [B, B]

    # ------------------------------------------------------------------
    # Action-local refinement
    # ------------------------------------------------------------------
    alpha = (base_weights * pair_sqdist).sum(dim=1, keepdim=True).clamp_min(eps)  # [B, 1]
    refine = torch.exp(-pair_sqdist / alpha)                                        # [B, B]

    ref_weights = base_weights * refine
    ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(eps)

    # ------------------------------------------------------------------
    # Global diagonal geometry
    # ------------------------------------------------------------------
    global_var = act_flat.var(dim=0, unbiased=False).clamp_min(eps)  # [D]
    global_var_mean = global_var.mean().clamp_min(eps)

    global_geo_scale = torch.sqrt(global_var_mean / global_var)      # [D]
    global_geo_scale = global_geo_scale / global_geo_scale.mean().clamp_min(eps)
    global_geo_scale = global_geo_scale.detach()

    # ------------------------------------------------------------------
    # Local diagonal geometry
    # ------------------------------------------------------------------
    local_var = (ref_weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1).clamp_min(eps)  # [B, D]
    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(eps)

    local_geo_scale = torch.sqrt(local_var_mean / local_var)  # [B, D]
    local_geo_scale = local_geo_scale / local_geo_scale.mean(dim=-1, keepdim=True).clamp_min(eps)
    local_geo_scale = local_geo_scale.detach()

    # ------------------------------------------------------------------
    # GeoFuse
    # ------------------------------------------------------------------
    fused_geo_scale = local_geo_scale * global_geo_scale.unsqueeze(0)   # [B, D]
    fused_geo_scale = fused_geo_scale / fused_geo_scale.mean(dim=-1, keepdim=True).clamp_min(eps)
    fused_geo_scale = fused_geo_scale.detach()

    fused_geo_excess = F.relu(fused_geo_scale - 1.0).detach()           # [B, D]

    # ------------------------------------------------------------------
    # Shared diagnostics
    # ------------------------------------------------------------------
    def _weight_stats(weights_):
        n_eff = 1.0 / weights_.pow(2).sum(dim=1).clamp_min(1e-12)
        top1 = weights_.max(dim=1).values
        top5 = weights_.topk(min(5, B), dim=1).values.sum(dim=1)
        top10 = weights_.topk(min(10, B), dim=1).values.sum(dim=1)
        radius = (weights_ * pair_l2).sum(dim=1)
        return n_eff, top1, top5, top10, radius

    base_n_eff, base_top1, base_top5, base_top10, base_radius = _weight_stats(base_weights)
    ref_n_eff, ref_top1, ref_top5, ref_top10, ref_radius = _weight_stats(ref_weights)

    if B > 1:
        eye_mask = torch.eye(B, device=device, dtype=torch.bool)
        sim_nonself = sim.masked_fill(eye_mask, -1e9)
        self_next_margin = sim.diagonal() - sim_nonself.max(dim=1).values
    else:
        self_next_margin = torch.zeros((B,), device=device, dtype=sim.dtype)

    geom = {
        "base_weights": base_weights.detach(),
        "ref_weights": ref_weights.detach(),
        "global_geo_scale": global_geo_scale.detach(),
        "local_geo_scale": local_geo_scale.detach(),
        "fused_geo_scale": fused_geo_scale.detach(),
        "fused_geo_excess": fused_geo_excess.detach(),
        "global_var": global_var.detach(),
        "local_var": local_var.detach(),
        "alpha": alpha.detach(),
    }

    stats = {
        "base_w_self_mean": base_weights.diagonal().mean().detach(),
        "ref_w_self_mean": ref_weights.diagonal().mean().detach(),
        "base_n_eff_mean": base_n_eff.mean().detach(),
        "ref_n_eff_mean": ref_n_eff.mean().detach(),
        "base_top1_mass_mean": base_top1.mean().detach(),
        "ref_top1_mass_mean": ref_top1.mean().detach(),
        "base_top5_mass_mean": base_top5.mean().detach(),
        "ref_top5_mass_mean": ref_top5.mean().detach(),
        "base_top10_mass_mean": base_top10.mean().detach(),
        "ref_top10_mass_mean": ref_top10.mean().detach(),
        "self_next_margin_mean": self_next_margin.mean().detach(),
        "base_neighbor_radius_l2_mean": base_radius.mean().detach(),
        "ref_neighbor_radius_l2_mean": ref_radius.mean().detach(),
        "global_var_mean": global_var.mean().detach(),
        "local_var_mean": local_var.mean().detach(),
        "global_geo_scale_max_mean": global_geo_scale.max().detach(),
        "local_geo_scale_max_mean": local_geo_scale.max(dim=-1).values.mean().detach(),
        "fused_geo_scale_max_mean": fused_geo_scale.max(dim=-1).values.mean().detach(),
        "fused_geo_excess_mean": fused_geo_excess.mean().detach(),
        "fused_geo_excess_max_mean": fused_geo_excess.max(dim=-1).values.mean().detach(),
        "fused_geo_excess_active_frac": (fused_geo_excess > 0).float().mean().detach(),
    }

    return geom, stats


def geofuse_align_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    GeoFuse-Align:
    - zero-start coarse branch
    - standard MIP noisy short branch
    - shared GeoFuse geometry
    - geometry-aware short loss
    - short-to-coarse geometry-aware alignment
    """
    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()

    # ------------------------------------------------------------------
    # MIP skeleton
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # Shared GeoFuse geometry
    # ------------------------------------------------------------------
    geom, stats = _geofuse_shared_geometry(obs_emb, act, eps=geo_eps)
    fused_geo_excess = geom["fused_geo_excess"]  # [B, D]

    # ------------------------------------------------------------------
    # Base losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    # ------------------------------------------------------------------
    # Geometry-aware short loss
    # ------------------------------------------------------------------
    short_res_flat = (act_pred_1 - act).reshape(B, D)
    weighted_short_res = (fused_geo_excess * short_res_flat).reshape_as(act_pred_1)

    loss_geo_short = torch.mean(
        get_norm(weighted_short_res / (1 - t_star), norm_type)
    )

    # ------------------------------------------------------------------
    # Geometry-aware short -> coarse alignment
    # coarse is pulled toward short branch only on geometry-important dims
    # ------------------------------------------------------------------
    align_flat = fused_geo_excess * (
        act_pred_0.reshape(B, D) - act_pred_1.reshape(B, D).detach()
    )
    align_res = align_flat.reshape_as(act_pred_0)

    loss_align = torch.mean(
        get_norm(align_res / t_star, norm_type)
    )

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_align
    )
    loss = loss_scale * loss

    # ------------------------------------------------------------------
    # Extra diagnostics for Align
    # ------------------------------------------------------------------
    coarse_short_gap = (
        act_pred_0.reshape(B, D).detach() - act_pred_1.reshape(B, D).detach()
    )
    coarse_short_gap_mean = coarse_short_gap.norm(dim=-1).mean()

    coarse_short_gap_excess = (fused_geo_excess * coarse_short_gap).norm(dim=-1).mean()

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_align": float(loss_align.detach()),
        "loss_geo_coarse": 0.0,
        "loss_drift": 0.0,
        "loss_distill": 0.0,
        "coarse_seed_sensitivity_mean": 0.0,
        "coarse_short_gap_mean": float(coarse_short_gap_mean.detach()),
        "coarse_short_gap_excess_mean": float(coarse_short_gap_excess.detach()),
    }

    aux.update({k: float(v) for k, v in stats.items()})
    return loss, aux


def geofuse_noise_loss(
    config,
    flow_map,
    encoder,
    interp,      # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    GeoFuse-Noise:
    - noise-start coarse branch
    - standard MIP noisy short branch
    - shared GeoFuse geometry
    - geometry-aware short loss
    - geometry-aware coarse loss
    - train/infer both use noise-start
    """
    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()

    # ------------------------------------------------------------------
    # MIP skeleton, but coarse branch starts from pure noise
    # ------------------------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    coarse_noise = torch.empty_like(act).normal_(0, 1)
    short_noise = torch.empty_like(act).normal_(0, 1)

    act_t = act + (1 - t_star) * short_noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, coarse_noise, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # ------------------------------------------------------------------
    # Shared GeoFuse geometry
    # ------------------------------------------------------------------
    geom, stats = _geofuse_shared_geometry(obs_emb, act, eps=geo_eps)
    fused_geo_excess = geom["fused_geo_excess"]  # [B, D]

    # ------------------------------------------------------------------
    # Base losses
    # ------------------------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    # ------------------------------------------------------------------
    # Geometry-aware short loss
    # ------------------------------------------------------------------
    short_res_flat = (act_pred_1 - act).reshape(B, D)
    weighted_short_res = (fused_geo_excess * short_res_flat).reshape_as(act_pred_1)

    loss_geo_short = torch.mean(
        get_norm(weighted_short_res / (1 - t_star), norm_type)
    )

    # ------------------------------------------------------------------
    # Geometry-aware coarse loss
    # ------------------------------------------------------------------
    coarse_res_flat = (act_pred_0 - act).reshape(B, D)
    weighted_coarse_res = (fused_geo_excess * coarse_res_flat).reshape_as(act_pred_0)

    loss_geo_coarse = torch.mean(
        get_norm(weighted_coarse_res / t_star, norm_type)
    )

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_geo_coarse
    )
    loss = loss_scale * loss

    # ------------------------------------------------------------------
    # Extra diagnostics for Noise
    # ------------------------------------------------------------------
    with torch.no_grad():
        obs_emb_det = obs_emb.detach()

        z1 = torch.empty_like(act).normal_(0, 1)
        z2 = torch.empty_like(act).normal_(0, 1)

        pred1 = flow_map.get_velocity(s, z1, obs_emb_det)
        pred2 = flow_map.get_velocity(s, z2, obs_emb_det)

        coarse_seed_sensitivity = (
            pred1.reshape(B, D) - pred2.reshape(B, D)
        ).norm(dim=-1).mean()

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_geo_coarse": float(loss_geo_coarse.detach()),
        "loss_align": 0.0,
        "loss_drift": 0.0,
        "loss_distill": 0.0,
        "coarse_seed_sensitivity_mean": float(coarse_seed_sensitivity.detach()),
        "coarse_short_gap_mean": 0.0,
        "coarse_short_gap_excess_mean": 0.0,
    }

    aux.update({k: float(v) for k, v in stats.items()})
    return loss, aux


def _drifting7_build_teacher(
    obs_emb: torch.Tensor,
    act: torch.Tensor,
    short_pred: torch.Tensor,
    eps: float = 1e-6,
):
    """Build GT-anchored, proposal-conditioned teacher for Drifting7."""
    B = act.shape[0]
    act_flat = act.reshape(B, -1).detach()
    proposal_flat = short_pred.reshape(B, -1).detach()

    # --------------------------------------------------
    # 1. Conditional prior from observation embeddings
    # --------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=eps)

    sim = obs_feat @ obs_feat.transpose(0, 1)  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    logits = (sim - sim_mean) / sim_std
    base_weights = logits.softmax(dim=1)  # [B, B]

    # --------------------------------------------------
    # 2. Proposal-centered refinement around short branch
    # --------------------------------------------------
    prop_diff = act_flat.unsqueeze(0) - proposal_flat.unsqueeze(1)  # [B, B, D]
    prop_sqdist = prop_diff.pow(2).mean(dim=-1)  # [B, B]
    prop_l2 = prop_diff.norm(dim=-1)  # [B, B]

    alpha = (base_weights * prop_sqdist).sum(dim=1, keepdim=True).clamp_min(eps)
    refine = torch.exp(-prop_sqdist / alpha)
    ref_weights = base_weights * refine
    ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(eps)

    # --------------------------------------------------
    # 3. Proposal-centered neighborhood only estimates anisotropy.
    #    Teacher itself stays anchored to each sample's own GT.
    # --------------------------------------------------
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)  # [B, B, D], a_j - a_i
    local_var = (ref_weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1)  # [B, D]
    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(eps)

    # rho in (0, 1): stable dims move more, loose dims move less
    rho = local_var_mean / (local_var + local_var_mean + eps)  # [B, D]

    teacher_flat = proposal_flat + rho * (act_flat - proposal_flat)
    teacher_flat = teacher_flat.detach()

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    def _weight_stats(weights_):
        n_eff_ = 1.0 / weights_.pow(2).sum(dim=1).clamp_min(1e-12)
        top1_ = weights_.max(dim=1).values
        top5_ = weights_.topk(min(5, B), dim=1).values.sum(dim=1)
        top10_ = weights_.topk(min(10, B), dim=1).values.sum(dim=1)
        radius_ = (weights_ * prop_l2).sum(dim=1)
        return n_eff_, top1_, top5_, top10_, radius_

    base_n_eff, base_top1, base_top5, base_top10, base_radius = _weight_stats(base_weights)
    ref_n_eff, ref_top1, ref_top5, ref_top10, ref_radius = _weight_stats(ref_weights)

    proposal_to_gt_l2 = (proposal_flat - act_flat).norm(dim=-1)
    teacher_to_gt_l2 = (teacher_flat - act_flat).norm(dim=-1)
    teacher_shift_l2 = (teacher_flat - proposal_flat).norm(dim=-1)

    direct_gt = act_flat - proposal_flat
    teacher_step = teacher_flat - proposal_flat
    teacher_align = F.cosine_similarity(teacher_step, direct_gt, dim=-1, eps=eps)
    teacher_step_ratio = teacher_shift_l2 / proposal_to_gt_l2.clamp_min(eps)
    teacher_progress = (proposal_to_gt_l2 - teacher_to_gt_l2) / proposal_to_gt_l2.clamp_min(eps)

    stats = {
        "base_w_self_mean": base_weights.diagonal().mean().detach(),
        "ref_w_self_mean": ref_weights.diagonal().mean().detach(),
        "base_n_eff_mean": base_n_eff.mean().detach(),
        "ref_n_eff_mean": ref_n_eff.mean().detach(),
        "base_top1_mass_mean": base_top1.mean().detach(),
        "ref_top1_mass_mean": ref_top1.mean().detach(),
        "base_top5_mass_mean": base_top5.mean().detach(),
        "ref_top5_mass_mean": ref_top5.mean().detach(),
        "base_top10_mass_mean": base_top10.mean().detach(),
        "ref_top10_mass_mean": ref_top10.mean().detach(),
        "base_neighbor_radius_l2_mean": base_radius.mean().detach(),
        "ref_neighbor_radius_l2_mean": ref_radius.mean().detach(),
        "alpha_mean": alpha.mean().detach(),
        "teacher_shift_l2_mean": teacher_shift_l2.mean().detach(),
        "proposal_to_gt_l2_mean": proposal_to_gt_l2.mean().detach(),
        "teacher_to_gt_l2_mean": teacher_to_gt_l2.mean().detach(),
        "teacher_better_frac": (teacher_to_gt_l2 < proposal_to_gt_l2).float().mean().detach(),
        "teacher_progress_mean": teacher_progress.mean().detach(),
        "teacher_step_ratio_mean": teacher_step_ratio.mean().detach(),
        "teacher_align_cos_mean": teacher_align.mean().detach(),
        "rho_mean": rho.mean().detach(),
        "rho_min_mean": rho.min(dim=-1).values.mean().detach(),
        "rho_max_mean": rho.max(dim=-1).values.mean().detach(),
        "local_var_mean": local_var.mean().detach(),
    }
    return teacher_flat, stats


def drifting_policy_loss7(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift7:
    MIP skeleton + GT-anchored, proposal-conditioned anisotropic teacher.

    Core idea:
    - short branch proposes where we currently are
    - batch expert actions define a conditional neighborhood around that proposal
    - the neighborhood only estimates which dims should trust GT more
    - coarse branch amortizes that anisotropic correction in one step
    """
    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    # --------------------------------------------------
    # 1. Standard MIP two-branch skeleton
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # --------------------------------------------------
    # 2. Proposal-conditioned local teacher
    # --------------------------------------------------
    teacher_flat, stats = _drifting7_build_teacher(
        obs_emb=obs_emb,
        act=act,
        short_pred=act_pred_1,
        eps=geo_eps,
    )
    teacher = teacher_flat.reshape_as(act)

    # --------------------------------------------------
    # 3. Losses
    # --------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))
    loss_transport = torch.mean(get_norm((act_pred_0 - teacher) / t_star, norm_type))

    loss = loss_coarse + loss_short + loss_transport
    loss = loss_scale * loss

    coarse_flat = act_pred_0.reshape(act.shape[0], -1).detach()
    act_flat = act.reshape(act.shape[0], -1).detach()
    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    coarse_to_teacher_l2 = (coarse_flat - teacher_flat).norm(dim=-1)

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_transport": float(loss_transport.detach()),
        "loss_drift": 0.0,
        "loss_distill": float(loss_transport.detach()),
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "coarse_to_teacher_l2_mean": float(coarse_to_teacher_l2.mean().detach()),
    }
    aux.update({k: float(v) for k, v in stats.items()})
    return loss, aux


def _drifting8_build_teacher(
    obs_emb: torch.Tensor,
    act: torch.Tensor,
    short_pred: torch.Tensor,
    eps: float = 1e-6,
):
    """Build GT-anchored, proposal-conditioned matrix teacher for Drifting8."""
    B = act.shape[0]
    act_flat = act.reshape(B, -1).detach()
    proposal_flat = short_pred.reshape(B, -1).detach()
    D = act_flat.shape[1]
    device = act.device
    dtype = act.dtype
    eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)

    # --------------------------------------------------
    # 1. Conditional prior from observation embeddings
    # --------------------------------------------------
    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=eps)

    sim = obs_feat @ obs_feat.transpose(0, 1)  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    logits = (sim - sim_mean) / sim_std
    base_weights = logits.softmax(dim=1)  # [B, B]

    # --------------------------------------------------
    # 2. Proposal-centered refinement around short branch
    # --------------------------------------------------
    prop_diff = act_flat.unsqueeze(0) - proposal_flat.unsqueeze(1)  # [B, B, D], a_j - y_i
    prop_sqdist = prop_diff.pow(2).mean(dim=-1)  # [B, B]
    prop_l2 = prop_diff.norm(dim=-1)  # [B, B]

    alpha = (base_weights * prop_sqdist).sum(dim=1, keepdim=True).clamp_min(eps)
    refine = torch.exp(-prop_sqdist / alpha)
    ref_weights = base_weights * refine
    ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(eps)

    # --------------------------------------------------
    # 3. GT-anchored local covariance around current sample
    # --------------------------------------------------
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)  # [B, B, D], a_j - a_i
    local_cov = torch.einsum("ijd,ije,ij->ide", pair_diff, pair_diff, ref_weights)
    local_cov = local_cov + eps * eye

    local_var = local_cov.diagonal(dim1=-2, dim2=-1)  # [B, D]
    lambda_bar = local_var.mean(dim=-1, keepdim=True).clamp_min(eps)  # [B, 1]

    # Matrix shrinkage gate: eigenvalues in (0, 1)
    solve_mat = local_cov + lambda_bar.unsqueeze(-1) * eye  # [B, D, D]
    chol, info = torch.linalg.cholesky_ex(solve_mat)
    if torch.any(info > 0):
        solve_mat = solve_mat + 1e-4 * eye
        chol = torch.linalg.cholesky(solve_mat)

    direct_gt = (act_flat - proposal_flat).unsqueeze(-1)  # [B, D, 1]
    matrix_step = lambda_bar.view(B, 1, 1) * torch.cholesky_solve(direct_gt, chol)
    matrix_step = matrix_step.squeeze(-1)  # [B, D]
    teacher_flat = (proposal_flat + matrix_step).detach()

    # Diagonal counterpart for diagnostics
    rho_diag = lambda_bar / (local_var + lambda_bar + eps)  # [B, D]
    diag_teacher_flat = (proposal_flat + rho_diag * (act_flat - proposal_flat)).detach()
    diag_step = diag_teacher_flat - proposal_flat

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    def _weight_stats(weights_):
        n_eff_ = 1.0 / weights_.pow(2).sum(dim=1).clamp_min(1e-12)
        top1_ = weights_.max(dim=1).values
        top5_ = weights_.topk(min(5, B), dim=1).values.sum(dim=1)
        top10_ = weights_.topk(min(10, B), dim=1).values.sum(dim=1)
        radius_ = (weights_ * prop_l2).sum(dim=1)
        return n_eff_, top1_, top5_, top10_, radius_

    base_n_eff, base_top1, base_top5, base_top10, base_radius = _weight_stats(base_weights)
    ref_n_eff, ref_top1, ref_top5, ref_top10, ref_radius = _weight_stats(ref_weights)

    proposal_to_gt_l2 = (proposal_flat - act_flat).norm(dim=-1)
    teacher_to_gt_l2 = (teacher_flat - act_flat).norm(dim=-1)
    diag_teacher_to_gt_l2 = (diag_teacher_flat - act_flat).norm(dim=-1)

    teacher_progress = (proposal_to_gt_l2 - teacher_to_gt_l2) / proposal_to_gt_l2.clamp_min(eps)
    diag_teacher_progress = (proposal_to_gt_l2 - diag_teacher_to_gt_l2) / proposal_to_gt_l2.clamp_min(eps)

    teacher_shift_l2 = matrix_step.norm(dim=-1)
    diag_teacher_shift_l2 = diag_step.norm(dim=-1)

    teacher_align = F.cosine_similarity(matrix_step, act_flat - proposal_flat, dim=-1, eps=eps)
    diag_teacher_align = F.cosine_similarity(diag_step, act_flat - proposal_flat, dim=-1, eps=eps)
    matrix_diag_align = F.cosine_similarity(matrix_step, diag_step, dim=-1, eps=eps)

    teacher_step_ratio = teacher_shift_l2 / proposal_to_gt_l2.clamp_min(eps)
    diag_teacher_step_ratio = diag_teacher_shift_l2 / proposal_to_gt_l2.clamp_min(eps)

    diag_cov = torch.diag_embed(local_var)
    offdiag_norm = (local_cov - diag_cov).pow(2).sum(dim=(-1, -2)).sqrt()
    full_norm = local_cov.pow(2).sum(dim=(-1, -2)).sqrt().clamp_min(eps)
    cov_offdiag_ratio = offdiag_norm / full_norm

    stats = {
        "base_w_self_mean": base_weights.diagonal().mean().detach(),
        "ref_w_self_mean": ref_weights.diagonal().mean().detach(),
        "base_n_eff_mean": base_n_eff.mean().detach(),
        "ref_n_eff_mean": ref_n_eff.mean().detach(),
        "base_top1_mass_mean": base_top1.mean().detach(),
        "ref_top1_mass_mean": ref_top1.mean().detach(),
        "base_top5_mass_mean": base_top5.mean().detach(),
        "ref_top5_mass_mean": ref_top5.mean().detach(),
        "base_top10_mass_mean": base_top10.mean().detach(),
        "ref_top10_mass_mean": ref_top10.mean().detach(),
        "base_neighbor_radius_l2_mean": base_radius.mean().detach(),
        "ref_neighbor_radius_l2_mean": ref_radius.mean().detach(),
        "alpha_mean": alpha.mean().detach(),
        "proposal_to_gt_l2_mean": proposal_to_gt_l2.mean().detach(),
        "teacher_to_gt_l2_mean": teacher_to_gt_l2.mean().detach(),
        "diag_teacher_to_gt_l2_mean": diag_teacher_to_gt_l2.mean().detach(),
        "teacher_better_frac": (teacher_to_gt_l2 < proposal_to_gt_l2).float().mean().detach(),
        "diag_teacher_better_frac": (diag_teacher_to_gt_l2 < proposal_to_gt_l2).float().mean().detach(),
        "matrix_over_diag_better_frac": (teacher_to_gt_l2 < diag_teacher_to_gt_l2).float().mean().detach(),
        "teacher_progress_mean": teacher_progress.mean().detach(),
        "diag_teacher_progress_mean": diag_teacher_progress.mean().detach(),
        "teacher_align_cos_mean": teacher_align.mean().detach(),
        "diag_teacher_align_cos_mean": diag_teacher_align.mean().detach(),
        "matrix_diag_align_cos_mean": matrix_diag_align.mean().detach(),
        "teacher_step_ratio_mean": teacher_step_ratio.mean().detach(),
        "diag_teacher_step_ratio_mean": diag_teacher_step_ratio.mean().detach(),
        "teacher_shift_l2_mean": teacher_shift_l2.mean().detach(),
        "diag_teacher_shift_l2_mean": diag_teacher_shift_l2.mean().detach(),
        "rho_mean": rho_diag.mean().detach(),
        "rho_min_mean": rho_diag.min(dim=-1).values.mean().detach(),
        "rho_max_mean": rho_diag.max(dim=-1).values.mean().detach(),
        "local_var_mean": local_var.mean().detach(),
        "lambda_bar_mean": lambda_bar.mean().detach(),
        "cov_offdiag_ratio_mean": cov_offdiag_ratio.mean().detach(),
    }
    return teacher_flat, stats


def drifting_policy_loss8(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift8:
    MIP skeleton + GT-anchored, proposal-conditioned matrix teacher.

    Core idea:
    - short branch proposes where we currently are
    - batch expert actions define a conditional neighborhood around that proposal
    - local covariance captures coupled correction directions
    - coarse branch amortizes the matrix-shrinkage teacher in one step
    """
    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    # --------------------------------------------------
    # 1. Standard MIP two-branch skeleton
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - t_star) * noise

    obs_emb = encoder(obs, None)

    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # --------------------------------------------------
    # 2. Proposal-conditioned matrix teacher
    # --------------------------------------------------
    teacher_flat, stats = _drifting8_build_teacher(
        obs_emb=obs_emb,
        act=act,
        short_pred=act_pred_1,
        eps=geo_eps,
    )
    teacher = teacher_flat.reshape_as(act)

    # --------------------------------------------------
    # 3. Losses
    # --------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))
    loss_transport = torch.mean(get_norm((act_pred_0 - teacher) / t_star, norm_type))

    loss = loss_coarse + loss_short + loss_transport
    loss = loss_scale * loss

    coarse_flat = act_pred_0.reshape(act.shape[0], -1).detach()
    act_flat = act.reshape(act.shape[0], -1).detach()
    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    coarse_to_teacher_l2 = (coarse_flat - teacher_flat).norm(dim=-1)

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_transport": float(loss_transport.detach()),
        "loss_drift": 0.0,
        "loss_distill": float(loss_transport.detach()),
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "coarse_to_teacher_l2_mean": float(coarse_to_teacher_l2.mean().detach()),
    }
    aux.update({k: float(v) for k, v in stats.items()})
    return loss, aux


def _drifting9_build_geometry(
    obs_emb: torch.Tensor,
    act: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Build the same light conditional geometry used by Drift6.

    Geometry is only used as a metric/gate. It never becomes a teacher target.
    """
    B = act.shape[0]
    D = act[0].numel()
    device = act.device

    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=eps)

    sim = obs_feat @ obs_feat.transpose(0, 1)  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    logits = (sim - sim_mean) / sim_std
    weights = logits.softmax(dim=1)  # [B, B]

    act_flat = act.reshape(B, D).detach()
    pair_diff = act_flat.unsqueeze(0) - act_flat.unsqueeze(1)  # [B, B, D], a_j - a_i
    local_var = (weights.unsqueeze(-1) * pair_diff.pow(2)).sum(dim=1)  # [B, D]

    local_var_mean = local_var.mean(dim=-1, keepdim=True).clamp_min(eps)
    local_radius_rms = local_var_mean.sqrt().detach()
    geo_scale = torch.sqrt(local_var_mean / (local_var + eps))
    geo_scale = geo_scale / geo_scale.mean(dim=-1, keepdim=True).clamp_min(eps)
    geo_scale = geo_scale.detach()
    geo_excess = F.relu(geo_scale - 1.0).detach()
    align_gate = (1.0 + geo_excess).detach()
    align_gate = align_gate / align_gate.mean(dim=-1, keepdim=True).clamp_min(eps)

    n_eff = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)
    top1_mass = weights.max(dim=1).values
    top5_mass = weights.topk(min(5, B), dim=1).values.sum(dim=1)
    top10_mass = weights.topk(min(10, B), dim=1).values.sum(dim=1)

    if B > 1:
        eye_mask = torch.eye(B, device=device, dtype=torch.bool)
        sim_nonself = sim.masked_fill(eye_mask, -1e9)
        self_next_margin = sim.diagonal() - sim_nonself.max(dim=1).values
    else:
        self_next_margin = torch.zeros((B,), device=device, dtype=sim.dtype)

    pair_l2 = pair_diff.norm(dim=-1)
    neighbor_radius_l2 = (weights * pair_l2).sum(dim=1)

    stats = {
        "w_self_mean": weights.diagonal().mean().detach(),
        "n_eff_mean": n_eff.mean().detach(),
        "top1_mass_mean": top1_mass.mean().detach(),
        "top5_mass_mean": top5_mass.mean().detach(),
        "top10_mass_mean": top10_mass.mean().detach(),
        "self_next_margin_mean": self_next_margin.mean().detach(),
        "neighbor_radius_l2_mean": neighbor_radius_l2.mean().detach(),
        "local_var_mean": local_var.mean().detach(),
        "local_radius_rms_mean": local_radius_rms.mean().detach(),
        "geo_scale_mean": geo_scale.mean().detach(),
        "geo_scale_min_mean": geo_scale.min(dim=-1).values.mean().detach(),
        "geo_scale_max_mean": geo_scale.max(dim=-1).values.mean().detach(),
        "geo_excess_mean": geo_excess.mean().detach(),
        "geo_excess_max_mean": geo_excess.max(dim=-1).values.mean().detach(),
        "geo_excess_active_frac": (geo_excess > 0).float().mean().detach(),
        "align_gate_mean": align_gate.mean().detach(),
        "align_gate_min_mean": align_gate.min(dim=-1).values.mean().detach(),
        "align_gate_max_mean": align_gate.max(dim=-1).values.mean().detach(),
    }
    return geo_excess, align_gate, local_radius_rms, stats


def _drifting9_build_q_input(
    proposal: torch.Tensor,
    noise: torch.Tensor,
    act: torch.Tensor,
    local_radius_rms: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Proposal-centered local neighborhood for the short branch.

    The radius is a middle ground between proposal error scale and conditional
    geometry scale. This avoids both failure modes we already saw:
    over-collapsing into tiny late-stage edits, and exploding into a nearly
    fixed-width denoising cloud.
    """
    proposal_flat = proposal.reshape(proposal.shape[0], -1).detach()
    act_flat = act.reshape(act.shape[0], -1).detach()
    noise_flat = noise.reshape(noise.shape[0], -1)

    proposal_res = act_flat - proposal_flat
    proposal_rms = proposal_res.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    local_radius_rms = local_radius_rms.detach().clamp_min(eps)
    q_radius_rms = torch.sqrt(proposal_rms * local_radius_rms).clamp_min(eps)
    q_flat = proposal_flat + q_radius_rms * noise_flat
    q = q_flat.reshape_as(proposal)

    local_geom_over_error = local_radius_rms / proposal_rms.clamp_min(eps)
    local_geom_wider = (local_radius_rms > proposal_rms).float()

    stats = {
        "proposal_error_rms_mean": proposal_rms.mean().detach(),
        "proposal_error_rms_max": proposal_rms.max().detach(),
        "proposal_local_geom_rms_mean": local_radius_rms.mean().detach(),
        "proposal_local_geom_over_error_mean": local_geom_over_error.mean().detach(),
        "proposal_local_geom_wider_frac": local_geom_wider.mean().detach(),
        # Compatibility aliases for the previous dashboard names.
        "proposal_local_floor_rms_mean": local_radius_rms.mean().detach(),
        "proposal_local_floor_active_frac": local_geom_wider.mean().detach(),
        "proposal_noise_rms_mean": q_radius_rms.mean().detach(),
        "proposal_noise_rms_max": q_radius_rms.max().detach(),
        "proposal_noise_over_error_mean": (
            q_radius_rms / proposal_rms.clamp_min(eps)
        ).mean().detach(),
        "proposal_noise_over_geom_mean": (
            q_radius_rms / local_radius_rms.clamp_min(eps)
        ).mean().detach(),
    }
    return q, stats


def drifting_policy_loss9(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift9:
    - coarse branch stays a clean one-step GT predictor
    - short branch becomes a proposal-conditioned local corrector
    - q-side radius is a blended scale between coarse error and conditional
      geometry, so it neither collapses nor explodes
    - geometry only gates which dimensions are worth aligning back to coarse
    - proposal locality is enforced on q-side, but amortization stays in the
      safer state-alignment regime instead of treating an absolute-action head
      as if it predicted a clean residual
    """
    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale

    geo_short_weight = getattr(config, "drifting_geo_short_weight", 0.25)
    geo_coarse_weight = getattr(config, "drifting_geo_coarse_weight", 0.5)
    geo_eps = getattr(config, "drifting_geo_eps", 1e-6)

    B = act.shape[0]
    D = act[0].numel()

    # --------------------------------------------------
    # 1. Coarse branch: same strict one-step deployment object
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)

    # --------------------------------------------------
    # 2. Geometry stays a light gate, and also provides a q-radius floor
    # --------------------------------------------------
    geo_excess, align_gate, local_radius_rms, geo_stats = _drifting9_build_geometry(
        obs_emb, act, eps=geo_eps
    )

    # --------------------------------------------------
    # 3. Short branch: proposal-conditioned local correction
    # --------------------------------------------------
    q_noise = torch.empty_like(act).normal_(0, 1)
    act_q, q_stats = _drifting9_build_q_input(
        proposal=act_pred_0,
        noise=q_noise,
        act=act,
        local_radius_rms=local_radius_rms,
        eps=geo_eps,
    )
    act_pred_1 = flow_map.get_velocity(t, act_q, obs_emb)

    # --------------------------------------------------
    # 4. Losses
    # --------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))

    short_res_flat = (act_pred_1 - act).reshape(B, D)
    weighted_short_res = (geo_excess * short_res_flat).reshape_as(act_pred_1)
    loss_geo_short = torch.mean(
        get_norm(weighted_short_res / (1 - t_star), norm_type)
    )

    proposal_flat_live = act_pred_0.reshape(B, D)
    proposal_flat = proposal_flat_live.detach()
    q_flat = act_q.reshape(B, D).detach()
    short_flat = act_pred_1.reshape(B, D).detach()

    align_flat = align_gate * (proposal_flat_live - short_flat)
    loss_align = torch.mean(
        get_norm(align_flat.reshape_as(act_pred_0) / t_star, norm_type)
    )

    loss = (
        loss_coarse
        + loss_short
        + geo_short_weight * loss_geo_short
        + geo_coarse_weight * loss_align
    )
    loss = loss_scale * loss

    # --------------------------------------------------
    # 5. Diagnostics: branch gap and correction progress
    # --------------------------------------------------
    coarse_flat = proposal_flat
    act_flat = act.reshape(B, D).detach()

    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    q_input_to_gt_l2 = (q_flat - act_flat).norm(dim=-1)
    q_input_to_proposal_l2 = (q_flat - proposal_flat).norm(dim=-1)

    direct_gt = act_flat - proposal_flat
    correction_step = short_flat - proposal_flat
    correction_step_l2 = correction_step.norm(dim=-1)
    correction_align = F.cosine_similarity(
        correction_step, direct_gt, dim=-1, eps=geo_eps
    )
    correction_step_ratio = correction_step_l2 / coarse_to_gt_l2.clamp_min(geo_eps)
    correction_progress = (
        coarse_to_gt_l2 - short_to_gt_l2
    ) / coarse_to_gt_l2.clamp_min(geo_eps)
    proposal_locality_ratio = q_input_to_proposal_l2 / coarse_to_gt_l2.clamp_min(geo_eps)

    coarse_short_gap = coarse_flat - short_flat
    coarse_short_gap_l2 = coarse_short_gap.norm(dim=-1)
    coarse_short_gap_excess = (geo_excess * coarse_short_gap).norm(dim=-1)
    coarse_short_gap_gate = (align_gate * coarse_short_gap).norm(dim=-1)

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_geo_short": float(loss_geo_short.detach()),
        "loss_align": float(loss_align.detach()),
        "loss_geo_coarse": 0.0,
        "loss_transport": 0.0,
        "loss_drift": 0.0,
        "loss_distill": 0.0,
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "proposal_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "alignment_target_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "q_input_to_gt_l2_mean": float(q_input_to_gt_l2.mean().detach()),
        "q_input_to_proposal_l2_mean": float(q_input_to_proposal_l2.mean().detach()),
        "proposal_locality_ratio_mean": float(proposal_locality_ratio.mean().detach()),
        "correction_step_l2_mean": float(correction_step_l2.mean().detach()),
        "correction_step_ratio_mean": float(correction_step_ratio.mean().detach()),
        "correction_align_cos_mean": float(correction_align.mean().detach()),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < coarse_to_gt_l2).float().mean().detach()
        ),
        "coarse_short_gap_mean": float(coarse_short_gap_l2.mean().detach()),
        "coarse_short_gap_excess_mean": float(coarse_short_gap_excess.mean().detach()),
        "coarse_short_gap_gate_mean": float(coarse_short_gap_gate.mean().detach()),
    }
    aux.update({k: float(v) for k, v in q_stats.items()})
    aux.update({k: float(v) for k, v in geo_stats.items()})
    return loss, aux


def _drifting10_build_obs_weights(
    obs_emb: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Minimal conditional weights from observation features.

    Batch structure is only used to define a local family of model errors.
    No geometry targets or per-dimension anisotropy are constructed here.
    """
    B = obs_emb.shape[0]
    device = obs_emb.device

    obs_feat = obs_emb.reshape(B, -1).detach()
    obs_feat = F.normalize(obs_feat, dim=-1, eps=eps)

    sim = obs_feat @ obs_feat.transpose(0, 1)  # [B, B]
    sim_mean = sim.mean(dim=1, keepdim=True)
    sim_std = sim.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    logits = (sim - sim_mean) / sim_std

    if B > 1:
        eye_mask = torch.eye(B, device=device, dtype=torch.bool)
        weights = logits.masked_fill(eye_mask, float("-inf")).softmax(dim=1)
    else:
        weights = torch.ones((B, B), device=device, dtype=logits.dtype)

    n_eff = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)
    top1_mass = weights.max(dim=1).values

    stats = {
        "n_eff_mean": n_eff.mean().detach(),
        "top1_mass_mean": top1_mass.mean().detach(),
    }
    return weights, stats


def _drifting10_build_error_family_input(
    proposal: torch.Tensor,
    act: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Build a proposal-conditioned noisy input from the local family of coarse errors.

    Each sample borrows one residual direction from a conditionally similar
    neighbor, then rescales that direction to the current sample's coarse error
    magnitude. This keeps C2 focused on realistic failure modes of the policy.
    """
    B = proposal.shape[0]
    D = proposal[0].numel()

    proposal_flat = proposal.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    current_residual = act_flat - proposal_flat
    sampled_idx = torch.multinomial(weights, num_samples=1).squeeze(-1)
    borrowed_residual = current_residual.index_select(0, sampled_idx)

    current_scale = current_residual.norm(dim=-1, keepdim=True).clamp_min(eps)
    borrowed_scale = borrowed_residual.norm(dim=-1, keepdim=True).clamp_min(eps)

    delta_flat = current_scale * borrowed_residual / borrowed_scale
    q_flat = proposal_flat + delta_flat
    q = q_flat.reshape_as(proposal)

    shared_dir = weights @ current_residual
    borrowed_align = F.cosine_similarity(
        borrowed_residual, current_residual, dim=-1, eps=eps
    )
    shared_align = F.cosine_similarity(shared_dir, current_residual, dim=-1, eps=eps)

    shared_proj_coef = (
        (current_residual * shared_dir).sum(dim=-1, keepdim=True)
        / shared_dir.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    shared_proj = shared_proj_coef * shared_dir
    shared_proj_frac = shared_proj.norm(dim=-1) / current_scale.squeeze(-1).clamp_min(
        eps
    )

    stats = {
        "q_input_to_proposal_l2_mean": delta_flat.norm(dim=-1).mean().detach(),
        "borrowed_error_align_cos_mean": borrowed_align.mean().detach(),
        "shared_dir_align_cos_mean": shared_align.mean().detach(),
        "shared_proj_frac_mean": shared_proj_frac.mean().detach(),
    }
    return q, stats


def drifting_policy_loss10(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift10:
    - coarse branch stays the clean one-step deployment object
    - short branch sees proposal-local perturbations sampled from the local
      family of model errors, not an isotropic action-space neighborhood
    - shared directional structure is logged, but not optimized yet
    """
    del interp

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()

    # --------------------------------------------------
    # 1. Same strict one-step coarse deployment object
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)

    # --------------------------------------------------
    # 2. Observation-conditioned local error family for C2
    # --------------------------------------------------
    weights, weight_stats = _drifting10_build_obs_weights(obs_emb, eps=eps)
    act_q, q_stats = _drifting10_build_error_family_input(
        proposal=act_pred_0,
        act=act,
        weights=weights,
        eps=eps,
    )
    act_pred_1 = flow_map.get_velocity(t, act_q, obs_emb)

    # --------------------------------------------------
    # 3. Minimal two-branch losses
    # --------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))
    loss = loss_scale * (loss_coarse + loss_short)

    # --------------------------------------------------
    # 4. Minimal diagnostics
    # --------------------------------------------------
    coarse_flat = act_pred_0.reshape(B, D).detach()
    short_flat = act_pred_1.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    correction_progress = (
        coarse_to_gt_l2 - short_to_gt_l2
    ) / coarse_to_gt_l2.clamp_min(eps)

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < coarse_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in weight_stats.items()})
    aux.update({k: float(v) for k, v in q_stats.items()})
    return loss, aux


def _drifting11_build_residual_tube_input(
    proposal: torch.Tensor,
    act: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Build a proposal-centered residual tube for the short branch.

    The main direction always comes from the current sample's own residual
    (GT - coarse proposal). Stochasticity is injected only as orthogonal tube
    noise around that residual ray, so we keep C2 without asking batch neighbors
    to provide the principal correction direction.
    """
    B = proposal.shape[0]
    D = proposal[0].numel()

    proposal_flat = proposal.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    residual = act_flat - proposal_flat
    residual_norm = residual.norm(dim=-1, keepdim=True).clamp_min(eps)
    residual_dir = residual / residual_norm

    lam = torch.rand((B, 1), device=proposal.device, dtype=proposal_flat.dtype)
    noise = torch.empty_like(proposal_flat).normal_(0, 1)
    noise_perp = noise - (noise * residual_dir).sum(dim=-1, keepdim=True) * residual_dir
    noise_perp_norm = noise_perp.norm(dim=-1, keepdim=True).clamp_min(eps)
    noise_perp_unit = noise_perp / noise_perp_norm

    axis_step = lam * residual
    ortho_scale = residual_norm * torch.sqrt((lam * (1.0 - lam)).clamp_min(0.0))
    ortho_step = ortho_scale * noise_perp_unit
    delta_flat = axis_step + ortho_step

    q_flat = proposal_flat + delta_flat
    q = q_flat.reshape_as(proposal)

    q_to_gt_l2 = (q_flat - act_flat).norm(dim=-1)
    q_to_prop_l2 = delta_flat.norm(dim=-1)
    axis_align = F.cosine_similarity(delta_flat, residual, dim=-1, eps=eps)
    ortho_ratio = ortho_scale.squeeze(-1) / residual_norm.squeeze(-1).clamp_min(eps)

    stats = {
        "q_input_to_proposal_l2_mean": q_to_prop_l2.mean().detach(),
        "q_input_to_gt_l2_mean": q_to_gt_l2.mean().detach(),
        "tube_lambda_mean": lam.mean().detach(),
        "tube_lambda_std": lam.std(unbiased=False).detach(),
        "tube_axis_align_cos_mean": axis_align.mean().detach(),
        "tube_ortho_ratio_mean": ortho_ratio.mean().detach(),
    }
    return q, stats


def drifting_policy_loss11(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift11:
    - coarse branch stays the clean one-step deployment object
    - short branch sees a sample-wise residual tube around the current coarse
      error ray, instead of batch-provided principal directions
    - keeps the objective minimal: only coarse and short supervision
    """
    del interp

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()

    # --------------------------------------------------
    # 1. Same strict one-step coarse deployment object
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)

    # --------------------------------------------------
    # 2. Sample-wise residual tube for C2
    # --------------------------------------------------
    act_q, q_stats = _drifting11_build_residual_tube_input(
        proposal=act_pred_0,
        act=act,
        eps=eps,
    )
    act_pred_1 = flow_map.get_velocity(t, act_q, obs_emb)

    # --------------------------------------------------
    # 3. Minimal two-branch losses
    # --------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))
    loss = loss_scale * (loss_coarse + loss_short)

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    coarse_flat = act_pred_0.reshape(B, D).detach()
    short_flat = act_pred_1.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    direct_gt = act_flat - coarse_flat
    correction_step = short_flat - coarse_flat
    correction_progress = (
        coarse_to_gt_l2 - short_to_gt_l2
    ) / coarse_to_gt_l2.clamp_min(eps)
    correction_align = F.cosine_similarity(
        correction_step, direct_gt, dim=-1, eps=eps
    )

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "proposal_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_align_cos_mean": float(correction_align.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < coarse_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in q_stats.items()})
    return loss, aux


def drifting_policy_loss12(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift12:
    - keeps Drift11's sample-wise residual tube for the short branch
    - adds the minimal amortization step: coarse aligns to detached short output
    - introduces no new weighting hyperparameter; transport distillation is a
      first-class part of the drifting objective
    """
    del interp

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()

    # --------------------------------------------------
    # 1. Same strict one-step coarse deployment object
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)

    # --------------------------------------------------
    # 2. Sample-wise residual tube for C2
    # --------------------------------------------------
    act_q, q_stats = _drifting11_build_residual_tube_input(
        proposal=act_pred_0,
        act=act,
        eps=eps,
    )
    act_pred_1 = flow_map.get_velocity(t, act_q, obs_emb)

    # --------------------------------------------------
    # 3. Minimal drifting objective with transport distillation
    # --------------------------------------------------
    loss_coarse = torch.mean(get_norm((act_pred_0 - act) / t_star, norm_type))
    loss_short = torch.mean(get_norm((act_pred_1 - act) / (1 - t_star), norm_type))
    short_teacher = act_pred_1.detach()
    loss_transport = torch.mean(
        get_norm((act_pred_0 - short_teacher) / t_star, norm_type)
    )
    loss = loss_scale * (loss_coarse + loss_short + loss_transport)

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    coarse_flat_live = act_pred_0.reshape(B, D)
    coarse_flat = coarse_flat_live.detach()
    short_flat = act_pred_1.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    coarse_to_short_l2 = (coarse_flat - short_flat).norm(dim=-1)
    direct_gt = act_flat - coarse_flat
    correction_step = short_flat - coarse_flat
    correction_progress = (
        coarse_to_gt_l2 - short_to_gt_l2
    ) / coarse_to_gt_l2.clamp_min(eps)
    correction_align = F.cosine_similarity(
        correction_step, direct_gt, dim=-1, eps=eps
    )

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_transport": float(loss_transport.detach()),
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "proposal_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "coarse_to_short_l2_mean": float(coarse_to_short_l2.mean().detach()),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_align_cos_mean": float(correction_align.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < coarse_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in q_stats.items()})
    return loss, aux


def drifting_policy_loss13(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift13:
    - coarse and short share the same field semantics: the network always predicts
      a correction field v(o, x, tau)
    - the start state uses a long-step field target
    - the residual tube still provides local C2 perturbations for short-field learning
    - a fixed-point loss trains the coarse state to match its own terminally
      transported target, staying closer to the original Drifting template
    """
    del interp

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()

    # --------------------------------------------------
    # 1. Long-step field from zero start
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star

    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    field_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    coarse_state = act_0 + t_star * field_pred_0

    # --------------------------------------------------
    # 2. Short-step field on a sample-wise residual tube
    # --------------------------------------------------
    act_q, q_stats = _drifting11_build_residual_tube_input(
        proposal=coarse_state,
        act=act,
        eps=eps,
    )
    field_pred_1 = flow_map.get_velocity(t, act_q, obs_emb)
    short_state = act_q + (1 - t_star) * field_pred_1

    # --------------------------------------------------
    # 3. Unified field objectives
    # --------------------------------------------------
    field_target_0 = (act - act_0) / t_star
    field_target_1 = (act - act_q) / (1 - t_star)

    loss_coarse = torch.mean(get_norm(field_pred_0 - field_target_0, norm_type))
    loss_short = torch.mean(get_norm(field_pred_1 - field_target_1, norm_type))

    with torch.no_grad():
        coarse_state_detached = coarse_state.detach()
        terminal_field = flow_map.get_velocity(t, coarse_state_detached, obs_emb.detach())
        transported_target = coarse_state_detached + (1 - t_star) * terminal_field

    loss_fixed_point = torch.mean(
        get_norm((coarse_state - transported_target) / t_star, norm_type)
    )
    loss = loss_scale * (loss_coarse + loss_short + loss_fixed_point)

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    field_pred_0_flat = field_pred_0.reshape(B, D).detach()
    field_pred_1_flat = field_pred_1.reshape(B, D).detach()
    terminal_field_flat = terminal_field.reshape(B, D).detach()
    coarse_flat = coarse_state.reshape(B, D).detach()
    short_flat = short_state.reshape(B, D).detach()
    transported_flat = transported_target.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    coarse_to_gt_l2 = (coarse_flat - act_flat).norm(dim=-1)
    short_to_gt_l2 = (short_flat - act_flat).norm(dim=-1)
    transported_to_gt_l2 = (transported_flat - act_flat).norm(dim=-1)
    fixed_point_gap_l2 = (coarse_flat - transported_flat).norm(dim=-1)
    direct_gt = act_flat - coarse_flat
    correction_step = short_flat - coarse_flat
    correction_progress = (
        coarse_to_gt_l2 - short_to_gt_l2
    ) / coarse_to_gt_l2.clamp_min(eps)
    correction_align = F.cosine_similarity(
        correction_step, direct_gt, dim=-1, eps=eps
    )

    aux = {
        "loss_coarse": float(loss_coarse.detach()),
        "loss_short": float(loss_short.detach()),
        "loss_fixed_point": float(loss_fixed_point.detach()),
        "coarse_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "proposal_to_gt_l2_mean": float(coarse_to_gt_l2.mean().detach()),
        "short_to_gt_l2_mean": float(short_to_gt_l2.mean().detach()),
        "transported_target_to_gt_l2_mean": float(
            transported_to_gt_l2.mean().detach()
        ),
        "fixed_point_gap_l2_mean": float(fixed_point_gap_l2.mean().detach()),
        "long_field_l2_mean": float(field_pred_0_flat.norm(dim=-1).mean().detach()),
        "short_field_l2_mean": float(field_pred_1_flat.norm(dim=-1).mean().detach()),
        "terminal_field_l2_mean": float(
            terminal_field_flat.norm(dim=-1).mean().detach()
        ),
        "correction_progress_mean": float(correction_progress.mean().detach()),
        "correction_align_cos_mean": float(correction_align.mean().detach()),
        "correction_better_frac": float(
            (short_to_gt_l2 < coarse_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in q_stats.items()})
    return loss, aux


def drifting_policy_loss14(
    config,
    flow_map,
    encoder,
    interp,  # kept for interface compatibility
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Drift14:
    - keep the original network unchanged
    - proposal y = f(o, 0, 0) is the deployed one-step object
    - center transport uses detached proposal input, so short-center teaches a
      local corrector around y without directly pulling y itself toward an
      easier-to-correct intermediate state
    - neighborhood transport f(o, q, t*) teaches local robustness around y
    - fixed-point amortizes the center correction back into y
    """
    del interp

    t_star = config.t_two_step
    norm_type = config.norm_type
    loss_scale = config.loss_scale
    eps = 1e-6

    B = act.shape[0]
    D = act[0].numel()

    # --------------------------------------------------
    # 1. Proposal: strict one-step deployment object
    # --------------------------------------------------
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + t_star
    act_0 = torch.zeros_like(act, device=act.device)

    obs_emb = encoder(obs, None)
    proposal = flow_map.get_velocity(s, act_0, obs_emb)

    # --------------------------------------------------
    # 2. Center transport at the deployed proposal
    # --------------------------------------------------
    proposal_short = proposal.detach()
    transported_center = flow_map.get_velocity(t, proposal_short, obs_emb)
    implicit_field_center = transported_center - proposal

    # --------------------------------------------------
    # 3. Proposal-centered neighborhood transport
    # --------------------------------------------------
    act_q, q_stats = _drifting11_build_residual_tube_input(
        proposal=proposal,
        act=act,
        eps=eps,
    )
    transported_q = flow_map.get_velocity(t, act_q, obs_emb)
    implicit_field_q = transported_q - act_q

    # --------------------------------------------------
    # 4. Proposal / center-short / neighborhood-short / fixed-point objectives
    # --------------------------------------------------
    loss_prop = torch.mean(get_norm((proposal - act) / t_star, norm_type))
    loss_short_center = torch.mean(
        get_norm((transported_center - act) / (1 - t_star), norm_type)
    )
    loss_short_nbr = torch.mean(
        get_norm((transported_q - act) / (1 - t_star), norm_type)
    )
    transported_target = transported_center.detach()

    loss_fixed_point = torch.mean(
        get_norm((proposal - transported_target) / t_star, norm_type)
    )
    loss = loss_scale * (
        loss_prop + loss_short_center + loss_short_nbr + loss_fixed_point
    )

    # --------------------------------------------------
    # 5. Diagnostics: proposal first, then center correction, then neighborhood
    # --------------------------------------------------
    proposal_flat = proposal.reshape(B, D).detach()
    transported_center_flat = transported_center.reshape(B, D).detach()
    implicit_field_center_flat = implicit_field_center.reshape(B, D).detach()
    act_q_flat = act_q.reshape(B, D).detach()
    transported_q_flat = transported_q.reshape(B, D).detach()
    implicit_field_q_flat = implicit_field_q.reshape(B, D).detach()
    transported_target_flat = transported_target.reshape(B, D).detach()
    act_flat = act.reshape(B, D).detach()

    proposal_to_gt_l2 = (proposal_flat - act_flat).norm(dim=-1)
    center_transport_to_gt_l2 = (transported_center_flat - act_flat).norm(dim=-1)
    q_to_gt_l2 = (act_q_flat - act_flat).norm(dim=-1)
    q_transport_to_gt_l2 = (transported_q_flat - act_flat).norm(dim=-1)
    proposal_transport_to_gt_l2 = (transported_target_flat - act_flat).norm(dim=-1)
    fixed_point_gap_l2 = (proposal_flat - transported_target_flat).norm(dim=-1)

    center_correction = transported_center_flat - proposal_flat
    q_correction = transported_q_flat - act_q_flat
    direct_gt = act_flat - proposal_flat
    center_correction_progress = (
        proposal_to_gt_l2 - proposal_transport_to_gt_l2
    ) / proposal_to_gt_l2.clamp_min(eps)
    q_correction_progress = (
        proposal_to_gt_l2 - q_transport_to_gt_l2
    ) / proposal_to_gt_l2.clamp_min(eps)
    center_correction_align = F.cosine_similarity(
        center_correction, direct_gt, dim=-1, eps=eps
    )
    q_correction_align = F.cosine_similarity(
        q_correction, act_flat - act_q_flat, dim=-1, eps=eps
    )

    aux = {
        "loss_coarse": float(loss_prop.detach()),
        "loss_prop": float(loss_prop.detach()),
        "loss_short_center": float(loss_short_center.detach()),
        "loss_short_nbr": float(loss_short_nbr.detach()),
        "loss_fixed_point": float(loss_fixed_point.detach()),
        "proposal_to_gt_l2_mean": float(proposal_to_gt_l2.mean().detach()),
        "coarse_to_gt_l2_mean": float(proposal_to_gt_l2.mean().detach()),
        "center_transport_to_gt_l2_mean": float(
            center_transport_to_gt_l2.mean().detach()
        ),
        "q_to_gt_l2_mean": float(q_to_gt_l2.mean().detach()),
        "q_transport_to_gt_l2_mean": float(q_transport_to_gt_l2.mean().detach()),
        "proposal_transport_to_gt_l2_mean": float(
            proposal_transport_to_gt_l2.mean().detach()
        ),
        "fixed_point_gap_l2_mean": float(fixed_point_gap_l2.mean().detach()),
        "center_field_l2_mean": float(
            implicit_field_center_flat.norm(dim=-1).mean().detach()
        ),
        "q_field_l2_mean": float(implicit_field_q_flat.norm(dim=-1).mean().detach()),
        "center_correction_progress_mean": float(
            center_correction_progress.mean().detach()
        ),
        "q_correction_progress_mean": float(q_correction_progress.mean().detach()),
        "center_correction_align_cos_mean": float(
            center_correction_align.mean().detach()
        ),
        "q_correction_align_cos_mean": float(q_correction_align.mean().detach()),
        "center_correction_better_frac": float(
            (proposal_transport_to_gt_l2 < proposal_to_gt_l2).float().mean().detach()
        ),
        "q_transport_better_frac": float(
            (q_transport_to_gt_l2 < q_to_gt_l2).float().mean().detach()
        ),
    }
    aux.update({k: float(v) for k, v in q_stats.items()})
    return loss, aux


def mip_origin_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss (original form with scale in first iteration)."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = config.t_two_step * act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_target_0 = config.t_two_step * act
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    act_target_1 = act

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act_target_0) / config.t_two_step, config.norm_type)
    loss1 = get_norm(
        (act_pred_1 - act_target_1) / (1 - config.t_two_step), config.norm_type
    )
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def lmd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian map matching loss for distillation."""
    # sample
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    label = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)
    Xst_Is, dt_Xst = flow_map.jvp_t(s, t, Is, label)

    # compute the target velocity field
    b_eval = flow_map.get_reference_velocity(t, Xst_Is, label)

    # lmd loss
    loss = torch.mean(
        (dt_Xst.flatten(start_dim=1) - b_eval.flatten(start_dim=1)) ** 2, dim=-1
    )
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def ctm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Consistency trajectory model loss."""
    # sample
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    s_plus = s + config.discrete_dt
    t = torch.maximum(t, s_plus)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)

    # compute the CTM loss
    Xst_Is_pred = flow_map(s, t, Is, obs_emb)
    b_s = flow_map.get_reference_velocity(s, Is, obs_emb)
    Is_plus = Is + config.discrete_dt * b_s
    Xst_Is_target = flow_map(s_plus, t, Is_plus, obs_emb)
    # make sure loss is not too small
    loss = config.loss_scale * torch.mean(
        ((Xst_Is_target - Xst_Is_pred) / config.discrete_dt) ** 2
    )

    return loss, {}


def psd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Progressive Self-Distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. PSD term that encourages consistency between single-step and multi-step predictions

    The PSD term uses uniform weighting between intermediate steps.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== PSD term ==========
    # sample s, t, u like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # sample u uniformly between s and t
    h = torch.empty_like(delta_t).uniform_(0, 1)
    u = s + h * (t - s)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute full jump s -> t (student)
    _, f_xst = flow_map.get_map_and_velocity(s, t, Is, obs_emb)

    # compute two-step jump s -> u -> t (teacher, no stopgrad)
    xsu, f_xsu = flow_map.get_map_and_velocity(s, u, Is, obs_emb)
    _, f_xut = flow_map.get_map_and_velocity(u, t, xsu, obs_emb)

    # uniform PSD: teacher = (1 - h) * phi_su + h * phi_ut
    # where h is the relative position of u between s and t
    student = f_xst
    # expand h to match f_xsu dimensions: [batch, horizon, act_dim]
    h_expanded = h.view(-1, 1, 1)
    teacher = (1 - h_expanded) * f_xsu + h_expanded * f_xut

    # compute PSD loss using get_norm (ignore weight_st as requested)
    psd_term = get_norm(student - teacher, config.norm_type)
    psd_term = config.loss_scale * torch.mean(psd_term)

    # combine losses
    total_loss = flow_matching_loss + psd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "psd_term": psd_term.item(),
    }


def lsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian self-distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. LSD term that encourages consistency in the velocity field

    The LSD term uses uniform sampling between s and t without stopgrad.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== LSD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and dt_Xst using jvp_t
    xst, dt_xst = flow_map.jvp_t(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (no stopgrad)
    b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # lsd loss (ignore weight_st)
    error = b_eval - dt_xst
    lsd_term = get_norm(error, config.norm_type)
    lsd_term = config.loss_scale * torch.mean(lsd_term)

    # combine losses
    total_loss = flow_matching_loss + lsd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "lsd_term": lsd_term.item(),
    }


def esd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Euler self-distillation loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== ESD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # compute jvp
    _, grad_xst_b = flow_map.jvp_x(s, t, Is, b_eval, obs_emb)

    # esd loss
    error = ds_xst + grad_xst_b
    esd_term = get_norm(error, config.norm_type)
    esd_term = config.loss_scale * torch.mean(esd_term)

    # combine losses
    total_loss = flow_matching_loss + esd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "esd_term": esd_term.item(),
    }


def mf_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Mean flow loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== Mean flow term ==========
    # sample s, t
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)
    dot_Is = interp.calc_It_dot(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        # Difference 1: use dot_Is instead of b_eval
        # Difference 2: also disable gradient for jvp_x
        # compute jvp
        _, grad_xst_b = flow_map.jvp_x(s, t, Is, dot_Is, obs_emb)

    # mf loss
    error = ds_xst + grad_xst_b
    mf_term = get_norm(error, config.norm_type)
    mf_term = config.loss_scale * torch.mean(mf_term)

    # combine losses
    total_loss = flow_matching_loss + mf_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "mf_term": mf_term.item(),
    }


def bridge_loss_old(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Generalized Bridge Policy Loss (Sliding Window Flow Matching).

    Learns to predict the velocity field that transforms the current window (X0)
    to the future window (X1) shifted by 'prediction_offset'.

    Uses Pinned Brownian Bridge to construct intermediate noisy states.
    """
    # 1. Parse Parameters
    # K: Prediction Offset (The sliding step)
    K = getattr(config, "prediction_offset", 16)
    # W: Dataset total horizon
    W = act.shape[1]
    # H: Network Horizon (Window Size)
    H = W - K

    # 2. Slice Start Window (X0) and Target Window (X1)
    # X0: [t : t+H]
    x0 = act[:, :H, :]
    # X1: [t+K : t+H+K]
    x1 = act[:, K:, :]

    # 3. Sample Time and Noise (Pinned Brownian Bridge)
    bs = act.shape[0]
    device = act.device

    #! === [关键修改] Mix Training / Flat Start Augmentation ===
    #! 以 10% 概率，将 x0 替换为 "Repeat(Start_Point)"
    #! 模拟推理时的 Cold Start 输入分布
    mask = torch.rand((bs,), device=device) < 0.1

    # * 构造 Flat x0
    # * 取 x0 的第一个点 (B, 1, D) -> 扩展成 (B, H, D)
    x0_flat = x0[:, 0:1, :].expand(-1, H, -1)

    # * 使用 mask 进行替换
    # * mask 为 True 的地方用 flat，否则用原始 trajectory
    x0_aug = torch.where(mask.view(-1, 1, 1), x0_flat, x0)
    x0 = x0_aug

    # * 注意：Target 依然是 x1 - x0_aug
    # * 也就是说，如果输入是平板，网络要预测 "从平板变到真实轨迹" 的巨大位移
    # * 如果输入是轨迹，网络要预测 "从轨迹变到下一帧轨迹" 的微小位移

    # === 剩下的流程不变 ===

    # Sample t uniformly from [0, 1]
    t = torch.rand((bs,), device=device)
    # Sample noise
    eps = torch.randn_like(x0)

    # Construct intermediate state x_tau
    # PBB formula: x_tau = (1 - t) * x0 + t * x1 + sigma * eps
    # sigma = sqrt(t * (1 - t))

    # Broadcast t for calculation: (B, 1, 1)
    t_exp = t.view(-1, 1, 1)
    sigma = torch.sqrt(t_exp * (1 - t_exp))

    ## 多加一个控制噪声强度的超参数 lambda
    lambda_noise = 0.1

    x_train = (1 - t_exp) * x0 + t_exp * x1 + lambda_noise * eps

    # 4. Network Forward
    # Get observation embedding
    obs_emb = encoder(obs, None)

    # Predict velocity field v
    # Input is the full window (B, H, D)
    v_pred = flow_map.get_velocity(t, x_train, obs_emb)

    # 5. Compute Target and Loss
    # Target velocity is the straight displacement vector (plus noise correction implied by PBB)
    # v_target = x1 - x0
    v_target = x1 - x0

    # Compute MSE Loss over the entire window
    loss = get_norm(v_pred - v_target, config.norm_type) ** 2
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def bridge_loss_old2(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Generalized Bridge Policy Loss (Sliding Window Flow Matching).

    Learns to predict the velocity field that transforms the current window (X0)
    to the future window (X1) shifted by 'prediction_offset'.

    Uses Pinned Brownian Bridge to construct intermediate noisy states.
    """
    # 1. Parse Parameters
    # K: Prediction Offset (The sliding step)
    K = getattr(config, "prediction_offset", 16)
    # W: Dataset total horizon
    W = act.shape[1]
    # H: Network Horizon (Window Size)
    H = W - K

    # 2. Slice Start Window (X0) and Target Window (X1)
    # X0: [t : t+H]
    x0 = act[:, :H, :]
    # X1: [t+K : t+H+K]
    x1 = act[:, K:, :]

    # 3. Sample Time and Noise (Pinned Brownian Bridge)
    bs = act.shape[0]
    device = act.device

    # ---------------- [修改开始] Mix Training 逻辑 ----------------

    # (1) 构造 Flat Anchor (X_flat)
    # 取 x0 的第一个点复制
    x_anchor = x0[:, 0:1, :].expand(-1, H, -1)

    # (2) 采样 Alpha (混合比例)
    # 推荐使用 Beta 分布，偏向于 1 (因为推理时 alpha 较大，如 0.8)
    # 或者简单起见先用 Uniform [0, 1]
    # shape: (B, 1, 1) 用于广播
    alpha = torch.rand((bs, 1, 1), device=device)
    # 若想偏向 1: alpha = torch.distributions.Beta(5, 2).sample((bs, 1, 1)).to(device)

    # (3) 构造 Noisy GT (X_GT + Noise)
    # 模拟上一帧预测的误差
    noise_level = 0.1  # 超参数，可调
    x0_noisy = x0 + noise_level * torch.randn_like(x0)

    # (4) 构造混合输入 X_mix
    # 这是喂给网络的 x (在 Flow Matching 语境下通常叫 x_t 或 x_source)
    x_mix = alpha * x0_noisy + (1 - alpha) * x_anchor

    # ---------------- [修改结束] ----------------

    # ... (t 的采样和 PBB 加噪逻辑) ...

    # [注意] 这里 PBB 的构造也需要调整！
    # 原来的 PBB: x_train = (1-t) * x0 + t * x1 + ...
    # 现在的 PBB 起点应该是 x_mix！
    # 因为我们要学的是从 x_mix 到 x1 的流

    # Sample t
    t = torch.rand((bs,), device=device)
    t_exp = t.view(-1, 1, 1)

    # PBB Noise
    eps = torch.randn_like(x0)
    sigma = torch.sqrt(t_exp * (1 - t_exp))
    lambda_noise = 0  # Pinned Brownian Bridge 的噪声强度，暂时设成 0

    # 构造 Flow Matching 的训练数据 x_train
    # 时刻 0 是 x_mix，时刻 1 是 x1
    x_train = (1 - t_exp) * x_mix + t_exp * x1 + lambda_noise * sigma * eps

    # 4. Network Forward
    obs_emb = encoder(obs, None)
    v_pred = flow_map.get_velocity(t, x_train, obs_emb)

    # 5. Target
    # [关键修改] Target 必须指向 x1，且起点是当前的 x_train (近似 x_mix)
    # Flow Matching 的 Target 也就是 Vector Field 的方向
    # V = X_target - X_source
    v_target = x1 - x_mix

    # Loss
    loss = get_norm(v_pred - v_target, config.norm_type) ** 2
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def bridge_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    C_start: torch.Tensor,   # 新接口: 起点 Chunk
    C_end: torch.Tensor,     # 新接口: 终点 Chunk
    h_curr: torch.Tensor,    # 新接口: 对应 tau 时刻的 Observation
    tau: torch.Tensor,       # 新接口: 归一化时间进度 (B,)
) -> float:
    """New Bridge Policy Loss (BP.md Design).

    Matches the vector field from C_start to C_end, conditioned on streaming observation h_curr.
    """
    bs = C_start.shape[0]
    device = C_start.device

    # 1. Construct Bridge State X_tau (with Noise)
    # Formula: X_tau = (1 - tau) * C_start + tau * C_end + sigma(tau) * epsilon
    epsilon = torch.randn_like(C_start)

    # Broadcast tau: (B,) -> (B, 1, 1)
    tau_exp = tau.view(-1, 1, 1)

    # Noise Schedule: sigma(tau) = tau * (1 - tau) (Simple convex schedule)
    # Or standard Brownian Bridge sigma = sqrt(t(1-t)).
    # BP.md mentions "sigma(tau) usually tau(1-tau)" in table, but code often uses something simple.
    # Let's use sqrt(tau(1-tau)) to be consistent with standard diffusion/bridge literature unless specified.
    # User's MD table says "usually tau(1-tau)", let's stick to that for simplicity as requested.
    sigma_tau = torch.sqrt(tau_exp * (1 - tau_exp)) # Using sqrt for variance consistency

    sigma_alpha = 0.5

    # Interpolation
    X_tau = (1 - tau_exp) * C_start + tau_exp * C_end + sigma_tau * epsilon

    # 2. Get Observation Embedding
    # h_curr is already the correct observation at t_curr
    obs_emb = encoder(h_curr, None)

    # 3. Network Prediction
    # v_theta(X_tau, tau, h_curr)
    # Note: flow_map.get_velocity expects t as (B,) or (B,1,1)
    v_pred = flow_map.get_velocity(tau, X_tau, obs_emb)

    # 4. Compute Optimal Transport Target
    # v_target = C_end - C_start
    v_target = C_end - C_start

    # 5. Loss
    loss = get_norm(v_pred - v_target, config.norm_type) ** 2
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def bridge_v2_loss_old(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Bridge Policy V2 Loss (Recurrent Refinement).

    Network learns: F(X_t + noise, obs) -> X_{t+1}
    Input 'act' contains the full horizon sequence of length H+1.
    We slice it to get X_t (current) and X_{t+1} (next).
    """
    # 1. Slice Data
    # act: (B, H+1, D)
    # X_curr: act[:, :-1, :] -> Length H
    # X_next: act[:, 1:, :]  -> Length H
    X_curr = act[:, :-1, :]
    X_next = act[:, 1:, :]

    bs = X_curr.shape[0]
    device = X_curr.device

    # # 2. Add Training Noise (sigma)
    # # sigma_max default 0.2, or read from config
    # sigma_max = getattr(config, "sigma_max", 0.2)
    # # Sample sigma per batch item: (B, 1, 1)
    # sigma = torch.rand((bs, 1, 1), device=device) * sigma_max

    # epsilon = torch.randn_like(X_curr)
    # X_in = X_curr + sigma * epsilon

    # 2. Log-Uniform Noise Sampling (Correct)
    sigma_min = 1e-2
    sigma_max = 0.5
    # Log-Uniform sampling for sigma
    log_sigma = torch.rand((bs, 1, 1), device=device) * np.log(sigma_max / sigma_min) + np.log(sigma_min)
    sigma = torch.exp(log_sigma)

    # Structured Noise Injection (Correct)
    eps_white = torch.randn_like(X_curr)
    eps_bias = torch.randn(bs, 1, X_curr.shape[-1], device=device) # Broadcast over H
    epsilon = 0.2 * eps_white + 0.8 * eps_bias

    # Input Construction
    X_in = X_curr + sigma * epsilon

    # 3. CFG Dropout (Obs Masking)
    # p_uncond default 0.1
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

    # Encode Observation
    # obs is already correct corresponding to X_curr start time
    obs_emb = encoder(obs, None)

    # Apply Mask: if mask=0, zero out obs_emb
    # obs_emb: (B, D_emb) or (B, N, D_emb)
    # Expand mask to match obs_emb dimensions
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    # 4. Input "Time" as Condition Token (The Hack)
    # If mask=1 (Visual), t=1.0
    # If mask=0 (Blind), t=0.0
    t_input = mask # (B,) in {0, 1}

    # 5. Network Prediction
    # Output is X_{t+1}
    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)

    # 6. Loss
    # Target is clean X_next
    loss = get_norm(X_pred - X_next, config.norm_type) ** 2
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def bridge_v2_loss_old2(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Bridge Policy V2 Loss (Residual Prediction + Log-Uniform Noise)."""

    # 1. Slice Data (Correct)
    X_curr = act[:, :-1, :]
    X_next = act[:, 1:, :]
    bs = X_curr.shape[0]
    device = X_curr.device

    # 2. Log-Uniform Noise Sampling (Correct)
    # sigma_min = 1e-4
    # sigma_max = 0.5
    # # Log-Uniform sampling for sigma
    # log_sigma = torch.rand((bs, 1, 1), device=device) * np.log(sigma_max / sigma_min) + np.log(sigma_min)
    # sigma = torch.exp(log_sigma)

    # New: Log-Normal (集中火力攻坚中间区域)
    # P_mean = np.log(0.1) # 约 -2.3
    # P_std = 1.0
    log_sigma = torch.randn((bs, 1, 1), device=device) * 1.0 + np.log(0.1)
    sigma = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)


    # Structured Noise Injection (Correct)
    eps_white = torch.randn_like(X_curr)
    eps_bias = torch.randn(bs, 1, X_curr.shape[-1], device=device) # Broadcast over H
    epsilon = 0.8 * eps_white + 0.2 * eps_bias

    # Input Construction
    X_in = X_curr + sigma * epsilon

    # 3. CFG Dropout (Standard Trick)
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    # Time Embedding Hack: t=1 (Visual), t=0 (Blind)
    t_input = mask

    # 4. Network Prediction (PREDICT RESIDUAL!)
    # Network Output: X_residual (Expected to be X_in - X_next)
    X_pred_residual = flow_map.get_velocity(t_input, X_in, obs_cond)

    # 5. Target Calculation (Residual)
    # We want X_next = X_in - X_pred_residual
    # So Target = X_in - X_next
    target_residual = X_in - X_next

    # 6. Loss Calculation (With Implicit Weighting)
    # Use MSE directly on residual
    # This automatically weights large noise samples more (if unscaled),
    # OR use sigma scaling to balance.

    # Karras / EDM Style: Weight by 1/sigma^2 ??
    # Let's keep it simple first: Standard MSE on Residual is robust.
    # It means: Minimize error in physical space.
    # loss = F.mse_loss(X_pred_residual, target_residual, reduction='none')
    loss = get_norm(X_pred_residual - target_residual, config.norm_type) ** 2

    # Optional: Weighting to focus on small noise (Precision)
    # weight = 1 / (sigma + 0.1)
    # loss = loss * weight

    loss = config.loss_scale * torch.mean(loss)

    return loss, {}



def bridge_v2_loss_0221(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Bridge Policy V2 Loss (HAN + EDM Preconditioning + CFG)."""

    # 1. Slice Data
    # X_curr: (B, H, D)
    # X_next: (B, H, D)
    X_curr = act[:, :-1, :]
    X_next = act[:, 1:, :]

    bs, H, D = X_curr.shape
    device = X_curr.device

    # ==========================
    # 2. Noise & Sigma Schedule
    # ==========================

    # A. Base Sigma (Log-Normal, EDM Style)
    # P_mean=-1.2, P_std=1.2 (Covers 0.001 to 0.5 effectively)
    # Shape: (B, 1, 1)
    rnd_normal = torch.randn((bs, 1, 1), device=device)
    sigma_base = (rnd_normal * 1.2 - 1.2).exp()

    # B. Horizon-Adaptive Profile (HAN)
    # sigma_k ~ sqrt(k) from 0.1 to 1.0
    # Shape: (1, H, 1)
    time_scale = torch.sqrt(torch.linspace(0.01, 1.0, H, device=device)).view(1, H, 1)

    # C. Final Sigma Map
    # Shape: (B, H, 1) - Per-element noise level
    sigma = sigma_base * time_scale

    # D. Correlated Noise (OU Process)
    # rho=0.9 is a good default
    noise = generate_ou_noise(bs, H, D, device, rho=0.9)

    # E. Construct Noisy Input
    X_in = X_curr + sigma * noise

    # ==========================
    # 3. CFG / Condition
    # ==========================

    # p_uncond default 0.1
    p_uncond = getattr(config, "p_uncond", 0.1)
    # Mask: 1 for Keep Obs, 0 for Drop Obs
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

    # Encode Observation
    obs_emb = encoder(obs, None)

    # Apply Mask
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    # Time Embedding Hack (for CFG Mode)
    # t=1 (Visual), t=0 (Blind)
    t_cfg = mask # (B,)

    # ==========================
    # 4. EDM Preconditioning
    # ==========================

    # Calculate Scaling Coefficients
    # c_in: Input scaling (to unit variance)
    c_in = 1 / (sigma ** 2 + 1).sqrt()

    # c_skip: Skip connection weight
    c_skip = 1 / (sigma ** 2 + 1)

    # c_out: Network output weight
    c_out = sigma / (sigma ** 2 + 1).sqrt()

    # c_noise: Noise level embedding (replacing simple t)
    # Note: We pass c_noise to the network if it accepts continuous t.
    # BUT since we use t for CFG (0/1), we need to inject c_noise differently?
    # GMD says: Use AdaLN for c_noise AND t_cfg?
    # For simplicity/compatibility with UNet(x, t, cond):
    # We stick to t_cfg for the `timestep` input to enable CFG.
    # The network will be "Blind" to the exact sigma level (Blind Denoising).
    # This is fine for BPv2.

    # ==========================
    # 5. Network Forward
    # ==========================

    # Input: Scaled X_in
    # Timestep: t_cfg (0 or 1)
    # Condition: obs_cond
    # Output: F_out (Normalized Signal)
    F_out = flow_map.get_velocity(t_cfg, X_in * c_in, obs_cond)

    # ==========================
    # 6. Denoised Prediction
    # ==========================

    # Reconstruct X_pred (in physical space)
    # X_pred = c_skip * X_in + c_out * F_out
    # This effectively predicts X_next (Clean Data)
    X_pred = c_skip * X_in + c_out * F_out

    # ==========================
    # 7. Weighted Loss
    # ==========================

    # EDM Weighting
    # weight = (sigma^2 + 1) / sigma^2
    # This balances the gradient magnitude across all noise levels
    loss_weight = (sigma ** 2 + 1) / (sigma ** 2)

    # Weighted MSE
    # reduction='none' to apply per-element weighting
    loss_raw = (X_pred - X_next) ** 2
    loss = (loss_raw * loss_weight).mean()

    loss = config.loss_scale * loss

    return loss, {}



def bridge_v2_loss_0226(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Bridge Policy V2 Loss (Horizon-Adaptive Noise + Weighted MSE).

    Network learns: F(X_t + adaptive_noise, obs) -> X_{t+1}
    Input 'act' contains the full horizon sequence of length H+1.
    """
    # 1. Slice Data
    X_curr = act[:, :-1, :] # (B, H, D)
    X_next = act[:, 1:, :]  # (B, H, D)

    bs, H, D = X_curr.shape
    device = X_curr.device

    # # 2. Horizon-Adaptive Noise (HAN) Construction
    # # A. Base Sigma (Log-Uniform) - Global noise level
    # # Range: [1e-4, 0.2] (Base scale doesn't need to be huge, profile will scale it)
    # sigma_min = 1e-4
    # sigma_max = 0.2
    # log_sigma = torch.rand((bs, 1, 1), device=device) * np.log(sigma_max / sigma_min) + np.log(sigma_min)
    # sigma_base = torch.exp(log_sigma) # (B, 1, 1)

    # # B. Temporal Profile (Linear or Quadratic)
    # # [0.1, ..., 1.0] -> Increasing noise along horizon
    # # Use broadcasting: (1, H, 1)
    # steps = torch.linspace(0, 1, H, device=device).view(1, H, 1)
    # scale_vec = 0.1 + 0.9 * steps  # Linear profile from 0.1x to 1.0x
    # # Optional: Quadratic for sharper focus? scale_vec = 0.1 + 0.9 * (steps ** 2)

    # # C. Final Sigma Map
    # # sigma: (B, H, 1) - Noise level for each step
    # sigma = sigma_base * scale_vec

    # # D. Structured Noise Injection
    # # 80% White, 20% Bias (Low freq)
    # eps_white = torch.randn_like(X_curr)
    # eps_bias = torch.randn(bs, 1, D, device=device) # Broadcast over H
    # epsilon = 0.8 * eps_white + 0.2 * eps_bias

    # # Input Construction
    # X_in = X_curr + sigma * epsilon



    ##!! 新的加噪方案 —— OU-like Correlated Noise
    # # 1. Base Sigma (Log-Uniform)
    # # Range: [1e-4, 0.2] (Base scale doesn't need to be huge, profile will scale it)
    # sigma_min = 1e-2
    # sigma_max = 0.1
    # log_sigma = torch.rand((bs, 1, 1), device=device) * np.log(sigma_max / sigma_min) + np.log(sigma_min)
    # sigma_base = torch.exp(log_sigma) # (B, 1, 1)

    # #?? New: Log-Normal (集中火力攻坚中间区域)
    # # P_mean = np.log(0.1) # 约 -2.3
    # # P_std = 1.0
    # log_sigma = torch.randn((bs, 1, 1), device=device) * 1.0 + np.log(0.1)
    # sigma_base = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)

    #** New_v2: Log Normal + 随着训练进度动态变化 log_mean
    current_step = int(delta_t[0].item())   # 注意：delta_t 是一个 (B,) 的 Tensor，取第一个元素即可
    total_steps = config.gradient_steps
    # 1. Compute Progress (0.0 -> 1.0)
    # 假设我们知道 total_steps (e.g. 100k)
    # 如果不知道，可以用 n_gradient_step 做一个衰减函数
    # e.g. sigma_center = 0.1 * (0.99995 ^ step)
    progress = min(1.0, current_step / total_steps)

    # 2. Anneal Center (Linear Decay)
    # Start: 0.1 (Robustness)
    # End: 0.001 (Precision)
    sigma_center = 0.1 * (1 - progress) + 0.001 * progress

    # 3. Sample Log-Normal
    # P_std=1.0 保持不变，保证在任何时候都有一定的探索宽度
    log_mean = np.log(sigma_center)
    log_sigma = torch.randn((bs, 1, 1), device=device) * 1.0 + log_mean
    sigma_base = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)

    # k_ratio = torch.linspace(0, 1, H, device=device).view(1, H, 1)
    # log_mean = np.log(sigma_center)
    # log_mean_k = log_mean * (1 + k_ratio)  # 让 k=0 保持原值，k=H-1 翻倍
    # log_sigma = torch.randn((bs, H, 1), device=device) * 1.0 + log_mean_k
    # sigma_base = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)

    # #!!! --- Mode B: Reset (Recovery) ---
    # # Mode Mask: 10% Reset, 90% Tracking
    # is_reset = torch.rand((bs, 1, 1), device=device) < 0.1
    # # Fixed Large Sigma
    # sigma_reset = torch.full((bs, 1, 1), 2.0, device=device)

    # # Combine Sigma
    # sigma_base = torch.where(is_reset, sigma_reset, sigma_base)


    # 2. Temporal Scaling (Brownian Diffusion)
    # sigma_k ~ sqrt(k)
    time_scale = torch.sqrt(torch.linspace(0.01, 1.0, H, device=device)).view(1, H, 1)


    sigma_floor = 1e-3
    sigma = sigma_base * time_scale
    sigma = torch.maximum(sigma_base * time_scale, torch.tensor(sigma_floor, device=device))

    # sigma = torch.where(is_reset, sigma, torch.maximum(sigma, torch.tensor(sigma_floor, device=device)))
    # sigma = sigma_base * time_scale

    # 3. Correlated Noise (OU Process)
    # Replaces the "0.2 white + 0.8 bias" trick
    # rho=0.9 is a good default for smooth trajectories
    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)

    # White Noise (Uncorrelated)
    noise_white = torch.randn_like(X_curr)

    # For Reset: Use White Noise (Chaos). For Tracking: Use OU (Drift).
    # This matches the physical intuition.
    # noise = torch.where(is_reset, noise_white, noise_ou)

    # === Shift Operation ===
    # 构造 Shifted Input
    # 前 H-1 位: 直接取 X_curr 的后 H-1 位
    X_shifted = X_curr[:, 1:, :]
    # 最后 1 位: 重复最后一位，或者补 0，或者补噪声
    # 推荐: 重复最后一位 (Stationary Prior)
    X_tail = X_curr[:, -1:, :]
    X_in_base = torch.cat([X_shifted, X_tail], dim=1)

    # 4. Final Input
    X_in = X_in_base + sigma * noise_ou

    # # === The Magic Fix: Random Restart Injection ===
    # # 以 10% 概率，把输入替换为“彻底的垃圾” (Reset)
    # # 这强迫网络不仅会修，还会画

    # # 生成 Mask: 10% 为 True
    # reset_mask = torch.rand((bs, 1, 1), device=device) < 0.1

    # # 构造垃圾输入 (有两种选择)
    # # A. 纯高斯噪声 (像 DP 那样)
    # # X_garbage = torch.randn_like(X_curr)

    # # B. Stationary Padding (像推理时的 Cold Start)
    # # 取当前 Pose (act 的第一帧)，重复 H 次
    # # act: (B, H+1, D) -> act[:, 0, :] -> (B, D)
    # current_pose = act[:, 0, :].unsqueeze(1).expand(-1, H, -1)
    # # 给它加点小噪声，防止过拟合数值
    # X_garbage = current_pose

    # # 替换
    # X_in = torch.where(reset_mask, X_garbage, X_in)

    # 3. CFG Dropout (Standard Trick)
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

    obs_emb = encoder(obs, None)
    # Expand mask to match obs_emb dimensions
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    # Time Embedding Hack: t=1 (Visual), t=0 (Blind)
    t_input = mask # (B,)

    # 4. Network Prediction (Predict Clean X_next)
    # Output is X_{t+1}
    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)

    # 5. Weighted Loss Calculation
    # Raw MSE per element
    loss_raw = (X_pred - X_next) ** 2

    # Weighting: Inverse Variance (Focus on low noise areas)
    # weight = 1 / (sigma^2 + epsilon)
    # This makes the loss contribution roughly equal across all noise levels
    weight = 1.0 / (sigma ** 2 + 1e-4)

    # Weighted Mean
    loss = (loss_raw * weight).mean()

    # # --- Scheme A: Smoothness Consistency ---
    # # 惩罚预测轨迹的二阶差分 (加速度)
    # # X_pred: (B, H, D)
    # # Acc = X[t+2] - 2*X[t+1] + X[t]
    # # 这迫使轨迹变平滑，减少推理时的抖动

    # # Slice
    # p_t = X_pred[:, :-2, :]
    # p_tp1 = X_pred[:, 1:-1, :]
    # p_tp2 = X_pred[:, 2:, :]

    # acc = p_tp2 - 2 * p_tp1 + p_t
    # loss_smooth = (acc ** 2).mean()

    # # Coefficient: need tuning (e.g. 0.1)
    # loss = loss + 0.1 * loss_smooth
    # # ----------------------------------------

    # # --- Scheme B: Velocity Weighting (The New Magic) ---
    # # 计算 GT 速度: ||X_next - X_curr|| (平均速度)
    # # shape: (B,)
    # velocity = torch.norm(X_next - X_curr, dim=-1).mean(dim=-1)

    # # 定义权重函数: W = 1 + alpha * exp(-beta * v)
    # # alpha=4.0 (最大加权到 5倍)
    # # beta=10.0 (速度 0.1 时衰减到 1+4*0.36 ~= 2.5倍)
    # # 这里的参数需要根据 PushT 的速度范围微调。
    # # PushT 速度通常在 0.0 ~ 1.0 之间。
    # w_vel = 1.0 + 4.0 * torch.exp(-10.0 * velocity)

    # # 扩展维度以匹配 loss_raw: (B,) -> (B, 1, 1)
    # w_vel = w_vel.view(bs, 1, 1)
    # # ----------------------------------------------------

    # # Existing Sigma Weighting: Inverse Variance
    # w_sigma = 1.0 / (sigma ** 2 + 1e-4)

    # # Combined Weight
    # final_weight = w_sigma * w_vel

    # # Weighted Mean
    # loss = (loss_raw * final_weight).mean()


    # Scale by config
    loss = config.loss_scale * loss

    return loss, {}


# def bridge_v2_loss(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ) -> float:
#     """Bridge Policy V2 Loss (Horizon-Adaptive Noise + Weighted MSE).

#     Network learns: F(X_t + adaptive_noise, obs) -> X_{t+1}
#     Input 'act' contains the full horizon sequence of length H+1.
#     """
#     # 1. Slice Data
#     X_curr = act[:, :-1, :] # (B, H, D)
#     X_next = act[:, 1:, :]  # (B, H, D)

#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     # 2. Horizon-Adaptive Noise (HAN) Construction
#     # #?? New: Log-Normal (集中火力攻坚中间区域)
#     # # P_mean = np.log(0.1) # 约 -2.3
#     # # P_std = 1.0
#     # log_sigma = torch.randn((bs, 1, 1), device=device) * 1.0 + np.log(0.1)
#     # sigma_base = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)



#     #** New_v2: Log Normal + 随着训练进度动态变化 log_mean
#     current_step = int(delta_t[0].item())   # 注意：delta_t 是一个 (B,) 的 Tensor，取第一个元素即可
#     total_steps = config.gradient_steps
#     # 2.1. Compute Progress (0.0 -> 1.0)
#     # 假设我们知道 total_steps (e.g. 100k)
#     # 如果不知道，可以用 n_gradient_step 做一个衰减函数
#     # e.g. sigma_center = 0.1 * (0.99995 ^ step)
#     progress = min(1.0, current_step / total_steps)

#     # 2.2. Anneal Center (Linear Decay)
#     # Start: 0.1 (Robustness)
#     # End: 0.001 (Precision)
#     sigma_center = 0.1 * (1 - progress) + 0.001 * progress

#     # 2.3. Sample Log-Normal
#     # P_std=1.0 保持不变，保证在任何时候都有一定的探索宽度
#     log_mean = np.log(sigma_center)
#     log_sigma = torch.randn((bs, 1, 1), device=device) * 1.0 + log_mean
#     sigma_base = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)

#     # k_ratio = torch.linspace(0, 1, H, device=device).view(1, H, 1)
#     # log_mean = np.log(sigma_center)
#     # log_mean_k = log_mean * (1 + k_ratio)  # 让 k=0 保持原值，k=H-1 翻倍
#     # log_sigma = torch.randn((bs, H, 1), device=device) * 1.0 + log_mean_k
#     # sigma_base = torch.exp(log_sigma).clamp(min=1e-4, max=0.5)

#     # 2. Temporal Scaling (Brownian Diffusion)
#     # sigma_k ~ sqrt(k)
#     time_scale = torch.sqrt(torch.linspace(0.001, 1.0, H, device=device)).view(1, H, 1)

#     sigma_floor = 1e-3
#     sigma = sigma_base * time_scale
#     sigma = torch.maximum(sigma_base * time_scale, torch.tensor(sigma_floor, device=device))

#     # sigma = torch.where(is_reset, sigma, torch.maximum(sigma, torch.tensor(sigma_floor, device=device)))
#     # sigma = sigma_base * time_scale

#     # ##!! 简便方式： $\sigma_{base} \sim U(0, 1)$，$sigma_k=\sqrt{k+\frac{\sigma_{\text {base }}}{H}}$
#     # sigma_base = torch.rand((bs, 1, 1), device=device)
#     # k = torch.linspace(0.001, 1.0, H, device=device).view(1, H, 1)    # 构建时间索引 k = [0, 1, ..., H-1]
#     # sigma = torch.sqrt(k + sigma_base / H) # (k + sigma_base) / H
#     # sigma = torch.clamp(sigma, min=1e-4)    # 最小值保护 (防止 sigma 过小导致除以零或数值不稳定)

#     ##*** 新简便方式的实现：
#     scale_max = 0.5
#     scale_random = torch.rand(1, device=device)  # U[0, 1]
#     scale = 0.01 + scale_random * (0.5 - 0.01)
#     jitter = torch.rand((bs, 1, 1), device=device) / H
#     time_scale = torch.sqrt(torch.linspace(0, 1.0, H, device=device).view(1, H, 1) + jitter)
#     sigma = scale * time_scale

#     # 3. Correlated Noise (OU Process)
#     # Replaces the "0.2 white + 0.8 bias" trick
#     # rho=0.9 is a good default for smooth trajectories
#     noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)

#     # White Noise (Uncorrelated)
#     noise_white = torch.randn_like(X_curr)

#     # 4. Final Input
#     X_in = X_curr + sigma * noise_ou

#     # 3. CFG Dropout (Standard Trick)
#     p_uncond = getattr(config, "p_uncond", 0.1)
#     mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

#     obs_emb = encoder(obs, None)
#     # Expand mask to match obs_emb dimensions
#     mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
#     obs_cond = obs_emb * mask_expanded

#     # Time Embedding Hack: t=1 (Visual), t=0 (Blind)
#     t_input = mask # (B,)

#     # 4. Network Prediction (Predict Clean X_next)
#     # Output is X_{t+1}
#     X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)

#     # 5. Weighted Loss Calculation
#     # Raw MSE per element
#     loss_raw = (X_pred - X_next) ** 2

#     # Weighting: Inverse Variance (Focus on low noise areas)
#     # weight = 1 / (sigma^2 + epsilon)
#     # This makes the loss contribution roughly equal across all noise levels
#     weight = 1.0 / (sigma ** 2 + 1e-4)

#     # Weighted Mean
#     loss = (loss_raw * weight).mean()

#     # loss = loss_raw.mean()

#     # Scale by config
#     loss = config.loss_scale * loss

#     return loss, {}


def bridge_v2_loss0227(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Bridge Policy V2 Loss (Geometric Noise Schedule)."""

    # 1. Slice Data
    X_curr = act[:, :-1, :] # (B, H, D)
    X_next = act[:, 1:, :]  # (B, H, D)

    bs, H, D = X_curr.shape
    device = X_curr.device

    # ==========================
    # 2. Geometric Noise Schedule
    # ==========================

    # A. Define Boundary (Fixed, No Annealing needed)
    # sigma_min: Execution precision (2mm)
    # sigma_max: Max lookahead uncertainty (20cm)
    sigma_min = 0.002
    sigma_max = 0.2

    # B. Geometric Profile (Log-Linear Interpolation)
    # k = 0 -> sigma_min
    # k = H-1 -> sigma_max
    # Formula: log_sigma_k = (1-t)*log_min + t*log_max
    t_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    log_profile = (1 - t_k) * log_min + t_k * log_max # (1, H, 1)

    # C. Random Scale Shift (Log-Normal P_std=1.0)
    # Allows network to see "Easy Mode" (Overall small noise) and "Hard Mode" (Overall large noise)
    # while maintaining the geometric relationship.
    # Center = 0 (No shift on average)
    log_shift = torch.randn((bs, 1, 1), device=device) * 1.0

    # D. Final Sigma Map
    # sigma = exp(profile + shift)
    # Clamp to avoid numerical issues (e.g. shift could be very large/small)
    sigma = torch.exp(log_profile + log_shift).clamp(min=1e-5, max=1.0)

    # ==========================
    # 3. Noise Injection
    # ==========================

    # OU Noise (Structure)
    # Helps with spatial correlation
    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)

    # Construct Input
    X_in = X_curr + sigma * noise_ou

    # ==========================
    # 4. CFG / Condition
    # ==========================
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded
    t_input = mask # (B,)

    # ==========================
    # 5. Network Prediction
    # ==========================
    # Predict Clean X_next
    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)

    # ==========================
    # 6. Weighted Loss
    # ==========================
    loss_raw = (X_pred - X_next) ** 2

    # Weighting: Inverse Variance
    # Standard Score Matching Weighting
    weight = 1.0 / (sigma ** 2 + 1e-4)

    # Optional: Clamp weight to prevent explosion at small sigma
    # 1e-4 -> weight=10000. This is huge.
    # Let's soft-clamp it to 2000.0 (Reasonable gradient scale)
    # weight = torch.clamp(weight, max=2000.0)

    loss = (loss_raw * weight).mean()
    loss = config.loss_scale * loss

    return loss, {}

def bridge_v2_loss1(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Bridge Policy V2 Loss with RDB-PT (Residual Diffusion Bridge + Positional Tempering).

    Network learns: F(X_t + adaptive_noise, obs) -> X_{t+1}
    Input 'act' contains the full horizon sequence of length H+1.

    New Noise Design (3 Core Parameters):
    - S (Scale): Global noise strength [0.005, 0.2]
    - sigma_0 (Precision): Base noise at position 0
    - alpha (Refinement): Tempering exponent
    """
    # ========== 1. Slice Data ==========
    X_curr = act[:, :-1, :]  # (B, H, D)
    X_next = act[:, 1:, :]   # (B, H, D)

    bs, H, D = X_curr.shape
    device = X_curr.device

    # ========== 2. RDB-PT Noise Construction ==========
    # 2.1 Sample Global Scale (Truncated Power-Law = Log-Uniform)
    # This ensures equal coverage across all noise magnitudes
    sigma_min = 0.005
    sigma_max = 0.2
    gamma = 0.5  # Power-law exponent (0.5 = balanced coverage)

    U = torch.rand((bs, 1, 1), device=device)
    S = sigma_min * (sigma_max / sigma_min) ** (U ** (1 / (1 - gamma)))

    # 2.2 Positional Profile (Langevin Growth + Tempering)
    # Core Parameters (Tunable)
    sigma_0 = 0.01   # Precision at first position (CRITICAL!)
    alpha = 0.75     # Tempering exponent (controls refinement strength)

    # Derived: tau = H/2 (characteristic time constant)
    tau = H / 2.0

    k = torch.arange(H, device=device).float().view(1, H, 1)

    # Component 1: Langevin-style error growth
    # sqrt(1 + (k/tau)^2) grows slower than sqrt(k), better for feedback systems
    sigma_growth = torch.sqrt(1 + (k / tau) ** 2)

    # Component 2: Positional Tempering
    # Models "how many refinement chances left"
    beta_k = 1 - k / (H + 1)
    tempering = beta_k ** alpha

    # Combined Profile (before scaling)
    profile_raw = sigma_0 * sigma_growth * tempering

    # Normalize so mean = 1 (keeps S as the interpretable global scale)
    profile = profile_raw / profile_raw.mean()

    # Final sigma map: (B, H, 1)
    sigma = S * profile

    # Safety floor (prevents numerical instability)
    sigma_floor = 1e-4
    sigma = torch.maximum(sigma, torch.tensor(sigma_floor, device=device))

    # ========== 3. Correlated Noise (OU Process) ==========
    # rho=0.9 for smooth trajectories (from original design)
    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)

    # ========== 4. Construct Noisy Input ==========
    X_in = X_curr + sigma * noise_ou

    # ========== 5. CFG Dropout ==========
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    t_input = mask  # Time embedding: 1=visual, 0=blind

    # ========== 6. Network Prediction ==========
    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)

    # ========== 7. Weighted Loss ==========
    # Inverse variance weighting: focus on low-noise (high-precision) regions
    loss_raw = (X_pred - X_next) ** 2
    weight = 1.0 / (sigma ** 2 + 1e-4)
    loss = (loss_raw * weight).mean()

    loss = config.loss_scale * loss

    # ========== 8. Logging (Optional) ==========
    info = {
        "sigma_mean": sigma.mean().item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
        "sigma_pos0": sigma[0, 0, 0].item(),  # First position noise
        "sigma_posH": sigma[0, -1, 0].item(), # Last position noise
    }

    return loss, info

####!!!! 几何调度 + 部分手工超参数，也是目前 PushT State 中最大成功率最高的一组（98.97% / 91.67%）
def bridge_v2_loss03092(config, flow_map, encoder, interp, act, obs, delta_t):
    """Bridge Policy V2 Loss — Principled Schedule."""

    # 1. Slice Data
    X_curr = act[:, :-1, :]  # (B, H, D)
    X_next = act[:, 1:, :]   # (B, H, D)
    bs, H, D = X_curr.shape
    device = X_curr.device

    # ==========================
    # 2. Noise Schedule
    # ==========================

    # --- A. Training Curriculum (Cosine Annealing) ---
    current_step = int(delta_t[0].item())
    total_steps = config.gradient_steps
    progress = min(1.0, current_step / total_steps)

    sigma_high = 0.08   # 训练初期：重鲁棒性
    sigma_low  = 0.003  # 训练后期：重精度
    cos_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    sigma_center = sigma_low + (sigma_high - sigma_low) * cos_decay
    # 轨迹: 0.08 → 0.003, cosine 衰减

    # --- B. Per-Sample Random Scale (Log-Normal) ---
    log_mean = np.log(sigma_center)
    P_std = 1.0
    sigma_base = torch.exp(
        torch.randn((bs, 1, 1), device=device) * P_std + log_mean
    ).clamp(min=1e-4, max=0.5)

    # --- C. Position Profile (Geometric / Contraction Mapping) ---
    #
    # 数学动机：若每步 refinement 将误差缩小为 (1-α) 倍，
    # 则经过 m 步后残差 = σ_init · (1-α)^m。
    # 位置 k 还需经历 H-1-k 步 refinement 才被执行，
    # 所以: σ_k/σ_{H-1} = (1-α)^{H-1-k}
    # 令 r = (1-α)^{H-1} = σ_0/σ_{H-1}，则:
    #   profile(k) = r^{(H-1-k)/(H-1)}
    # 这恰好是 σ_min 到 σ_max 的几何插值。
    #
    # r 是唯一的形状超参数（控制 position-0 噪声占 base 的比例）

    r = 0.03  # position 0 获得 base 噪声的 3%
    t_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)
    profile = r ** (1.0 - t_k)   # k=0 → r, k=H-1 → 1.0

    # --- D. Final Sigma ---
    sigma = sigma_base * profile
    sigma = sigma.clamp(min=1e-4)

    # ==========================
    # 3. Noise (OU)
    # ==========================
    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
    X_in = X_curr + sigma * noise_ou

    # ==========================
    # 4. CFG (不变)
    # ==========================
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded
    t_input = mask

    # ==========================
    # 5. Prediction
    # ==========================
    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)

    # ==========================
    # 6. Loss (Uniform Weighting)
    # ==========================
    loss_raw = (X_pred - X_next) ** 2

    # BPv2 预测 X_next（而非噪声），即使 σ→0 任务也不 trivial，
    # 因此 1/σ² 加权没有理论依据，反而造成梯度方差爆炸。
    # 推荐：先试 uniform，若想微调精度可用 mild weighting。

    # Plan A: Uniform (推荐先试)
    loss = loss_raw.mean()

    # Plan B: Mild (若 uniform 精度不够，可试这个)
    # sigma_ref = 0.01
    # weight = sigma_ref / (sigma + sigma_ref)  # max ratio ~100x, 远小于 10000x
    # loss = (loss_raw * weight).mean()

    loss = config.loss_scale * loss
    return loss, {}


# def bridge_v2_loss(config, flow_map, encoder, interp, act, obs, delta_t):
#     """Bridge Policy V2 Loss — Chunk Diffusion Process (CDP).

#     One SDE describes the entire noise structure:
#         de_s = g(s) dW_s,  with σ(s) = σ_0 · (σ_max/σ_0)^s  (geometric)
#     Correlation emerges naturally: Corr(e_i, e_j) = r^{|i-j|/(H-1)}
#     Only 3 hyperparams (r, mu_scale, eta_scale). No curriculum, no OU.
#     """

#     # 1. Slice Data
#     X_curr = act[:, :-1, :]  # (B, H, D)
#     X_next = act[:, 1:, :]   # (B, H, D)
#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     # ===================================
#     # 2. CDP Noise Schedule
#     # ===================================
#     # --- Hyperparameters (only 3) ---
#     r         = 0.02    # σ_0 / σ_{H-1}  (contraction ratio α^{H-1})
#     mu_scale  = -3.0    # center of ln(σ_{H-1}), ≈ 0.05
#     eta_scale = 1.0     # spread of ln(σ_{H-1})

#     # A. Per-sample σ_max  (log-normal, covers all noise scales)
#     log_sigma_max = torch.randn((bs, 1, 1), device=device) * eta_scale + mu_scale
#     sigma_max = torch.exp(log_sigma_max).clamp(1e-4, 1.0)

#     # B. Geometric variance profile (from CDP SDE)
#     #    σ_k = σ_0 · R^{k/(H-1)},  R = σ_max / σ_0 = 1/r
#     s_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)  # (1, H, 1)
#     log_s0 = torch.log(r * sigma_max)    # (B, 1, 1)
#     log_sH = torch.log(sigma_max)        # (B, 1, 1)
#     sigmas = torch.exp((1 - s_k) * log_s0 + s_k * log_sH)  # (B, H, 1)

#     # C. Correlated noise via forward diffusion (cumsum)
#     #    Var(e_k) = σ_k²,  Cov(e_i, e_j) = σ_{min(i,j)}²
#     var_prof = sigmas ** 2
#     inc_var = torch.cat([var_prof[:, :1, :],
#                          var_prof[:, 1:, :] - var_prof[:, :-1, :]], dim=1)
#     inc_std = torch.sqrt(inc_var.clamp(min=1e-10))

#     z = torch.randn(bs, H, D, device=device)
#     # z = generate_ou_noise(bs, H, D, device, rho=0.9)
#     noise = torch.cumsum(inc_std * z, dim=1)   # (B, H, D)

#     # D. Corrupt
#     X_in = X_curr + noise

#     # ===================================
#     # 3. CFG
#     # ===================================
#     p_uncond = getattr(config, "p_uncond", 0.1)
#     mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
#     obs_emb = encoder(obs, None)
#     mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
#     obs_cond = obs_emb * mask_expanded
#     t_input = mask

#     # ===================================
#     # 4. Prediction & Loss
#     # ===================================
#     output = flow_map.get_velocity(t_input, X_in, obs_cond)

#     # ---------- MODE A: Direct Prediction (推荐) ----------
#     # 网络直接输出 X_next 的预测。
#     # Target 始终干净，网络隐式学习去噪 + 动力学。
#     # 推理: X_next = output
#     X_pred = output
#     loss = ((X_pred - X_next) ** 2).mean()

#     # ---------- MODE B: Clean Velocity (备选, 若 A 不稳定) ----------
#     # Target = X_next - X_curr (不含噪声!), 保持残差学习的数值优势。
#     # 推理: X_next = X_curr + output
#     # 注意: 此模式稳定但缺乏误差修正能力, 可能 plateau。
#     # V_target = X_next - X_curr
#     # loss = ((output - V_target) ** 2).mean()

#     loss = config.loss_scale * loss
#     return loss, {}


def bridge_v2_loss0310(config, flow_map, encoder, interp, act, obs, delta_t):
    """Bridge Policy V2 — Fixed CDP (geometric profile + OU noise, no curriculum)."""

    X_curr = act[:, :-1, :]
    X_next = act[:, 1:, :]
    bs, H, D = X_curr.shape
    device = X_curr.device

    # ===================================
    # CDP Noise — 3 处修改
    # ===================================

    # 修改 1: r 从 0.05 → 0.03（匹配 working 版本）
    r         = 0.03
    # 修改 2: mu_scale 从 -3.0 → -3.5（中心从 0.05 降到 0.03，更平衡）
    #          eta_scale 从 1.0 → 1.2（更宽的覆盖，弥补没有 curriculum）
    mu_scale  = -3.5    # center ≈ 0.03
    eta_scale = 1.2     # 更宽，让 ~8% 的样本落入 σ<0.01 精度区

    # Per-sample sigma_max（同 CDP）
    log_sigma_max = torch.randn((bs, 1, 1), device=device) * eta_scale + mu_scale
    sigma_max = torch.exp(log_sigma_max).clamp(1e-4, 0.5)

    # Geometric profile（同 CDP，这部分理论是对的）
    s_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)
    log_s0 = torch.log(r * sigma_max)
    log_sH = torch.log(sigma_max)
    sigmas = torch.exp((1 - s_k) * log_s0 + s_k * log_sH)  # (B, H, 1)

    # 修改 3: 噪声生成从 cumsum → sigma × OU
    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
    X_in = X_curr + sigmas * noise_ou    # 不再用 cumsum！

    # === 以下和之前完全相同 ===
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded
    t_input = mask

    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)
    loss = ((X_pred - X_next) ** 2).mean()

    loss = config.loss_scale * loss
    return loss, {}


def bridge_v2_loss0311(config, flow_map, encoder, interp, act, obs, delta_t):
    """Bridge Policy V2 — Fixed CDP + Sigma Conditioning."""

    X_curr = act[:, :-1, :]
    X_next = act[:, 1:, :]
    bs, H, D = X_curr.shape
    device = X_curr.device

    r         = 0.03
    mu_scale  = -3.5
    eta_scale = 1.2

    log_sigma_max = torch.randn((bs, 1, 1), device=device) * eta_scale + mu_scale
    sigma_max = torch.exp(log_sigma_max).clamp(1e-4, 0.5)

    s_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)
    log_s0 = torch.log(r * sigma_max)
    log_sH = torch.log(sigma_max)
    sigmas = torch.exp((1 - s_k) * log_s0 + s_k * log_sH)

    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
    X_in = X_curr + sigmas * noise_ou

    # === CFG + Sigma Conditioning ===
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    # # # ---- 核心改动：t_input 从 binary mask 变成 sigma level ----
    # # # 把 sigma_max 映射到 [0,1] 区间（适配网络的 time embedding）
    # # # log(sigma_max) ∈ [-9.2, -0.7]，线性映射到 [0, 1]
    # # log_sigma_val = torch.log(sigma_max).view(bs)       # (B,)
    # # t_sigma = ((log_sigma_val + 10.0) / 10.0).clamp(0.01, 0.99)
    # # t_input = t_sigma * mask                             # (B,)

    # # unconditional → t=0,  conditional → t=t_sigma
    # sigma_level = sigma_max.view(bs)    # (B,) 把 per-sample 噪声量级拿出来
    # t_input = sigma_level * mask        # uncond=0, cond=该 sample 的 sigma_max

    t_input = mask


    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)
    loss = ((X_pred - X_next) ** 2).mean()

    loss = config.loss_scale * loss
    return loss, {}



# def bridge_v2_loss(config, flow_map, encoder, interp, act, obs, delta_t):
#     """Bridge Policy V2 — Fixed CDP + Sigma Conditioning."""

#     X_curr = act[:, :-1, :]
#     X_next = act[:, 1:, :]
#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     r         = 0.03
#     mu_scale  = -3.5
#     eta_scale = 1.2

#     log_sigma_max = torch.randn((bs, 1, 1), device=device) * eta_scale + mu_scale
#     sigma_max = torch.exp(log_sigma_max).clamp(1e-4, 0.5)

#     s_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)
#     log_s0 = torch.log(r * sigma_max)
#     log_sH = torch.log(sigma_max)
#     sigmas = torch.exp((1 - s_k) * log_s0 + s_k * log_sH)

#     noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
#     X_in = X_curr + sigmas * noise_ou

#     # === CFG + Sigma Conditioning ===
#     p_uncond = getattr(config, "p_uncond", 0.1)
#     mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
#     obs_emb = encoder(obs, None)
#     mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
#     obs_cond = obs_emb * mask_expanded

#     # # # ---- 核心改动：t_input 从 binary mask 变成 sigma level ----
#     # # # 把 sigma_max 映射到 [0,1] 区间（适配网络的 time embedding）
#     # # # log(sigma_max) ∈ [-9.2, -0.7]，线性映射到 [0, 1]
#     # # log_sigma_val = torch.log(sigma_max).view(bs)       # (B,)
#     # # t_sigma = ((log_sigma_val + 10.0) / 10.0).clamp(0.01, 0.99)
#     # # t_input = t_sigma * mask                             # (B,)

#     # # unconditional → t=0,  conditional → t=t_sigma
#     # sigma_level = sigma_max.view(bs)    # (B,) 把 per-sample 噪声量级拿出来
#     # t_input = sigma_level * mask        # uncond=0, cond=该 sample 的 sigma_max

#     t_input = mask

#     ##!! 直接预测 X
#     X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)
#     loss = ((X_pred - X_next) ** 2).mean()

#     ##!! 预测速度场
#     # 构造 shifted noisy input (X_in 平移一步，末尾重复)
#     X_in_shifted = torch.cat([X_in[:, 1:, :], X_in[:, -1:, :]], dim=1)  # (B, H, D)

#     # 错位 velocity target
#     V_target = X_next - X_in_shifted

#     # 网络预测
#     V_pred = flow_map.get_velocity(t_input, X_in, obs_cond)
#     loss = ((V_pred - V_target) ** 2).mean()

#     loss = config.loss_scale * loss
#     return loss, {}


def bridge_v2_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """Bridge Policy V2 — Fixed CDP + Sigma Conditioning."""

    X_curr = act[:, :-1, :]
    X_next = act[:, 1:, :]
    bs, H, D = X_curr.shape
    device = X_curr.device

    # r         = 0.03
    sigma_lo  = 0.005
    sigma_hi  = 0.1
    alpha     = 0.80

    r = alpha ** (H - 1)   # ≈ 0.035

    log_sigma = torch.rand((bs, 1, 1), device=device) * (
        np.log(sigma_hi) - np.log(sigma_lo)
    ) + np.log(sigma_lo)
    sigma_max = torch.exp(log_sigma)   # ∈ [0.003, 0.3]，天然有界，无需 clamp

    s_k = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1)
    log_s0 = torch.log(r * sigma_max)
    log_sH = torch.log(sigma_max)
    sigmas = torch.exp((1 - s_k) * log_s0 + s_k * log_sH)

    noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
    X_in = X_curr + sigmas * noise_ou

    # === CFG + Sigma Conditioning ===
    p_uncond = getattr(config, "p_uncond", 0.1)
    mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
    obs_emb = encoder(obs, None)
    mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
    obs_cond = obs_emb * mask_expanded

    # # # ---- 核心改动：t_input 从 binary mask 变成 sigma level ----
    # # # 把 sigma_max 映射到 [0,1] 区间（适配网络的 time embedding）
    # # # log(sigma_max) ∈ [-9.2, -0.7]，线性映射到 [0, 1]
    # # log_sigma_val = torch.log(sigma_max).view(bs)       # (B,)
    # # t_sigma = ((log_sigma_val + 10.0) / 10.0).clamp(0.01, 0.99)
    # # t_input = t_sigma * mask                             # (B,)

    # # unconditional → t=0,  conditional → t=t_sigma
    # sigma_level = sigma_max.view(bs)    # (B,) 把 per-sample 噪声量级拿出来
    # t_input = sigma_level * mask        # uncond=0, cond=该 sample 的 sigma_max

    t_input = mask

    X_pred = flow_map.get_velocity(t_input, X_in, obs_cond)
    loss = ((X_pred - X_next) ** 2).mean()

    loss = config.loss_scale * loss
    return loss, {}


def generate_ou_noise(bs, H, D, device, rho=0.9):
    """
    Generate noise with OU-like correlation structure.
    rho: Correlation between adjacent steps (e.g. 0.9).
         rho=0 -> White Noise.
         rho=1 -> Bias (Perfect Correlation).
    """
    # 1. Construct Covariance Matrix
    # Sigma[i, j] = rho^|i-j|
    indices = torch.arange(H, device=device)
    # |i-j| matrix
    dist_matrix = torch.abs(indices.view(-1, 1) - indices.view(1, -1))
    cov_matrix = rho ** dist_matrix # (H, H)

    # 2. Cholesky Decomposition
    # L @ L.T = Cov
    L = torch.linalg.cholesky(cov_matrix) # (H, H)

    # 3. Generate Correlated Noise
    # z ~ N(0, I)
    z = torch.randn(bs, H, D, device=device)
    # x = z @ L.T
    # We want correlation along H dimension (dim 1)
    # (B, H, D) -> permute -> (B, D, H) @ (H, H) -> (B, D, H)
    noise = torch.matmul(z.transpose(1, 2), L.T).transpose(1, 2)

    return noise





# def bridge_v3_loss(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ) -> float:

#     """Bridge Policy V3 Loss — Spatiotemporal Unrolled Diffusion Bridge."""

#     # 专家动作直接作为 Ground Truth
#     X_gt = act
#     bs, H, D = X_gt.shape
#     device = X_gt.device

#     #! === BPv3 核心超参数 ===
#     sigma_min = 1e-4
#     sigma_max = 0.01
#     rho       = 0.9

#     R = sigma_max / sigma_min
#     obs_steps = 2
#     start_idx = obs_steps - 1  #!!! 极其关键：这是要执行的动作索引

#     # === 防线1: 连续空间采样 (Normal Equivalent) ===
#     # 采样 delta ~ N(0, 1)，让网络学到完全连续的流形，而不仅是离散的槽位
#     delta = torch.randn(bs, device=device)

#     # 计算当前批次中，每个槽位应该受到的真实物理噪声
#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     i_continuous = torch.clamp(i_indices + delta.unsqueeze(1), 0.0, float(H - 1))

#     # # 核心公式：几何调度
#     # sigma_train = sigma_min * (R ** (i_continuous / (H - 1))) # (bs, H)

#     ##!! 【核心修复】：只有在 start_idx 之后的动作才允许脏！
#     ## start_idx 之前的全部强制为 0 进度 (即 sigma_min)
#     L_eff = max(1.0, float(H - 1 - start_idx))
#     progress = torch.clamp((i_continuous - start_idx) / L_eff, 0.0, 1.0)

#     sigma_train = sigma_min * (R ** progress) # (bs, H)

#     # === 生成运动学平滑的 AR(1) 噪声 ===
#     # 直接复用你现成的完美函数！
#     noise_ou = generate_ou_noise(bs, H, D, device, rho)

#     # === 加噪过程 ===
#     X_noisy = X_gt + sigma_train.unsqueeze(-1) * noise_ou

#     # === 网络预测 ===
#     obs_emb = encoder(obs, None)

#     # 【神级替换】：直接把全局标量 delta 像以前的 t 一样喂给网络！
#     # 网络会自动把这个标量视作 "全局去噪进度条"
#     X_pred = flow_map.get_velocity(delta, X_noisy, obs_emb)

#     # === 极简 MSE Loss ===
#     loss = get_norm(X_pred - X_gt, config.norm_type)    ## L1 or L2 loss, 看配置选啥就是啥
#     loss = config.loss_scale * torch.mean(loss)

#     # === 3. 辅助约束: Spatiotemporal Consistency Loss (可选) ===
#     if getattr(config, "use_consistency_loss", False):
#         consist_beta = getattr(config, "consist_beta", 1.0)
#         with torch.no_grad():
#             # 模拟一步物理左移的 ODE 拉回
#             # 1. 算出向左平移1格后的目标噪声
#             i_continuous_shifted = torch.clamp(i_continuous - 1.0, 0.0, float(H - 1))

#             ##!! 【核心修复】：只有在 start_idx 之后的动作才允许脏！
#             ## start_idx 之前的全部强制为 0 进度 (即 sigma_min)
#             L_eff = max(1.0, float(H - 1 - start_idx))
#             progress = torch.clamp((i_continuous_shifted - start_idx) / L_eff, 0.0, 1.0)

#             sigma_target = sigma_min * (R ** progress) # (bs, H)
#             # sigma_target = sigma_min * (R ** (i_continuous_shifted / (H - 1)))

#             # 2. 施加一次向下的精确 ODE 欧拉步
#             X_pulled = X_noisy + (sigma_target - sigma_train).unsqueeze(-1) * (X_noisy - X_pred.detach()) / sigma_train.unsqueeze(-1)

#             # 3. 真正执行左移 Shift
#             X_shifted = torch.zeros_like(X_pulled)
#             X_shifted[:, :-1, :] = X_pulled[:, 1:, :]
#             X_shifted[:, -1, :] = torch.randn(bs, D, device=device) * sigma_max # 右侧填纯噪声

#             # 由于左移了一格，剩余数据相对于其槽位变得更"脏"了一档，delta必须 + 1.0
#             delta_new = delta + 1.0

#         # 再次送入网络预测
#         X_pred_consist = flow_map.get_velocity(delta_new, X_shifted, obs_emb)

#         # 约束：平移后的预测，必须对齐平移前预测的剩余部分
#         loss_consist = get_norm(X_pred_consist[:, :-1, :] - X_pred[:, 1:, :].detach(), config.norm_type)
#         loss += consist_beta * config.loss_scale * torch.mean(loss_consist)

#     return loss, {}


# def bridge_v3_loss(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ) -> float:

#     """Bridge Policy V3 Loss (Flow Matching with Spatiotemporal Consistency)."""

#     X_gt = act[:, 1:, :]
#     bs, H, D = X_gt.shape
#     device = X_gt.device

#     # === 1. 连续空间流形采样 (\delta 游标) ===
#     # 为了覆盖充分，我们在 [-0.5, 1.5] 附近采样
#     delta = torch.randn(bs, device=device)

#     # === 2. 映射为连续的 Flow 时间 \tau ===
#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     tau_train = torch.clamp(1.0 - (i_indices + delta.unsqueeze(1)) / (H - 1), 0.0, 1.0) # (B, H)

#     # === 3. 生成 AR(1) 物理噪声 ===
#     noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)

#     # === 4. 恒定方差的线性插值 (Forward Process) ===
#     X_noisy = tau_train.unsqueeze(-1) * X_gt + (1.0 - tau_train.unsqueeze(-1)) * noise_ou

#     # === 5. 网络预测速度场 V ===
#     obs_emb = encoder(obs, None)
#     V_pred = flow_map.get_velocity(delta, X_noisy, obs_emb)

#     # Flow Matching 完美目标速度： 指向干净数据
#     V_target = X_gt - noise_ou

#     # 主任务 Loss：预测速度
#     loss_task = get_norm(V_pred - V_target, config.norm_type)
#     loss = config.loss_scale * torch.mean(loss_task)

#     # === 6. 辅助约束: 时空一致性 Consistency Loss (极其优雅的 Flow 版本) ===
#     if getattr(config, "use_consistency_loss", False):
#         consist_beta = getattr(config, "consist_beta", 1.0)
#         with torch.no_grad():
#             # 模拟一步物理平移：如果系统向左走了一格，它的 delta 会增加 1.0
#             delta_shifted = delta + 1.0

#             # 计算平移一格后的理论 tau
#             tau_shifted = torch.clamp(1.0 - (i_indices + delta_shifted.unsqueeze(1)) / (H - 1), 0.0, 1.0)

#             # 我们直接在 "真实流形" 上进行插值平移，构建平移后的带噪数据
#             X_noisy_shifted = tau_shifted.unsqueeze(-1) * X_gt + (1.0 - tau_shifted.unsqueeze(-1)) * noise_ou

#         # 再次预测平移后的速度场
#         V_pred_shifted = flow_map.get_velocity(delta_shifted, X_noisy_shifted, obs_emb)

#         # 核心约束：平移后的预测，必须对齐平移前预测的剩余部分
#         # 注意：平移后槽位 0 装的是原来槽位 1 的数据，所以比较时索引错位
#         loss_consist = get_norm(V_pred_shifted[:, :-1, :] - V_pred[:, 1:, :].detach(), config.norm_type)
#         loss += consist_beta * config.loss_scale * torch.mean(loss_consist)

#     return loss, {}



# def bridge_v3_loss(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ) -> float:
#     """Bridge Policy V3 Continuous — Autoregressive Micro-Refiner with Delta Generalization."""
#     # ==========================================
#     # 1. 严格切片，构造 DAE 的输入和目标
#     # ==========================================
#     # 输入先验：保留过去的动作，舍弃最远端的未来
#     X_curr = act[:, :-1, :]  # 长度 16 (B, H, D)
#     # 预测目标：舍弃过去的动作，预测纯未来
#     X_next = act[:, 1:, :]   # 长度 16 (B, H, D)

#     bs, H, D = X_curr.shape
#     device = X_curr.device

#     # ==========================================
#     # 2. 局部微调的超参数
#     # ==========================================
#     sigma_min = 1.5e-3
#     sigma_max = 0.05  # 微小局部噪声，刚好覆盖物理形变和漂移
#     R = sigma_max / sigma_min

#     # ==========================================
#     # 3. 注入 BPv3 的灵魂：连续时空泛化 \delta
#     # ==========================================
#     # delta ~ N(0, 1.0)。这使得每个槽位见到的噪声不再固定
#     # 它在基准值附近做连续的指数级浮动！彻底解决泛化和 OOD 问题。

#     ##! delta = torch.randn(bs, device=device) * 1.0
#     # delta = torch.randn(bs, device=device).clamp(-1.0, 1.0)
#     delta = (torch.rand(bs, device=device) * 3.0) - 1.5

#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)

#     # 连续化调度：允许 delta 浮动，但限制最高进度不超过 H-1

#     ##! i_continuous = torch.clamp(i_indices + delta.unsqueeze(1), 0.0, float(H - 1))
#     i_continuous = i_indices + delta.unsqueeze(1)

#     # 几何计算
#     sigmas = sigma_min * (R ** (i_continuous / float(H - 1))) # (B, H)

#     # ==========================================
#     # 4. 生成运动学平滑的 AR(1) 噪声
#     # ==========================================
#     noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
#     X_in = X_curr + sigmas.unsqueeze(-1) * noise_ou

#     # # ==========================================
#     # # 5. CFG 与条件注入
#     # # ==========================================
#     # p_uncond = getattr(config, "p_uncond", 0.0)
#     # mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))

#     obs_emb = encoder(obs, None)
#     # mask_expanded = mask.view(-1, *([1] * (obs_emb.ndim - 1)))
#     # obs_cond = obs_emb * mask_expanded

#     # # 【核心】：把连续的 delta 传给网络！
#     # # 如果 mask=0 (uncond)，delta 强行置 0
#     # t_input = delta * mask

#     # ==========================================
#     # 6. 一步到位的 DAE 回归预测
#     # ==========================================
#     X_pred = flow_map.get_velocity(delta, X_in, obs_emb)   ##!! 先直接用 delta 和正常的 obs 试试吧

#     # ==========================================
#     # 7. 稳如泰山的 Uniform Loss
#     # ==========================================
#     loss = get_norm(X_pred - X_next, config.norm_type)
#     loss = config.loss_scale * torch.mean(loss)

#     # ==========================================
#     # 辅助 Loss: 强悍的时空一致性约束
#     # ==========================================
#     if getattr(config, "use_consistency_loss", True):
#         consist_beta = getattr(config, "consist_beta", 1.0)
#         with torch.no_grad():
#             # 物理模拟：向左平移 1 步
#             delta_shifted = delta + 1.0
#             progress_shifted = (i_indices + delta_shifted.unsqueeze(1)) / (H - 1)
#             sigmas_shifted = sigma_min * (R ** progress_shifted)

#             X_noisy_shifted = torch.zeros_like(X_in)

#             # 前 H-1 个元素，完美继承上一刻的数据（直接复用 X_in 的后 H-1 个）
#             # 为什么可以直接复用？
#             # 因为原公式 sigma_k(delta) = sigma_min * R^((k+delta)/(H-1))
#             # 所以 sigma_{k+1}(delta) 精确等于 sigma_k(delta+1)
#             X_noisy_shifted[:, :-1, :] = X_in[:, 1:, :]


#             # 最右侧用上一步动作 padding，并加上该槽位对应的噪声(用 X_curr 而不是 X_in，因为 X_in 是加噪后的，用 X_in 就相当于搞了两次噪声！)
#             X_noisy_shifted[:, -1, :] = X_curr[:, -1, :] + sigmas_shifted[:, -1].unsqueeze(-1) * torch.randn(bs, D, device=device)

#         X_pred_shifted = flow_map.get_velocity(delta_shifted, X_noisy_shifted, obs_emb)

#         # 约束平移前后的纯净预测意图一致
#         loss_consist = get_norm(X_pred_shifted[:, :-1, :] - X_pred[:, 1:, :].detach(), config.norm_type)
#         loss += consist_beta * config.loss_scale * torch.mean(loss_consist)


#     return loss, {}



# def bridge_v3_loss(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ) -> float:
#     """Bridge Policy V3 Loss — Decoupled Epsilon-Prediction for Ultimate Robustness."""
#     X_gt = act # 【解耦！】不再切片预测 X_next，目标就是当前时刻的完美 X
#     bs, H, D = X_gt.shape
#     device = X_gt.device

#     # === 1. 极其宽泛且普适的超参数 (不再敏感！) ===
#     sigma_min = 1.5e-4  # 左端执行位，极度干净
#     sigma_max = 0.05    # 右端盲区，纯粹的宇宙大爆炸！逼迫网络 Inpainting
#     R = sigma_max / sigma_min

#     # === 2. 连续时空泛化 (无 clamp，最平滑的对数滑动) ===
#     # delta ~ U[-1.5, 1.5]
#     delta = (torch.rand(bs, device=device) * 3.0) - 1.5

#     i_indices = torch.arange(H, device=device).unsqueeze(0).expand(bs, H)
#     delta_scale = math.log(R) / (H - 1)

#     # 几何曲线随 delta 平滑上下浮动
#     sigmas = sigma_min * (R ** (i_indices / (H - 1))) * torch.exp(delta.unsqueeze(1) * delta_scale)
#     sigmas = torch.clamp(sigmas, min=sigma_min, max=sigma_max) # 物理安全守卫

#     # === 3. 生成纯标准噪声 ===
#     # 【核心！】：网络的目标变成了预测这个纯正的 N(0,1)！
#     noise_standard = torch.randn(bs, H, D, device=device)

#     # 依然可以融入一点运动学 AR 惯性来构建输入
#     noise_ou = generate_ou_noise(bs, H, D, device, rho=0.9)
#     # 但是！我们主要由 standard noise 主导，以契合 epsilon 预测的数学期望
#     X_noisy = X_gt + sigmas.unsqueeze(-1) * noise_standard

#     # === 4. 条件注入 ===
#     p_uncond = getattr(config, "p_uncond", 0.1)
#     mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device))
#     obs_emb = encoder(obs, None)
#     obs_cond = obs_emb * mask.view(-1, *([1] * (obs_emb.ndim - 1)))

#     t_input = delta * mask

#     # === 5. 网络预测 Epsilon ===
#     # 【重参数化！】：网络输出的不再是 X，也不是被 sigma 缩放过的 V
#     # 网络输出的尺度永远在 [-3, 3] 之间！(标准正态的范围)
#     Eps_pred = flow_map.get_velocity(delta, X_noisy, obs_emb)

#     # === 6. 永远 O(1) 的 Loss ===
#     # 不管 sigma 是 1e-4 还是 100，Loss 的量级完全一样！
#     # 网络会被迫用一模一样的注意力去死磕最左侧那 1e-4 的微小偏差！
#     loss = get_norm(Eps_pred - noise_standard, config.norm_type)
#     loss = config.loss_scale * torch.mean(loss)

#     # (辅助 Consistency Loss 可以加，但在解耦模式下，由于网络不再承担平移包袱，通常不再需要)
#     return loss, {}




def bridge_v3_loss0318(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Bridge Policy V3 Loss -- Shift-Then-Refine Residual Learning

    IMPORTANT:
        act must have shape (B, H+1, D)

    Construction:
        C_s^*      = act[:, :-1, :]   # current teacher chunk, length H
        C_{s+1}^*  = act[:, 1:, :]    # next teacher chunk, length H

    We explicitly hard-code the shift operator, and train the network to predict
    only the residual refinement after shifting.
    """
    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device

    assert T >= 2, f"act length must be at least 2, got {T}"
    H = T - 1  # effective chunk length used by the policy

    # --------------------------------------------------
    # 1. Teacher chunks
    # --------------------------------------------------
    X_curr_gt = act[:, :-1, :]   # (B, H, D)
    X_next_gt = act[:, 1:, :]    # (B, H, D)

    # hard-coded shift warm-start
    X_shift_gt = torch.zeros_like(X_curr_gt)
    X_shift_gt[:, :-1, :] = X_curr_gt[:, 1:, :]
    X_shift_gt[:, -1, :] = X_curr_gt[:, -1, :]   # tail copy warm-start

    # residual target: what remains to be corrected after shift
    R_gt = X_next_gt - X_shift_gt

    # --------------------------------------------------
    # 2. Training perturbation on shifted chunk
    # --------------------------------------------------
    # This noise is ONLY for robustness. It should be much smaller / saner
    # than the previous "algorithm-defining" noise.
    # sigma_min = getattr(config, "bridge_sigma_min", 0.0)
    # sigma_max = getattr(config, "bridge_sigma_max", 0.0)
    sigma_min = 0.0
    sigma_max = 0.01
    noise_type = getattr(config, "bridge_noise_type", "ou")   # ["ou", "gaussian", "none"]

    # position-dependent schedule: cleaner on the left, noisier on the right
    if sigma_max > 0.0:
        if sigma_min <= 0.0:
            # linear schedule is safer when sigma_min == 0
            i_idx = torch.arange(H, device=device, dtype=X_curr_gt.dtype).unsqueeze(0).expand(bs, H)
            if H > 1:
                sigmas = sigma_max * (i_idx / float(H - 1))
            else:
                sigmas = torch.full((bs, H), sigma_max, device=device, dtype=X_curr_gt.dtype)
        else:
            R_sigma = sigma_max / sigma_min
            i_idx = torch.arange(H, device=device, dtype=X_curr_gt.dtype).unsqueeze(0).expand(bs, H)
            sigmas = sigma_min * (R_sigma ** (i_idx / float(max(H - 1, 1))))
    else:
        sigmas = torch.zeros((bs, H), device=device, dtype=X_curr_gt.dtype)

    if noise_type == "ou" and sigma_max > 0.0:
        noise = generate_ou_noise(bs, H, D, device, rho=getattr(config, "bridge_noise_rho", 0.9))
    elif noise_type == "gaussian" and sigma_max > 0.0:
        noise = torch.randn(bs, H, D, device=device, dtype=X_curr_gt.dtype)
    else:
        noise = torch.zeros(bs, H, D, device=device, dtype=X_curr_gt.dtype)

    X_in = X_shift_gt + sigmas.unsqueeze(-1) * noise

    # --------------------------------------------------
    # 3. Encode observations
    # --------------------------------------------------
    obs_emb = encoder(obs, None)

    # We keep a scalar conditioning "t" only as a lightweight interface token.
    # In this version it does NOT carry the whole algorithm semantics anymore.
    # Default: use 1.0 to indicate "one-step shifted input".
    t_input = torch.ones((bs,), device=device, dtype=X_curr_gt.dtype) * getattr(config, "bridge_t_value", 1.0)

    # Optional unconditional dropout for compatibility with CFG-style training
    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(torch.full((bs,), 1 - p_uncond, device=device, dtype=X_curr_gt.dtype))
        obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

    # --------------------------------------------------
    # 4. Predict residual refinement
    # --------------------------------------------------
    R_pred = flow_map.get_velocity(t_input, X_in, obs_emb)

    # refined next chunk
    X_next_pred = X_in + R_pred

    # --------------------------------------------------
    # 5. Losses
    # --------------------------------------------------
    # Main residual loss
    loss_res = get_norm(R_pred - R_gt, config.norm_type)
    loss_res = torch.mean(loss_res)

    # Optional reconstruction loss on next chunk
    recon_beta = getattr(config, "bridge_recon_beta", 1.0)
    loss_recon = get_norm(X_next_pred - X_next_gt, config.norm_type)
    loss_recon = torch.mean(loss_recon)

    loss = loss_res + recon_beta * loss_recon

    # Optional smoothness regularization on predicted chunk
    smooth_beta = getattr(config, "bridge_smooth_beta", 0.0)
    if smooth_beta > 0.0 and H > 1:
        diff1 = X_next_pred[:, 1:, :] - X_next_pred[:, :-1, :]
        loss_smooth = torch.mean(get_norm(diff1, config.norm_type))
        loss = loss + smooth_beta * loss_smooth
    else:
        loss_smooth = torch.tensor(0.0, device=device, dtype=X_curr_gt.dtype)

    aux = {
        "loss_res": float(loss_res.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "chunk_horizon": H,
    }

    return config.loss_scale * loss, {}



# def bridge_v3_loss(
#     config: OptimizationConfig,
#     flow_map: FlowMap,
#     encoder: BaseEncoder,
#     interp: Interpolant,
#     act: torch.Tensor,
#     obs: torch.Tensor,
#     delta_t: torch.Tensor,
# ):
#     """
#     Bridge Policy V3 Loss -- Explicit Shift + Full Next-Chunk Prediction

#     IMPORTANT:
#         act must have shape (B, H+1, D)

#     Meaning:
#         X_curr_gt = act[:, :-1, :]   # current teacher chunk, length H
#         X_next_gt = act[:, 1:, :]    # next teacher chunk, length H

#     We explicitly shift X_curr_gt, corrupt the shifted chunk with small noise,
#     and train the network to reconstruct the full clean next chunk X_next_gt.
#     """
#     assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
#     bs, T, D = act.shape
#     device = act.device
#     dtype = act.dtype

#     assert T >= 2, f"act length must be at least 2, got {T}"
#     H = T - 1

#     # ==========================================================
#     # 1. Teacher chunks
#     # ==========================================================
#     X_curr_gt = act[:, :-1, :]   # (B, H, D)
#     X_next_gt = act[:, 1:, :]    # (B, H, D)

#     # Explicit shift warm-start
#     X_shift_gt = torch.zeros_like(X_curr_gt)
#     X_shift_gt[:, :-1, :] = X_curr_gt[:, 1:, :]
#     X_shift_gt[:, -1, :] = X_curr_gt[:, -1, :]   # tail copy

#     # ==========================================================
#     # 2. Small structured corruption on shifted chunk
#     # ==========================================================
#     sigma_min = getattr(config, "bridge_sigma_min", 0.0)
#     sigma_max = getattr(config, "bridge_sigma_max", 0.01)
#     noise_type = getattr(config, "bridge_noise_type", "ou")   # ["ou", "gaussian", "none"]
#     noise_rho = getattr(config, "bridge_noise_rho", 0.9)

#     if sigma_max > 0.0:
#         i_idx = torch.arange(H, device=device, dtype=dtype).unsqueeze(0).expand(bs, H)

#         if sigma_min > 0.0:
#             ratio = sigma_max / sigma_min
#             sigmas = sigma_min * (ratio ** (i_idx / float(max(H - 1, 1))))
#         else:
#             # safer linear schedule when sigma_min == 0
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

#         X_in = X_shift_gt + sigmas.unsqueeze(-1) * noise
#     else:
#         X_in = X_shift_gt

#     # ==========================================================
#     # 3. Observation encoding
#     # ==========================================================
#     obs_emb = encoder(obs, None)

#     # Optional CFG-style dropout
#     p_uncond = getattr(config, "p_uncond", 0.0)
#     if p_uncond > 0.0:
#         keep_mask = torch.bernoulli(
#             torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
#         )
#         obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

#     # ==========================================================
#     # 4. Conditioning scalar t
#     # ==========================================================
#     # We keep t as a lightweight interface token for compatibility with existing backbones.
#     # In this version it simply indicates "shifted-and-corrupted one-step warm start".
#     t_input = torch.ones((bs,), device=device, dtype=dtype) * getattr(config, "bridge_t_value", 1.0)

#     # ==========================================================
#     # 5. Predict the FULL clean next chunk
#     # ==========================================================
#     X_pred = flow_map.get_velocity(t_input, X_in, obs_emb)

#     # ==========================================================
#     # 6. Loss
#     # ==========================================================
#     loss_main = get_norm(X_pred - X_next_gt, config.norm_type)
#     loss_main = torch.mean(loss_main)

#     loss = loss_main

#     # Optional smoothness on prediction
#     smooth_beta = getattr(config, "bridge_smooth_beta", 0.0)
#     if smooth_beta > 0.0 and H > 1:
#         diff1 = X_pred[:, 1:, :] - X_pred[:, :-1, :]
#         loss_smooth = torch.mean(get_norm(diff1, config.norm_type))
#         loss = loss + smooth_beta * loss_smooth
#     else:
#         loss_smooth = torch.tensor(0.0, device=device, dtype=dtype)

#     aux = {
#         "loss_main": float(loss_main.detach().cpu()),
#         "loss_smooth": float(loss_smooth.detach().cpu()),
#         "chunk_horizon": H,
#     }

#     return config.loss_scale * loss, aux





def bridge_v3_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
):
    """
    Bridge Policy V5 Loss
    Single-step Explicit Shift + Preconditioned Residual Prediction

    IMPORTANT:
        act must have shape (B, H+1, D)

    Teacher chunks:
        X_curr_gt = act[:, :-1, :]   # current chunk
        X_next_gt = act[:, 1:, :]    # next chunk

    We explicitly shift X_curr_gt to form a warm start, then ask the network
    to predict a per-slot normalized residual:
        Y = (X_next_gt - X_shift_gt) / scale

    This keeps target magnitude more uniform across chunk positions.
    """
    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device
    dtype = act.dtype

    assert T >= 2, f"act length must be at least 2, got {T}"
    H = T - 1

    # ==========================================================
    # 1. Teacher chunks
    # ==========================================================
    X_curr_gt = act[:, :-1, :]   # (B, H, D)
    X_next_gt = act[:, 1:, :]    # (B, H, D)

    # Explicit shift warm-start
    X_shift_gt = torch.zeros_like(X_curr_gt)
    X_shift_gt[:, :-1, :] = X_curr_gt[:, 1:, :]
    X_shift_gt[:, -1, :] = X_curr_gt[:, -1, :]   # tail copy

    # Raw residual teacher target
    R_gt = X_next_gt - X_shift_gt

    # ==========================================================
    # 2. Preconditioning scale per slot
    # ==========================================================
    # This scale is NOT algorithmic noise. It is a target normalization profile.
    scale_min = getattr(config, "bridge_scale_min", 0.02)
    scale_max = getattr(config, "bridge_scale_max", 1.0)
    scale_eps = getattr(config, "bridge_scale_eps", 1e-6)

    assert scale_max >= scale_min > 0.0

    i_idx = torch.arange(H, device=device, dtype=dtype).unsqueeze(0).expand(bs, H)

    if H > 1:
        if getattr(config, "bridge_scale_schedule", "exp") == "linear":
            scales = scale_min + (scale_max - scale_min) * (i_idx / float(H - 1))
        else:
            ratio = scale_max / scale_min
            scales = scale_min * (ratio ** (i_idx / float(H - 1)))
    else:
        scales = torch.full((bs, H), scale_max, device=device, dtype=dtype)

    # Normalized target
    Y_gt = R_gt / (scales.unsqueeze(-1) + scale_eps)

    # ==========================================================
    # 3. Optional small input perturbation on shifted chunk
    # ==========================================================
    noise_std = getattr(config, "bridge_input_noise_std", 0.0)
    noise_type = getattr(config, "bridge_input_noise_type", "none")  # ["none", "gaussian", "ou"]
    noise_rho = getattr(config, "bridge_input_noise_rho", 0.9)

    if noise_std > 0.0:
        if noise_type == "ou":
            noise = generate_ou_noise(bs, H, D, device, rho=noise_rho).to(dtype)
        elif noise_type == "gaussian":
            noise = torch.randn(bs, H, D, device=device, dtype=dtype)
        else:
            noise = torch.zeros(bs, H, D, device=device, dtype=dtype)

        X_in = X_shift_gt + noise_std * noise
    else:
        X_in = X_shift_gt

    # ==========================================================
    # 4. Encode obs
    # ==========================================================
    obs_emb = encoder(obs, None)

    # Optional cond dropout for CFG compatibility
    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

    # Lightweight condition token, no heavy semantics
    t_input = torch.ones((bs,), device=device, dtype=dtype) * getattr(config, "bridge_t_value", 1.0)

    # ==========================================================
    # 5. Predict normalized residual
    # ==========================================================
    Y_pred = flow_map.get_velocity(t_input, X_in, obs_emb)

    # Recover residual and next chunk
    R_pred = Y_pred * (scales.unsqueeze(-1) + scale_eps)
    X_next_pred = X_shift_gt + R_pred

    # ==========================================================
    # 6. Loss
    # ==========================================================
    # Main loss on normalized target
    loss_norm_res = get_norm(Y_pred - Y_gt, config.norm_type)
    loss_norm_res = torch.mean(loss_norm_res)

    loss = loss_norm_res

    # Optional auxiliary reconstruction loss on next chunk
    recon_beta = getattr(config, "bridge_recon_beta", 0.0)
    if recon_beta > 0.0:
        loss_recon = get_norm(X_next_pred - X_next_gt, config.norm_type)
        loss_recon = torch.mean(loss_recon)
        loss = loss + recon_beta * loss_recon
    else:
        loss_recon = torch.tensor(0.0, device=device, dtype=dtype)

    aux = {
        "loss_norm_res": float(loss_norm_res.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "chunk_horizon": H,
    }

    return config.loss_scale * loss, aux




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


def _build_maturity_profile(
    H: int,
    device,
    dtype,
    mode: str = "linear",
    gamma: float = 0.2,
    beta: float = 2.0,
):
    """
    Build maturity profile m in [gamma, 1], shape (1, H, 1).

    Semantics:
      m[0] = 1.0   -> leftmost slot fully mature
      m[-1] ~ gamma -> rightmost slot still coarse
    """
    if H == 1:
        m = torch.ones(1, 1, 1, device=device, dtype=dtype)
        return m

    idx = torch.arange(H, device=device, dtype=dtype)

    if mode == "exp":
        # normalized exp decay from 1 to gamma
        # raw_i = exp(-beta * i/(H-1))
        raw = torch.exp(-beta * idx / float(H - 1))
        raw = (raw - raw[-1]) / (raw[0] - raw[-1] + 1e-8)  # map to [1,0]
        m_vec = gamma + (1.0 - gamma) * raw
    else:
        # linear
        m_vec = 1.0 - (1.0 - gamma) * (idx / float(H - 1))

    return m_vec.view(1, H, 1)

def _build_noise_profile(
    H: int,
    device,
    dtype,
    sigma_min: float,
    sigma_max: float,
    mode: str = "linear",
    beta: float = 2.0,
):
    """
    Build age-structured noise scale profile, shape (1, H, 1).

    Left slots: cleaner
    Right slots: noisier
    """
    if H == 1:
        sigma = torch.full((1, 1, 1), sigma_max, device=device, dtype=dtype)
        return sigma

    idx = torch.arange(H, device=device, dtype=dtype)

    if mode == "exp":
        raw = torch.exp(beta * idx / float(H - 1))
        raw = (raw - raw[0]) / (raw[-1] - raw[0] + 1e-8)  # [0,1]
        sigma_vec = sigma_min + (sigma_max - sigma_min) * raw
    else:
        sigma_vec = sigma_min + (sigma_max - sigma_min) * (idx / float(H - 1))

    return sigma_vec.view(1, H, 1)

def _sample_structured_noise(
    bs: int,
    H: int,
    D: int,
    device,
    dtype,
    noise_type: str = "gaussian",
    rho: float = 0.9,
):
    if noise_type == "ou":
        return generate_ou_noise(bs, H, D, device, rho=rho).to(dtype)
    else:
        return torch.randn(bs, H, D, device=device, dtype=dtype)

def prcp_v1_loss0321(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
):
    """
    PRCP-v1:
    Single-step progressive rollout-ready supervision.

    Required action length:
        act.shape[1] >= config.task.horizon

    Internal chunk length:
        Hp = config.task.horizon - 1
    """
    assert hasattr(config, "roll_chunk_horizon"), "config.roll_chunk_horizon missing; call sync_runtime_config(config) first"
    assert hasattr(config, "obs_steps"), "config.obs_steps missing; call sync_runtime_config(config) first"

    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device
    dtype = act.dtype

    Hp = T - 1  # Internal chunk length
    # assert T >= Hp + 1, f"PRCP-v1 requires action length >= {Hp+1}, got {T}"

    # teacher chunks
    C0 = act[:, 0:Hp, :]       # (B, Hp, D)
    C1 = act[:, 1:Hp+1, :]     # (B, Hp, D)

    # shifted warm-start
    S0 = _shift_chunk(C0, act_steps=1)

    # maturity profile
    m = _build_maturity_profile(
        H=Hp,
        device=device,
        dtype=dtype,
        mode=getattr(config, "prcp_maturity_mode", "linear"),
        gamma=getattr(config, "prcp_gamma", 0.2),
        beta=getattr(config, "prcp_beta", 2.0),
    )  # (1, Hp, 1)

    # rollout-ready teacher target
    C1_ready = m * C1 + (1.0 - m) * S0

    # obs window: first obs_steps only
    To = config.obs_steps
    if isinstance(obs, dict):
        obs_t = {k: v[:, :To, ...] for k, v in obs.items()}
    else:
        obs_t = obs[:, :To, ...]

    # encode
    obs_emb = encoder(obs_t, None)

    # rolling stage token
    t_roll = getattr(config, "prcp_t_roll", 1.0)
    t_input = torch.full((bs,), t_roll, device=device, dtype=dtype)

    # optional cond dropout
    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

    # predict next rollout-ready chunk
    C1_pred = flow_map.get_velocity(t_input, S0, obs_emb)

    # main loss
    loss_ready = get_norm(C1_pred - C1_ready, config.norm_type)
    loss_ready = torch.mean(loss_ready)

    # optional auxiliary full-target loss (small weight if used)
    full_beta = getattr(config, "prcp_full_beta", 0.0)
    if full_beta > 0.0:
        loss_full = get_norm(C1_pred - C1, config.norm_type)
        loss_full = torch.mean(loss_full)
        loss = loss_ready + full_beta * loss_full
    else:
        loss_full = torch.tensor(0.0, device=device, dtype=dtype)
        loss = loss_ready

    aux = {
        "loss_ready": float(loss_ready.detach().cpu()),
        "loss_full": float(loss_full.detach().cpu()),
        "Hp": Hp,
    }
    return config.loss_scale * loss, aux


def prcp_v1_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
):
    """
    PRCP-v1 (noisy):
    Single-step progressive rollout-ready supervision
    with age-structured forward corruption on shifted teacher input.

    Expected:
        act.shape[1] = H+1
    Internal chunk length:
        Hp = T - 1
    """
    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device
    dtype = act.dtype

    Hp = T - 1
    C0 = act[:, 0:Hp, :]
    C1 = act[:, 1:Hp+1, :]

    # teacher shift
    S0 = _shift_chunk(C0, act_steps=1)

    # maturity profile
    m = _build_maturity_profile(
        H=Hp,
        device=device,
        dtype=dtype,
        mode=getattr(config, "prcp_maturity_mode", "linear"),
        gamma=getattr(config, "prcp_gamma", 0.2),
        beta=getattr(config, "prcp_beta", 2.0),
    )

    # rollout-ready target
    C1_ready = m * C1 + (1.0 - m) * S0

    # obs window
    To = config.obs_steps
    if isinstance(obs, dict):
        obs_t = {k: v[:, :To, ...] for k, v in obs.items()}
    else:
        obs_t = obs[:, :To, ...]

    # age-structured noise
    sigma = _build_noise_profile(
        H=Hp,
        device=device,
        dtype=dtype,
        sigma_min=getattr(config, "prcp_noise_min", 1.0e-3),
        sigma_max=getattr(config, "prcp_noise_max", 0.05),
        mode=getattr(config, "prcp_noise_profile", "linear"),
        beta=getattr(config, "prcp_noise_beta", 2.0),
    )
    eps = _sample_structured_noise(
        bs=bs,
        H=Hp,
        D=D,
        device=device,
        dtype=dtype,
        noise_type=getattr(config, "prcp_noise_type", "ou"),
        rho=getattr(config, "prcp_noise_rho", 0.9),
    )
    S0_noisy = S0 + sigma * eps

    # encode
    obs_emb = encoder(obs_t, None)

    # rolling stage token
    t_roll = getattr(config, "prcp_t_roll", 1.0)
    t_input = torch.full((bs,), t_roll, device=device, dtype=dtype)

    # optional cond dropout
    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

    # predict next rollout-ready chunk
    C1_pred = flow_map.get_velocity(t_input, S0_noisy, obs_emb)

    # main loss: progressive target
    loss_ready = torch.mean(get_norm(C1_pred - C1_ready, config.norm_type))
    loss = loss_ready

    # optional auxiliary full target
    full_beta = getattr(config, "prcp_full_beta", 0.0)
    if full_beta > 0.0:
        loss_full = torch.mean(get_norm(C1_pred - C1, config.norm_type))
        loss = loss + full_beta * loss_full
    else:
        loss_full = torch.tensor(0.0, device=device, dtype=dtype)

    aux = {
        "loss_ready": float(loss_ready.detach().cpu()),
        "loss_full": float(loss_full.detach().cpu()),
        "Hp": Hp,
    }
    return config.loss_scale * loss, aux


def prcp_v2_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
):
    """
    PRCP-v2 (noisy):
    Two-step rollout-ready supervision with:
      - step1: noisy teacher shift input
      - step2: student rollout input + optional smaller noise

    Expected:
        act.shape[1] = H+2
    Internal chunk length:
        Hp = T - 2
    """
    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device
    dtype = act.dtype

    Hp = T - 2

    C0 = act[:, 0:Hp, :]
    C1 = act[:, 1:Hp+1, :]
    C2 = act[:, 2:Hp+2, :]

    S0 = _shift_chunk(C0, act_steps=1)
    S1 = _shift_chunk(C1, act_steps=1)

    m = _build_maturity_profile(
        H=Hp,
        device=device,
        dtype=dtype,
        mode=getattr(config, "prcp_maturity_mode", "linear"),
        gamma=getattr(config, "prcp_gamma", 0.2),
        beta=getattr(config, "prcp_beta", 2.0),
    )

    C1_ready = m * C1 + (1.0 - m) * S0
    C2_ready = m * C2 + (1.0 - m) * S1

    To = config.obs_steps

    def slice_obs(obs_in, start, length):
        if isinstance(obs_in, dict):
            return {k: v[:, start:start+length, ...] for k, v in obs_in.items()}
        else:
            return obs_in[:, start:start+length, ...]

    obs_t = slice_obs(obs, 0, To)
    obs_t1 = slice_obs(obs, 1, To)

    # step1 noise profile
    sigma1 = _build_noise_profile(
        H=Hp,
        device=device,
        dtype=dtype,
        sigma_min=getattr(config, "prcp_noise_min", 1.0e-3),       # 0.0
        sigma_max=getattr(config, "prcp_noise_max", 0.05),      # 0.03
        mode=getattr(config, "prcp_noise_profile", "linear"),
        beta=getattr(config, "prcp_noise_beta", 2.0),
    )

    # step2 noise profile (usually smaller)
    step2_scale = getattr(config, "prcp_step2_noise_scale", 0.3)
    sigma2 = sigma1 * step2_scale

    noise_type = getattr(config, "prcp_noise_type", "ou")       # "gaussian"
    noise_rho = getattr(config, "prcp_noise_rho", 0.9)
    t_roll = getattr(config, "prcp_t_roll", 1.0)
    p_uncond = getattr(config, "p_uncond", 0.0)

    def encode_obs(obs_window):
        obs_emb = encoder(obs_window, None)
        if p_uncond > 0.0:
            keep_mask = torch.bernoulli(
                torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
            )
            obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))
        return obs_emb

    def predict_next(C_in, obs_window):
        obs_emb = encode_obs(obs_window)
        t_input = torch.full((bs,), t_roll, device=device, dtype=dtype)
        return flow_map.get_velocity(t_input, C_in, obs_emb)

    # -------------------------
    # Step 1: teacher shift + structured noise
    # -------------------------
    eps1 = _sample_structured_noise(
        bs=bs, H=Hp, D=D, device=device, dtype=dtype,
        noise_type=noise_type, rho=noise_rho
    )
    S0_noisy = S0 + sigma1 * eps1
    C1_pred = predict_next(S0_noisy, obs_t)

    # -------------------------
    # Step 2: student rollout + optional small noise
    # -------------------------
    S1_pred = _shift_chunk(C1_pred, act_steps=1)

    add_step2_noise = getattr(config, "prcp_use_step2_noise", True)
    if add_step2_noise:
        eps2 = _sample_structured_noise(
            bs=bs, H=Hp, D=D, device=device, dtype=dtype,
            noise_type=noise_type, rho=noise_rho
        )
        S1_pred_in = S1_pred + sigma2 * eps2
    else:
        S1_pred_in = S1_pred

    C2_pred = predict_next(S1_pred_in, obs_t1)

    # losses
    loss_step1 = torch.mean(get_norm(C1_pred - C1_ready, config.norm_type))
    loss_step2 = torch.mean(get_norm(C2_pred - C2_ready, config.norm_type))

    rollout_beta = getattr(config, "prcp_rollout_beta", 1.0)
    loss = loss_step1 + rollout_beta * loss_step2

    full_beta = getattr(config, "prcp_full_beta", 0.0)
    if full_beta > 0.0:
        loss_full1 = torch.mean(get_norm(C1_pred - C1, config.norm_type))
        loss_full2 = torch.mean(get_norm(C2_pred - C2, config.norm_type))
        loss = loss + full_beta * (loss_full1 + rollout_beta * loss_full2)
    else:
        loss_full1 = torch.tensor(0.0, device=device, dtype=dtype)
        loss_full2 = torch.tensor(0.0, device=device, dtype=dtype)

    aux = {
        "loss_step1": float(loss_step1.detach().cpu()),
        "loss_step2": float(loss_step2.detach().cpu()),
        "loss_full1": float(loss_full1.detach().cpu()),
        "loss_full2": float(loss_full2.detach().cpu()),
        "Hp": Hp,
    }
    return config.loss_scale * loss, aux




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



def rolling_policy_v1_loss0322(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
):
    """
    Rolling Policy v1
    - fixed rolling noise ladder
    - clean chunk prediction
    - per-slot noise embedding
    """
    assert hasattr(config, "obs_steps"), "config.obs_steps missing; sync task fields into optimization config first"

    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device
    dtype = act.dtype

    X_clean = act  # teacher clean window

    # obs window
    To = config.obs_steps
    if isinstance(obs, dict):
        obs_t = {k: v[:, :To, ...] for k, v in obs.items()}
    else:
        obs_t = obs[:, :To, ...]

    tau_vec = _build_tau_ladder(
        T=T,
        device=device,
        dtype=dtype,
        tau_min=config.rolling_tau_min,
        tau_max=config.rolling_tau_max,
        mode=config.rolling_tau_mode,
        beta=config.rolling_tau_beta,
    )  # (1, T)

    eps = _sample_noise(
        bs=bs,
        T=T,
        D=D,
        device=device,
        dtype=dtype,
        noise_type=config.rolling_noise_type,
        rho=config.rolling_noise_rho,
    )

    X_noisy = _forward_noise(X_clean, tau_vec, eps)

    obs_emb = encoder(obs_t, None)

    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

    # keep scalar t for compatibility; true slot-wise conditioning is in tau_vec
    t_scalar = torch.full(
        (bs,),
        float(tau_vec.mean().detach().cpu()),
        device=device,
        dtype=dtype,
    )

    X_pred = flow_map.get_velocity(
        t_scalar,
        X_noisy,
        obs_emb,
        slot_noise_levels=tau_vec.expand(bs, -1),
    )

    loss_clean = torch.mean(get_norm(X_pred - X_clean, config.norm_type))

    aux = {
        "loss_clean": float(loss_clean.detach().cpu()),
        "T_window": T,
    }
    return config.loss_scale * loss_clean, aux



def rolling_policy_v1_loss032202(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
):
    """
    Rolling Policy v2:
    fixed noise ladder + true rolling update supervision

    Expected:
        act.shape[1] = T + 1
        obs length >= obs_steps + 1
    """
    assert hasattr(config, "obs_steps"), "config.obs_steps missing; sync task fields into optimization config first"

    assert act.ndim == 3, f"act must be (B, T+1, D), got {act.shape}"
    bs, T_plus_1, D = act.shape
    device = act.device
    dtype = act.dtype

    T = T_plus_1 - 1
    assert T >= 1, f"T must be >=1, got {T}"

    # teacher clean windows
    X1_clean = act[:, 0:T, :]      # (B, T, D)
    X2_clean = act[:, 1:T+1, :]    # (B, T, D)

    # obs windows
    To = config.obs_steps

    def slice_obs(obs_in, start, length):
        if isinstance(obs_in, dict):
            return {k: v[:, start:start+length, ...] for k, v in obs_in.items()}
        else:
            return obs_in[:, start:start+length, ...]

    obs_t = slice_obs(obs, 0, To)
    obs_t1 = slice_obs(obs, 1, To)

    # fixed rolling ladder
    tau_vec = _build_tau_ladder(
        T=T,
        device=device,
        dtype=dtype,
        tau_min=getattr(config, "rolling_tau_min", 0.03),
        tau_max=getattr(config, "rolling_tau_max", 1.0),
        mode=getattr(config, "rolling_tau_mode", "linear"),
        beta=getattr(config, "rolling_tau_beta", 2.0),
    )  # (1, T)

    # ----------------------------------
    # Step 1: teacher clean -> noisy ladder window
    # ----------------------------------
    eps1 = _sample_noise(
        bs=bs,
        T=T,
        D=D,
        device=device,
        dtype=dtype,
        noise_type=getattr(config, "rolling_noise_type", "gaussian"),
        rho=getattr(config, "rolling_noise_rho", 0.9),
    )
    X1_noisy = _forward_noise(X1_clean, tau_vec, eps1)

    obs_emb_t = encoder(obs_t, None)

    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb_t = obs_emb_t * keep_mask.view(-1, *([1] * (obs_emb_t.ndim - 1)))

    t_scalar = torch.full(
        (bs,),
        float(tau_vec.mean().detach().cpu()),
        device=device,
        dtype=dtype,
    )

    X1_pred = flow_map.get_velocity(
        t_scalar,
        X1_noisy,
        obs_emb_t,
        slot_noise_levels=tau_vec.expand(bs, -1),
    )  # (B, T, D)

    # ----------------------------------
    # Rolling update: predicted clean -> next noisy window
    # ----------------------------------
    X2_noisy_pred = torch.zeros_like(X1_pred)

    if T > 1:
        tau_next = tau_vec[:, :-1]  # (1, T-1)
        eps2 = _sample_noise(
            bs=bs,
            T=T - 1,
            D=D,
            device=device,
            dtype=dtype,
            noise_type=getattr(config, "rolling_noise_type", "gaussian"),
            rho=getattr(config, "rolling_noise_rho", 0.9),
        )
        X2_noisy_pred[:, :-1, :] = _forward_noise(
            X1_pred[:, 1:, :],
            tau_next,
            eps2,
        )

    # tail fresh high-noise sample
    eps_tail = torch.randn(bs, 1, D, device=device, dtype=dtype)
    tau_tail = tau_vec[:, -1:].unsqueeze(-1)  # (1, 1, 1)
    X2_noisy_pred[:, -1:, :] = tau_tail * eps_tail

    # ----------------------------------
    # Step 2: predict next clean window
    # ----------------------------------
    obs_emb_t1 = encoder(obs_t1, None)

    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb_t1 = obs_emb_t1 * keep_mask.view(-1, *([1] * (obs_emb_t1.ndim - 1)))

    X2_pred = flow_map.get_velocity(
        t_scalar,
        X2_noisy_pred,
        obs_emb_t1,
        slot_noise_levels=tau_vec.expand(bs, -1),
    )  # (B, T, D)

    # ----------------------------------
    # Losses
    # ----------------------------------
    loss_step1 = torch.mean(get_norm(X1_pred - X1_clean, config.norm_type))
    loss_step2 = torch.mean(get_norm(X2_pred - X2_clean, config.norm_type))

    rollout_beta = getattr(config, "rolling_rollout_beta", 1.0)
    loss = loss_step1 + rollout_beta * loss_step2

    aux = {
        "loss_step1": float(loss_step1.detach().cpu()),
        "loss_step2": float(loss_step2.detach().cpu()),
        "T_window": T,
    }
    return config.loss_scale * loss, aux




def rolling_policy_v1_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
):
    """
    Rolling Policy (Local-Step)
    Train the model to predict one denoising level down:
        x^{tau_i} -> x^{tau_{i-1}}
    with tau_0 = 0 (clean).

    Expected:
        act.shape = (B, T, D)
    """
    assert hasattr(config, "obs_steps"), "config.obs_steps missing; sync task fields into optimization config first"

    assert act.ndim == 3, f"act must be (B, T, D), got {act.shape}"
    bs, T, D = act.shape
    device = act.device
    dtype = act.dtype

    X_clean = act  # teacher clean window

    # obs window
    To = config.obs_steps
    if isinstance(obs, dict):
        obs_t = {k: v[:, :To, ...] for k, v in obs.items()}
    else:
        obs_t = obs[:, :To, ...]

    # build ladder tau_1 ... tau_T
    tau_vec = _build_tau_ladder(
        T=T,
        device=device,
        dtype=dtype,
        tau_min=getattr(config, "rolling_tau_min", 1e-3),
        tau_max=getattr(config, "rolling_tau_max", 0.05),
        mode=getattr(config, "rolling_tau_mode", "linear"),
        beta=getattr(config, "rolling_tau_beta", 2.0),
    )  # (1, T)

    # sample one set of base noise for teacher corruption
    eps = _sample_noise(
        bs=bs,
        T=T,
        D=D,
        device=device,
        dtype=dtype,
        noise_type=getattr(config, "rolling_noise_type", "gaussian"),
        rho=getattr(config, "rolling_noise_rho", 0.9),
    )

    # input noisy window: tau_1 ... tau_T
    X_in = _forward_noise(X_clean, tau_vec, eps)

    # target window: tau_0 ... tau_{T-1}
    tau_prev = torch.zeros_like(tau_vec)
    if T > 1:
        tau_prev[:, 1:] = tau_vec[:, :-1]
    # tau_prev[:,0] stays 0 => clean target for first slot

    X_target = _forward_noise(X_clean, tau_prev, eps)

    # encode obs
    obs_emb = encoder(obs_t, None)

    # optional cond dropout
    p_uncond = getattr(config, "p_uncond", 0.0)
    if p_uncond > 0.0:
        keep_mask = torch.bernoulli(
            torch.full((bs,), 1 - p_uncond, device=device, dtype=dtype)
        )
        obs_emb = obs_emb * keep_mask.view(-1, *([1] * (obs_emb.ndim - 1)))

    # keep scalar t only for compatibility
    t_scalar = torch.full(
        (bs,),
        float(tau_vec.mean().detach().cpu()),
        device=device,
        dtype=dtype,
    )

    X_pred = flow_map.get_velocity(
        t_scalar,
        X_in,
        obs_emb,
        slot_noise_levels=tau_vec.expand(bs, -1),
    )

    loss_step = torch.mean(get_norm(X_pred - X_target, config.norm_type))

    # optional small clean-first-slot auxiliary
    first_slot_beta = getattr(config, "rolling_first_slot_beta", 0.0)
    if first_slot_beta > 0.0:
        loss_first = torch.mean(get_norm(X_pred[:, 0, :] - X_clean[:, 0, :], config.norm_type))
        loss = loss_step + first_slot_beta * loss_first
    else:
        loss_first = torch.tensor(0.0, device=device, dtype=dtype)
        loss = loss_step

    aux = {
        "loss_step": float(loss_step.detach().cpu()),
        "loss_first": float(loss_first.detach().cpu()),
        "T_window": T,
    }
    return config.loss_scale * loss, aux
