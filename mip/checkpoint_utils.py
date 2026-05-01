"""Checkpoint naming and teacher resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import loguru


GLOBAL_CHECKPOINTS_DIR = Path("checkpoints")


def _cfg_value(obj, name: str, default=None):
    return getattr(obj, name, default)


def legacy_task_id(config) -> str:
    task = config.task
    return f"{task.env_name}_{task.env_type}_{task.obs_type}"


def preferred_task_id(config) -> str:
    override = _cfg_value(config.optimization, "checkpoint_task_id", None)
    if override not in [None, "", "None"]:
        return str(override)

    task_id = legacy_task_id(config)
    action_type = _cfg_value(config.task, "action_type", None)
    if action_type not in [None, "", "None"]:
        task_id = f"{task_id}_{action_type}"
    return task_id


def _loss_suffix(config, loss_type: str) -> str:
    if loss_type == "bridge":
        k = config.optimization.prediction_offset
        h = config.task.horizon - k
        return f"_H{h}_K{k}"
    if loss_type == "bridge_v2":
        return "_BPv2"
    if loss_type == "bridge_v3":
        return "_BPv3"
    if loss_type == "prcp_v1":
        return "_PRCPv1"
    if loss_type == "prcp_v2":
        return "_PRCPv2"
    if loss_type == "rp_v1":
        return "_RPv1"
    return ""


def build_checkpoint_base_name(
    config,
    *,
    loss_type: str | None = None,
    seed: int | None = None,
    task_id: str | None = None,
) -> str:
    loss_type = loss_type or config.optimization.loss_type
    seed = config.optimization.seed if seed is None else seed
    task_id = task_id or preferred_task_id(config)
    base = (
        f"{task_id}_{loss_type}_{config.network.network_type}_"
        f"{config.network.emb_dim}_seed{seed}"
    )
    return base + _loss_suffix(config, loss_type)


def candidate_checkpoint_base_names(
    config,
    *,
    loss_type: str | None = None,
    seed: int | None = None,
) -> list[str]:
    primary = build_checkpoint_base_name(config, loss_type=loss_type, seed=seed)
    legacy = build_checkpoint_base_name(
        config,
        loss_type=loss_type,
        seed=seed,
        task_id=legacy_task_id(config),
    )
    if legacy == primary:
        return [primary]
    return [primary, legacy]


def find_latest_checkpoint_path(checkpoint_base_name: str) -> Path | None:
    matching_checkpoints = list(
        GLOBAL_CHECKPOINTS_DIR.glob(f"{checkpoint_base_name}_success*.pt")
    )
    if not matching_checkpoints:
        loguru.logger.info(
            f"No existing checkpoints found for pattern: {checkpoint_base_name}"
        )
        return None

    best_checkpoint = None
    best_success_rate = -1
    for checkpoint_path in matching_checkpoints:
        try:
            success_str = checkpoint_path.stem.split("_success")[-1]
            success_rate = int(success_str)
        except (ValueError, IndexError) as e:
            loguru.logger.warning(
                f"Could not parse success rate from {checkpoint_path.name}: {e}"
            )
            continue
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_checkpoint = checkpoint_path

    if best_checkpoint is not None:
        loguru.logger.info(
            f"Found checkpoint: {best_checkpoint.name} "
            f"with success rate: {best_success_rate}%"
        )
    return best_checkpoint


def find_first_checkpoint(
    logger,
    base_names: Iterable[str],
) -> tuple[Path | None, str | None]:
    for base_name in base_names:
        if logger is None:
            checkpoint_path = find_latest_checkpoint_path(base_name)
        else:
            checkpoint_path = logger.find_latest_checkpoint(base_name)
        if checkpoint_path is not None:
            return Path(checkpoint_path), base_name
    return None, None


def resolve_cp_teacher_checkpoint(config, logger) -> Path | None:
    opt = config.optimization
    explicit_path = _cfg_value(opt, "cp_teacher_path", None)
    if explicit_path not in [None, "", "None"]:
        path = Path(str(explicit_path))
        if path.exists():
            loguru.logger.info(f"Using explicit CP teacher checkpoint: {path}")
            return path
        msg = f"Explicit CP teacher checkpoint does not exist: {path}"
        if bool(_cfg_value(opt, "cp_teacher_required", True)):
            raise FileNotFoundError(msg)
        loguru.logger.warning(msg)
        return None

    if not bool(_cfg_value(opt, "cp_teacher_auto_find", True)):
        if bool(_cfg_value(opt, "cp_teacher_required", True)):
            raise FileNotFoundError(
                "CP teacher path is unset and cp_teacher_auto_find is false."
            )
        return None

    teacher_loss_type = str(_cfg_value(opt, "cp_teacher_loss_type", "edm"))
    teacher_seed = int(_cfg_value(opt, "cp_teacher_seed", 0))
    base_names = candidate_checkpoint_base_names(
        config,
        loss_type=teacher_loss_type,
        seed=teacher_seed,
    )
    checkpoint_path, matched_base = find_first_checkpoint(logger, base_names)
    if checkpoint_path is not None:
        loguru.logger.info(
            f"Using CP teacher checkpoint: {checkpoint_path} "
            f"(matched base: {matched_base})"
        )
        return checkpoint_path

    expected = ", ".join(base_names)
    msg = (
        "No CP teacher checkpoint found. Expected one of these base names in "
        f"checkpoints/: {expected}"
    )
    missing_policy = str(_cfg_value(opt, "cp_teacher_missing", "error"))
    if bool(_cfg_value(opt, "cp_teacher_required", True)) and missing_policy == "error":
        raise FileNotFoundError(msg)
    loguru.logger.warning(msg)
    return None
