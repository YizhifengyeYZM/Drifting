export HF_HOME=/2024233219/dataset/huggingface_cache


##** PushT State / Image
CUDA_VISIBLE_DEVICES=0 uv run examples/train_pusht.py \
    task=pusht_state \
    log.wandb_mode=online \
    optimization.loss_type=geofuse_noise \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.exp_name=geofuse_noise_pusht_state


# ##** ToolHang State / Image
# CUDA_VISIBLE_DEVICES=1 uv run examples/train_robomimic.py \
#     task=tool_hang_ph_state_rel \
#     log.wandb_mode=online \
#     optimization.loss_type=drifting2 \
#     optimization.auto_resume=False \
#     network=sudeepdit \
#     log.exp_name=drift2_toolhang_state
