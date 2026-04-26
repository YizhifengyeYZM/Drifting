export HF_HOME=/2024233219/dataset/huggingface_cache

# ##** PushT State / Image
# CUDA_VISIBLE_DEVICES=0 uv run examples/train_pusht.py \
#     task=pusht_image \
#     log.wandb_mode=online \
#     optimization.loss_type=drifting \
#     optimization.auto_resume=False \
#     network=sudeepdit \
#     log.save_video=True \
#     log.exp_name=drift_pusht_image
#     # optimization.gradient_steps=20000


##** ToolHang State / Image
CUDA_VISIBLE_DEVICES=1 uv run examples/train_robomimic.py \
    task=tool_hang_ph_state_rel \
    log.wandb_mode=online \
    optimization.loss_type=drifting \
    optimization.auto_resume=False \
    network=sudeepdit \
    optimization.t_two_step=0.95 \
    log.exp_name=drift_toolhang_state_t_0.95
