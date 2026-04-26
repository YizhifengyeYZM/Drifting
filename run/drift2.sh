export HF_HOME=/2024233219/dataset/huggingface_cache


##** ToolHang State / Image
CUDA_VISIBLE_DEVICES=1 uv run examples/train_robomimic.py \
    task=tool_hang_ph_state_rel \
    log.wandb_mode=online \
    optimization.loss_type=drifting2 \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.exp_name=drift2_toolhang_state

# ##** PushT State / Image
# CUDA_VISIBLE_DEVICES=0 uv run examples/train_pusht.py \
#     task=pusht_image \
#     log.wandb_mode=online \
#     optimization.loss_type=drifting2 \
#     optimization.auto_resume=False \
#     network=sudeepdit \
#     log.exp_name=drift2_pusht_image
#     # optimization.gradient_steps=20000
