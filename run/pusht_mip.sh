export HF_HOME=/2024233219/dataset/huggingface_cache

##** PushT State MIP
CUDA_VISIBLE_DEVICES=0 uv run examples/train_pusht.py \
    task=pusht_state \
    log.wandb_mode=online \
    optimization.loss_type=mip \
    optimization.auto_resume=False \
    network=chitransformer \
    log.exp_name=mip_pusht_state_one_step
    # optimization.gradient_steps=20000


# ##** ToolHang MIP
# export HF_HOME=/2024233219/dataset/huggingface_cache
# CUDA_VISIBLE_DEVICES=1 uv run examples/train_robomimic.py \
#     task=tool_hang_ph_state_abs \
#     log.wandb_mode=online \
#     optimization.loss_type=mip \
#     optimization.auto_resume=False \
#     network=chiunet \
#     log.save_video=True \
#     task.save_video=True
