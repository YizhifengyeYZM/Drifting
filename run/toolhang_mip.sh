
##** ToolHang MIP
export HF_HOME=/2024233219/dataset/huggingface_cache
CUDA_VISIBLE_DEVICES=1 uv run examples/train_robomimic.py \
    task=tool_hang_ph_state_rel \
    log.wandb_mode=online \
    optimization.loss_type=mip \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.exp_name=mip_toolhang_state_rel_2
    # optimization.gradient_steps=20000



# python examples/train_robomimic.py task=tool_hang_ph_state
