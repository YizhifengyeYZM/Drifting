export HF_HOME=/2024233219/dataset/huggingface_cache


##** ToolHang State / Image
CUDA_VISIBLE_DEVICES=0 uv run examples/train_robomimic.py \
    task=tool_hang_ph_state_rel \
    log.wandb_mode=online \
    optimization.loss_type=drift6matrix \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.exp_name=drift6matrix_toolhang_state
