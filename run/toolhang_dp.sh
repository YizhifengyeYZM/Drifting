
##** ToolHang DP(Flow Matching 版)
export HF_HOME=/2024233219/dataset/huggingface_cache
CUDA_VISIBLE_DEVICES=0 uv run examples/train_robomimic.py \
    task=tool_hang_ph_state_rel \
    log.wandb_mode=online \
    optimization.loss_type=flow \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.exp_name=flow_toolhang_state
    # optimization.gradient_steps=20000



# python examples/train_robomimic.py task=tool_hang_ph_state
