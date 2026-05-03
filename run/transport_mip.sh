
##** ToolHang MIP
export HF_HOME=/2024233219/dataset/huggingface_cache
export PYTHONPATH=/2024233219/code/drifting:${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 /2024233219/code/maan/.venv/bin/python examples/train_robomimic.py \
    task=transport_ph_image \
    log.wandb_mode=online \
    optimization.loss_type=mip \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.eval_episodes=4 \
    log.save_video=False \
    log.exp_name=test_transport_image_ph_mip
    # optimization.gradient_steps=20000



# python examples/train_robomimic.py task=tool_hang_ph_state
