export HF_HOME=/2024233219/dataset/huggingface_cache


##** PushT State / Image
CUDA_VISIBLE_DEVICES=1 uv run examples/train_pusht.py \
    task=pusht_state \
    log.wandb_mode=online \
    optimization.loss_type=drift6min_rebal \
    optimization.auto_resume=False \
    network=sudeepdit \
    log.exp_name=drift6min_rebal_pusht_state
