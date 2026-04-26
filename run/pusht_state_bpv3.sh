export HF_HOME=/2024233219/dataset/huggingface_cache


##** BPv3
CUDA_VISIBLE_DEVICES=2 uv run examples/train_pusht.py \
    task=pusht_state \
    log.wandb_mode=online \
    optimization.seed=0 \
    optimization.loss_type=bridge_v3 \
    optimization.use_consistency_loss=true \
    optimization.consist_beta=1.0 \
    optimization.auto_resume=False \
    network=chiunet \
    task.horizon=17 \
    task.act_steps=1 \
    log.save_video=True
    # optimization.use_consistency_loss=true \
    # optimization.gradient_steps=20000
