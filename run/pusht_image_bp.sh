export HF_HOME=/2024233219/dataset/huggingface_cache
# export HF_HUB_OFFLINE=1

##** 2.14，全新版本 BridgePolicy，H = 16，非 CFG 模式
CUDA_VISIBLE_DEVICES=0 uv run examples/train_pusht.py \
    task=pusht_image \
    log.wandb_mode=online \
    optimization.seed=0 \
    optimization.loss_type=bridge_v2 \
    optimization.p_uncond=0.0 \
    optimization.guidance_scale=1.0 \
    optimization.auto_resume=False \
    network=chiunet \
    task.horizon=17 \
    task.act_steps=1 \
    log.save_video=True
    # optimization.gradient_steps=20000
