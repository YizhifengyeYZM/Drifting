export HF_HOME=/2024233219/dataset/huggingface_cache
# export HF_HUB_OFFLINE=1

## K = 1, H = 4
# uv run examples/train_pusht.py task=pusht_state log.wandb_mode=offline optimization=bridge network=chiunet optimization.sample_mode=streaming optimization.prediction_offset=1 task.horizon=5 task.act_steps=1 log.save_video=True

##** 2.12 H20 新版 BP：跑 H = 16，T_pred = 8
# uv run examples/train_pusht.py \
#     task=pusht_state \
#     log.wandb_mode=online \
#     optimization=bridge \
#     network=chiunet \
#     optimization.prediction_offset=8 \
#     task.horizon=24 \
#     task.act_steps=1 \
#     log.save_video=True \
#    optimization.gradient_steps=20000

##** 17.29: 跑 H = 16，T_pred = 4
# CUDA_VISIBLE_DEVICES=1 uv run examples/train_pusht.py \
#     task=pusht_state \
#     log.wandb_mode=online \
#     optimization=bridge \
#     network=chiunet \
#     optimization.prediction_offset=16 \
#     task.horizon=32 \
#     task.act_steps=1 \
#     log.save_video=True
#     # optimization.gradient_steps=20000


##** 2.14，全新版本 BridgePolicy，H = 16，非 CFG 模式
CUDA_VISIBLE_DEVICES=0 uv run examples/train_pusht.py \
    task=pusht_state \
    log.wandb_mode=online \
    optimization.seed=0 \
    optimization.loss_type=rp_v1 \
    optimization.p_uncond=0.0 \
    optimization.guidance_scale=1.0 \
    optimization.auto_resume=False \
    network=sudeepdit \
    task.horizon=16 \
    task.act_steps=1 \
    log.save_video=True
    # optimization.gradient_steps=20000
