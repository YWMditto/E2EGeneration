



python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
    --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/v2/epoch=93-step=75012-validate_loss=0.11.ckpt \
    --config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/v2.yaml \
    --evaluate_config /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/evaluate.yaml \
    --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_template.pt \
    --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/v2/ \
    --device 1




