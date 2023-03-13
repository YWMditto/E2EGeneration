



python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
    --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/test_overfit_val/epoch=1-step=48000-validate_loss=0.0088.ckpt \
    --config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/test_overfit.yaml \
    --evaluate_config /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/evaluate.yaml \
    --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/lumi_template.pt \
    --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/test_overfit_val/ \
    --device 4


    # --phn_manifest_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60_validate.txt \


