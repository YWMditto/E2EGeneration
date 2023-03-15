


# python /data/lipsync/xgyang/E2EGeneration/V1/model_2/evaluate.py \
#     --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_2/hubert/neural_v1/epoch=20-step=9200-validate_loss=0.0618.ckpt \
#     --train_config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_2/hubert/neural_v1.yaml \
#     --eval_name_manifest_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/neural/data_validate.txt \
#     --eval_audio_dir /data/lipsync/xgyang/e2e_data/aligned_audios \
#     --eval_ctrl_label_dir /data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels \
#     --eval_normalize False \
#     --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/lumi_template.pt \
#     --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_2/hubert/neural_v1 \
#     --device 3



python /data/lipsync/xgyang/E2EGeneration/V1/model_2/evaluate.py \
    --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_2/ser_hubert/all_v1_large/epoch=35-step=113600-validate_loss=0.0696.ckpt \
    --train_config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_2/ser_hubert/all_v1_large.yaml \
    --eval_name_manifest_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/all/data_validate.txt \
    --eval_audio_dir /data/lipsync/xgyang/e2e_data/aligned_audios \
    --eval_ctrl_label_dir /data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels \
    --eval_normalize True \
    --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/lumi_template.pt \
    --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_2/ser_hubert/all_v1_large \
    --device 0






