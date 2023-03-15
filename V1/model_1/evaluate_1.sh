



# python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
#     --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/new_v1/epoch=84-step=67830-validate_loss=0.1399.ckpt \
#     --config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/v1.yaml \
#     --evaluate_config /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/evaluate.yaml \
#     --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_template.pt \
#     --phn_manifest_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60_validate.txt \
#     --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/new_v1/ \
#     --device 0



# python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
#     --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/new_v1/epoch=84-step=67830-validate_loss=0.1399.ckpt \
#     --config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/v1.yaml \
#     --evaluate_config /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/evaluate.yaml \
#     --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_template.pt \
#     --phn_manifest_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60_validate.txt \
#     --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/new_v1/ \
#     --device 0


# python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
#     --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/hubert_layer6/neural/v1/epoch=396-step=180000-validate_loss=0.0875.ckpt \
#     --config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/hubert_layer6/neural/v1.yaml \
#     --evaluate_config /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/hubert_layer6/neural/evaluate.yaml \
#     --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/lumi_template.pt \
#     --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/hubert_layer6/neural/v1/ \
#     --device 3


# python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
#     --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/ser_hubert/all_v1/epoch=48-step=39000-validate_loss=0.1128.ckpt \
#     --config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/ser_hubert/layer16/all_v1.yaml \
#     --evaluate_config /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/hubert_layer6/neural/evaluate.yaml \
#     --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/lumi_template.pt \
#     --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/hubert_layer6/neural/v1/ \
#     --device 3


python /data/lipsync/xgyang/E2EGeneration/V1/model_1/evaluate.py \
    --pl_ckpt_path /data/lipsync/xgyang/E2EGeneration/V1/checkpoint_dir/model_1/hubert_layer6/all_emotion_v1/epoch=16-step=13500-mouth_l1_validate_loss=0.1049.ckpt \
    --train_config_path /data/lipsync/xgyang/E2EGeneration/V1/config/model_1/hubert_layer6/all_emotion_v1.yaml \
    --eval_name_manifest_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/all/data_validate.txt \
    --eval_static_feature_dir  /data/lipsync/xgyang/e2e_data/static_feature/layer6/hubert_60\
    --eval_ctrl_label_dir /data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels \
    --lumi_template_path /data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/lumi_template.pt \
    --save_dir /data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/hubert/layer6/all_emotion_v1 \
    --device 0