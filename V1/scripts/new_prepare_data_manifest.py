


from pathlib import Path
import random


from utils import _process_bar






def prepare_name_manifest(*feature_dirs, save_path, name_filter_fn=None):


    name_set = None
    for feature_dir in feature_dirs:

        cur_name_set = set()
        feature_dir = Path(feature_dir)
        for path in feature_dir.iterdir():
            name = path.stem
            cur_name_set.add(name)

        if name_set is None:
            name_set = cur_name_set
        else:
            name_set = name_set.intersection(cur_name_set)
        
    if name_filter_fn is not None:
        filtered_name_set = []
        for name in name_set:
            if name_filter_fn(name):
                filtered_name_set.append(name)
        name_set = filtered_name_set
    

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_f = open(save_path, "w")

    for name in name_set:
        print(name, file=save_f)

    save_f.close()



def split_validate(manifest_path, validate_p):

    all_lines = []
    with open(manifest_path, "r") as f:
        for line in f:
            all_lines.append(line.rstrip())

    if isinstance(validate_p, float):
        validate_p = int(len(all_lines) * validate_p)
    
    manifest_path = Path(manifest_path)
    train_f = open(manifest_path.parent.joinpath(f"{manifest_path.stem}_train.txt"), "w")
    validate_f = open(manifest_path.parent.joinpath(f"{manifest_path.stem}_validate.txt"), "w")

    validate_indices = set(random.sample(range(len(all_lines)), validate_p))

    for idx, line in enumerate(all_lines):

        if idx in validate_indices:
            print(line, file=validate_f)
        else:
            print(line, file=train_f)

    train_f.close()
    validate_f.close()


def filter_fn_neural(name):
    if name.endswith("neutral"):
        return True




if __name__ == "__main__":

    # prepare_name_manifest(
    #     "/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
    #     "/data/lipsync/xgyang/e2e_data/static_feature/hubert_60",
    #     "/data/lipsync/xgyang/e2e_data/aligned_phonemes",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/all/data.txt",
    # )

    # split_validate(
    #     manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/all/data.txt",
    #     validate_p=30
    # )
    

    


    # prepare_name_manifest(
    #     "/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
    #     "/data/lipsync/xgyang/e2e_data/static_feature/hubert_60",
    #     "/data/lipsync/xgyang/e2e_data/aligned_phonemes",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/neural/data.txt",
    #     name_filter_fn=filter_fn_neural
    # )

    split_validate(
        manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/hubert/neural/data.txt",
        validate_p=30
    )






