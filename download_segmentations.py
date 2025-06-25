import shutil
import os
import pandas as pd
import sys

from huggingface_hub import hf_hub_download
from tqdm import tqdm


split = 'train_fixed'
batch_size = 100
start_at = 0

repo_id = 'ibrahimhamamci/CT-RATE'
directory_name = f'dataset/{split}/'
hf_token = os.getenv("HF_TOKEN")

data = pd.read_csv(f'{split.split("_")[0]}_labels.csv')  # changed from v1, take this file from https://github.com/sezginerr/example_download_script

for i in tqdm(range(start_at, len(data), batch_size)):

    data_batched = data[i:i+batch_size]

    for name in data_batched['VolumeName']:
        folder1 = name.split('_')[0]
        folder2 = name.split('_')[1]
        folder = folder1 + '_' + folder2
        folder3 = name.split('_')[2]
        subfolder = folder + '_' + folder3
        subfolder = directory_name + folder + '/' + subfolder
        ll = subfolder.split('/')  # added
        seg_folder = os.path.join(ll[0], 'ts_seg', 'ts_total', *ll[1:])  # added

        try:
            hf_hub_download(repo_id=repo_id,
                repo_type='dataset',
                token=hf_token,
                subfolder=seg_folder,
                filename=name,
                local_dir='<PATH_TO_CT-RATE>',  # should be <some_path>/CT-RATE
                resume_download=True,
            )
        except Exception as e:
            # append to file called errors.txt
            with open('errors.txt', 'a') as f:
                f.write(f"{e} :: {name=} {seg_folder=}\n")
            # print to stderr
            print(f"{e} :: {name=} {seg_folder=}", file=sys.stderr)
    try:
        # shutil.rmtree('<path_to_ct_rate>/.cache/huggingface/download/dataset/')  # need to delete the cache dir
        continue
    except Exception as e:
        print(f"Error removing cache directory: {e}", file=sys.stderr)
