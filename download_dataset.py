import shutil
import os
import pandas as pd

from huggingface_hub import hf_hub_download
from tqdm import tqdm


split = 'train_fixed'
batch_size = 100
start_at = 0

repo_id = 'ibrahimhamamci/CT-RATE'
directory_name = f'dataset/{split}/'
hf_token = os.getenv("HF_TOKEN")

data = pd.read_csv(f'{split.split("_")[0]}_labels.csv')

for i in tqdm(range(start_at, len(data), batch_size)):

    data_batched = data[i:i+batch_size]

    for name in data_batched['VolumeName']:
        folder1 = name.split('_')[0]
        folder2 = name.split('_')[1]
        folder = folder1 + '_' + folder2
        folder3 = name.split('_')[2]
        subfolder = folder + '_' + folder3
        subfolder = directory_name + folder + '/' + subfolder

        hf_hub_download(repo_id=repo_id,
            repo_type='dataset',
            token=hf_token,
            subfolder=subfolder,
            filename=name,
            local_dir='<>PATH_TO_CT-RATE>',  # should be <some_path>/CT-RATE
            resume_download=True,
        )

    shutil.rmtree('<PATH_TO_CT-RATE>/.cache/huggingface/download/dataset/train_fixed/')  # need to delete the cache dir
