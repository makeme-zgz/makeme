import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('machine', type=str)
args = parser.parse_args()


with open('sh/dir.json') as f:
    d = json.load(f)
d = d[args.machine]

save_dir = "/home/supergeorge/makeme/data/facescape/train"

cmd = f"""{d['train_environ']}
    python train.py
    --num_workers {d['num_workers']}
    --data_root {d['facescape_dir']}
    --dataset_name facescape
    --model_name model_cas
    --num_src 3
    --cas_depth_num 64,32,16
    --cas_interv_scale 4,2,1
    --resize 640,512
    --crop 640,512
    --mode soft
    --num_samples 160000
    --batch_size {d['batch_size']}
    --job_name temp
    --save_dir {save_dir}
"""

cmd = ' '.join(cmd.strip().split())
print(cmd)
os.system(cmd)