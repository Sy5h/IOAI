"""Download pinned official datasets and permitted BGE into a chosen work folder.

Released test labels and Potato judge embeddings go into evaluation-assets,
separate from solver inputs. Model/answer/data files must not be committed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

POTATO_REVISION='122716fc56a212e58b1756158bdeb4c2ea2da5e6'
GHOST_REVISION='6dcd17d529f154935cda550c6e80f9801887a316'
MODEL_REVISION='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--only',choices=['all','potato','ghost','model'],default='all')
    args=parser.parse_args()
    root=Path(args.data_root).resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(root/'hf-cache')
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import hf_hub_download
    jobs=[
        ('potato','IOAI-official/ioai-2026-potato','dataset',POTATO_REVISION,'potato-source',
         ['public/vocabulary.json','public/public_embeddings.npy','public/test_public.json']),
        ('potato','IOAI-official/ioai-2026-potato','dataset',POTATO_REVISION,'evaluation-assets/potato',
         ['private/vocabulary.json','private/private_embeddings.npy','private/secrets_leaderboard_a.json','private/secrets_leaderboard_b.json']),
        ('ghost','IOAI-official/ioai-2026-ghost-of-the-machine','dataset',GHOST_REVISION,'ghost-source',
         ['public/train/data.jsonl','public/train_answers.jsonl','public/pretrain/data.jsonl','public/pretrain_answers.jsonl',
          'public/pretest/data.jsonl']),
        ('ghost','IOAI-official/ioai-2026-ghost-of-the-machine','dataset',GHOST_REVISION,'evaluation-assets/ghost',
         ['public/pretest_answers.jsonl','private/test_leaderboard_a/data.jsonl','private/test_leaderboard_a_answers.jsonl',
          'private/test_leaderboard_b/data.jsonl','private/test_leaderboard_b_answers.jsonl']),
        ('model','BAAI/bge-base-en-v1.5','model',MODEL_REVISION,'models/bge-base-en-v1.5',
         ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']),
    ]
    records=[]
    for category,repo,kind,revision,folder,paths in jobs:
        if args.only not in ['all',category]:
            continue
        for remote in paths:
            started=time.monotonic()
            local=Path(hf_hub_download(repo,remote,repo_type=kind,revision=revision,local_dir=root/folder))
            record={'repo':repo,'revision':revision,'remote_path':remote,'local_path':local.relative_to(root).as_posix(),
                    'bytes':local.stat().st_size,'sha256':hashlib.sha256(local.read_bytes()).hexdigest(),
                    'download_seconds':time.monotonic()-started}
            records.append(record);print(json.dumps(record),flush=True)
            (root/('download_manifest_'+args.only+'.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
