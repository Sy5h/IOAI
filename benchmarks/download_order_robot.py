"""Fetch public IOAI data/model files into this benchmark workspace only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path('work/bench-order-robot').resolve()
REVISIONS = {
    'IOAI-official/ioai-2026-robot-chasing': '8a82acbdfda387269649d4bcf580fbdab1ed3cc6',
    'IOAI-official/ioai-2026-find-the-order': '71a35b5a566665e242416eba45f018dc378663b8',
    'openai/whisper-small': '973afd24965f72e36ca33b3055d56a652f456b4d',
    'Qwen/Qwen2.5-0.5B': '060db6499f32faf8b98477b0a26969ef7d8b9987',
}


def download(repo, kind, allow, destination):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=False)
    info = api.repo_info(repo_id=repo, repo_type=kind, revision=REVISIONS[repo], files_metadata=True)
    chosen = [s for s in info.siblings if allow(s.rfilename)]
    print(json.dumps({'repo': repo, 'revision': info.sha, 'files': len(chosen),
                      'bytes': sum(s.size or 0 for s in chosen)}), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    # Pre-create shared parents serially. Concurrent recursive mkdir can fail
    # under some Windows filesystem sandboxes even when exist_ok is true.
    for entry in chosen:
        relative = Path(entry.rfilename)
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        (destination / '.cache' / 'huggingface' / 'download' / relative).parent.mkdir(parents=True, exist_ok=True)
    manifest = {'repo': repo, 'type': kind, 'revision': info.sha, 'files': []}
    def fetch(entry):
        path = hf_hub_download(repo_id=repo, repo_type=kind, filename=entry.rfilename,
                               revision=info.sha, token=False, local_dir=destination,
                               cache_dir=ROOT / 'hf-cache' / 'hub')
        return {'name': entry.rfilename, 'size': Path(path).stat().st_size}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch, entry) for entry in chosen]
        for index, future in enumerate(as_completed(futures)):
            manifest['files'].append(future.result())
            if index % 20 == 0 or index + 1 == len(chosen):
                print(f'{repo}: {index+1}/{len(chosen)}', flush=True)
    manifest['files'].sort(key=lambda item: item['name'])
    (destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def layout_robot(source):
    destination = ROOT / 'robot' / 'official'
    truth = ROOT / 'scorer-only' / 'robot'
    truth.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'pretrain', 'pretest'):
        target = destination / ('test_public' if split == 'pretest' else split)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'public' / split / 'observations.json', target / 'observations.json')
        answer = source / 'public' / f'{split}_answers.json'
        if split == 'pretest':
            shutil.copyfile(answer, truth / 'pretest_answers.json')
        else:
            shutil.copyfile(answer, target / 'labels.json')
    for split in ('test_leaderboard_a', 'test_leaderboard_b'):
        origin = source / 'private' / split / 'observations.json'
        if not origin.exists():
            continue
        target = ROOT / 'robot' / split / 'dataset'
        (target / 'train').mkdir(parents=True, exist_ok=True)
        (target / 'test_public').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'public' / 'train' / 'observations.json', target / 'train' / 'observations.json')
        shutil.copyfile(source / 'public' / 'train_answers.json', target / 'train' / 'labels.json')
        shutil.copyfile(origin, target / 'test_public' / 'observations.json')
        shutil.copyfile(source / 'private' / f'{split}_answers.json', truth / f'{split}_answers.json')


def layout_order(source):
    destination = ROOT / 'order' / 'official'
    truth = ROOT / 'scorer-only' / 'order'
    truth.mkdir(parents=True, exist_ok=True)
    for split in ('pretrain', 'pretest'):
        target = destination / ('test_public' if split == 'pretest' else 'train')
        shutil.copytree(source / 'public' / split, target, dirs_exist_ok=True)
        answer = source / 'public' / f'{split}_answers.json'
        if split == 'pretest':
            shutil.copyfile(answer, truth / 'pretest_answers.json')
        else:
            shutil.copyfile(answer, target / 'answers.json')
    # Development uses labeled pretrain only. Keep its labels outside the
    # standalone solver's test directory, just as for final evaluation.
    development = ROOT / 'order' / 'development' / 'dataset' / 'test_public'
    shutil.copytree(source / 'public' / 'pretrain', development, dirs_exist_ok=True)
    shutil.copyfile(source / 'public' / 'pretrain_answers.json', truth / 'pretrain_answers.json')
    for split in ('test_leaderboard_a', 'test_leaderboard_b'):
        origin = source / 'private' / split
        if not origin.exists():
            continue
        target = ROOT / 'order' / split / 'dataset'
        shutil.copytree(origin, target / 'test_public', dirs_exist_ok=True)
        shutil.copyfile(source / 'private' / f'{split}_answers.json', truth / f'{split}_answers.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['robot', 'order-data', 'order-eval', 'models'])
    parser.add_argument('--workspace', default='work/bench-order-robot')
    args = parser.parse_args()
    ROOT = Path(args.workspace).resolve()
    os.environ['HF_HOME'] = str(ROOT / 'hf-cache')
    os.environ['HF_HUB_CACHE'] = str(ROOT / 'hf-cache' / 'hub')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    if args.part == 'robot':
        destination = ROOT / 'downloads' / 'robot'
        download('IOAI-official/ioai-2026-robot-chasing', 'dataset',
                 lambda p: p.startswith(('public/', 'private/')) or p == 'README.md', destination)
        layout_robot(destination)
    elif args.part == 'order-data':
        destination = ROOT / 'downloads' / 'order'
        download('IOAI-official/ioai-2026-find-the-order', 'dataset',
                 lambda p: p.startswith(('public/pretrain/', 'public/pretest/')) or
                 p in ('public/pretrain_answers.json', 'public/pretest_answers.json', 'README.md'), destination)
        layout_order(destination)
    elif args.part == 'order-eval':
        destination = ROOT / 'downloads' / 'order'
        download('IOAI-official/ioai-2026-find-the-order', 'dataset',
                 lambda p: p.startswith('private/') or p == 'README.md', destination)
        layout_order(destination)
    else:
        allow = lambda p: '/' not in p and (p.endswith(('.json', '.txt')) or p == 'model.safetensors' or p == 'LICENSE')
        download('openai/whisper-small', 'model', allow, ROOT / 'models' / 'whisper-small')
        download('Qwen/Qwen2.5-0.5B', 'model', allow, ROOT / 'models' / 'qwen2.5-0.5b')
