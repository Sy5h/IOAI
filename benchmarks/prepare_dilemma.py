"""Fetch the exact public assets used by benchmark_dilemma.py.

The manifest contains URLs and SHA-256 hashes only, never images or weights.
Run from the repository root; downloads go to work/bench-vision by default.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="work/bench-vision")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("dilemma_assets.json").read_text(encoding="utf-8"))
    start = time.monotonic()

    def fetch(asset):
        target = (root / asset["destination"]).resolve()
        target.relative_to(root)
        if target.is_file() and digest(target) == asset["sha256"]:
            return
        if args.verify_only:
            raise RuntimeError(f"Missing or mismatched asset: {asset['destination']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "IOAI-public-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if temporary.stat().st_size != asset["bytes"] or digest(temporary) != asset["sha256"]:
            raise RuntimeError(f"Downloaded asset hash mismatch: {asset['destination']}")
        temporary.replace(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch, manifest["assets"]))
    manifest["verification_or_download_seconds"] = time.monotonic() - start
    if not args.verify_only:
        (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(manifest['assets'])} assets in {manifest['verification_or_download_seconds']:.2f}s")


if __name__ == "__main__":
    main()
