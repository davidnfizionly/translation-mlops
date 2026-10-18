"""Download the English-Nuer-Dinka parallel corpus from Hugging Face.

The dataset is gated: accept its conditions on the dataset page with a
Hugging Face account and store a read token locally before running this.
Files land in data/raw/ (git-ignored). The dataset is never committed.

Dataset: English-Nuer-Dinka Parallel Corpus, by Tajnaam, CC BY-NC 4.0.
https://huggingface.co/datasets/dayomtechnologies/english-nuer-dinka-parallel-corpus
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "dayomtechnologies/english-nuer-dinka-parallel-corpus"
# Pinned revision so every run sees the same snapshot.
REVISION = "28db28e6b554975aad147c85655ddd82a4cb4ba4"
FILES = ["README.md", "nuer_dinka_translations_paired.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/raw", help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "files": {}}
    for name in FILES:
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=name,
                revision=REVISION,
                local_dir=out,
            )
        )
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"{name}: {path.stat().st_size} bytes")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written to {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
