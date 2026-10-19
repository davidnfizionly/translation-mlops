"""Build build/train-sourcedir.tar.gz (train.py and the serving code) reproducibly.

Fixed modification times and ordering, so the archive and its MD5 only change
when the code changes. Uploaded by Terraform (infra/ml) for the pipeline.
"""

import gzip
import io
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = ["train.py", "serving/inference.py"]


def main():
    out = REPO / "build" / "train-sourcedir.tar.gz"
    out.parent.mkdir(exist_ok=True)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in FILES:
            data = (REPO / "training" / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            tar.addfile(info, io.BytesIO(data))
    with gzip.GzipFile(filename="", mode="wb", fileobj=out.open("wb"), mtime=0) as gz:
        gz.write(raw.getvalue())
    print(out)


if __name__ == "__main__":
    main()
