"""
Download the INbreast dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset

Requirements:
    pip install kaggle

Credentials — add to .env in the project root:
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_api_key
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


KAGGLE_DATASET = "ramanathansp20/inbreast-dataset"
DEFAULT_DEST = Path(__file__).parent / "INbreast"
ENV_FILE = Path(__file__).parent / ".env"


def load_env_file():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def write_kaggle_json():
    """Write ~/.kaggle/kaggle.json from env vars so kaggle SDK can authenticate."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (username and key):
        return
    creds_dir = Path.home() / ".kaggle"
    creds_dir.mkdir(mode=0o700, exist_ok=True)
    creds_file = creds_dir / "kaggle.json"
    creds_file.write_text(json.dumps({"username": username, "key": key}))
    creds_file.chmod(0o600)


def setup_credentials():
    # Must happen BEFORE kaggle is imported — SDK authenticates at import time
    load_env_file()
    write_kaggle_json()

    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit(
            "No Kaggle credentials found. Add to .env in the project root:\n"
            "  KAGGLE_USERNAME=your_username\n"
            "  KAGGLE_KEY=your_api_key\n\n"
            "Get your API key at: https://www.kaggle.com/settings → API → Create New Token"
        )


def ensure_kaggle_installed():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "show", "kaggle"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("kaggle package not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


def download(dest: Path, force: bool):
    import kaggle  # import AFTER credentials are written to disk

    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {KAGGLE_DATASET} → {dest} ...")
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(dest),
        unzip=False,
        quiet=False,
        force=force,
    )

    zip_path = dest / "inbreast.zip"
    if not zip_path.exists():
        zips = list(dest.glob("*.zip"))
        if not zips:
            sys.exit("Download failed — no zip file found.")
        zip_path = zips[0]

    # Extract zip into a temp dir, then flatten into dest
    print(f"Extracting {zip_path} ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        zip_path.unlink()

        # Extract any nested .tgz / .tar.gz inside the temp dir
        for tgz in list(tmp_path.rglob("*.tgz")) + list(tmp_path.rglob("*.tar.gz")):
            print(f"Extracting nested archive {tgz.name} ...")
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(tgz.parent)
            tgz.unlink()

        # Find the actual data root — skip single wrapper folders
        contents = list(tmp_path.iterdir())
        data_root = tmp_path
        while len(contents) == 1 and contents[0].is_dir():
            data_root = contents[0]
            contents = list(data_root.iterdir())

        # Move everything into dest
        dest.mkdir(parents=True, exist_ok=True)
        for item in contents:
            target = dest / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(dest))

    print(f"\nDone. Dataset extracted to: {dest.resolve()}")
    for item in sorted(dest.iterdir()):
        print(f"  {item.name}")


def main():
    parser = argparse.ArgumentParser(description="Download INbreast dataset from Kaggle")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    ensure_kaggle_installed()
    setup_credentials()       # writes kaggle.json to disk
    download(args.dest, args.force)  # imports kaggle only after credentials exist


if __name__ == "__main__":
    main()
