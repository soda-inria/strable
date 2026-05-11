"""Download STRABLE datasets from the public Google Drive bundle."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.path_configs import path_configs  # noqa: E402

GDRIVE_FILE_ID = "1wq2edCjNUGi2uBwAKgGj5eikswmFxFEZ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Public CLI variant -> folder name inside the archive (== folder name on disk)
VARIANT_TO_FOLDER = {
    "default": "data_processed",
    "full": "data_processed_FULL",
    "feature-eng": "data_processed_feature_eng",
    "raw-targets": "data_processed_skewness_inverse_transformation",
}

# Which evaluate script reads each variant — informational, surfaced in `--help`.
VARIANT_USED_BY = {
    "default": "script_evaluate.py and most ablations",
    "full": "script_evaluate_FULL.py (Fig E.9)",
    "feature-eng": "script_evaluate_feature_eng.py (Fig E.6)",
    "raw-targets": "script_evaluate_skewness.py (Fig E.7)",
}


def download_archive(dest: Path) -> None:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown is required. Install it with `pip install gdown`."
        ) from e
    print(f"Downloading from Google Drive (id={GDRIVE_FILE_ID})...")
    gdown.download(GDRIVE_URL, str(dest), quiet=False)
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit(
            "Download failed or produced an empty file. The Drive link may "
            "have moved; re-check the URL printed above."
        )


def extract_archive(archive_path: Path, extract_to: Path) -> Path:
    """Extract zip and return the root containing the four variant folders."""
    print(f"Extracting {archive_path.name}...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_to)

    # Some zips wrap the four variant folders in an outer dir
    # (e.g. ``strable_data/``). Detect that case and dive in.
    entries = [p for p in extract_to.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name not in VARIANT_TO_FOLDER.values():
        return entries[0]
    return extract_to


def install(extracted_root: Path, target_root: Path, variants: list[str], force: bool) -> int:
    n_installed = 0
    for variant in variants:
        folder = VARIANT_TO_FOLDER[variant]
        src = extracted_root / folder
        dst = target_root / folder
        if not src.is_dir():
            print(f"  ⚠ archive does not contain {folder}/ — skipping {variant}")
            continue
        if dst.exists():
            if not force:
                print(f"  = {dst} already exists — skipping {variant} (pass --force to overwrite)")
                continue
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"  + {variant}: installed at {dst}")
        n_installed += 1
    return n_installed


def parse_variants(arg: str) -> list[str]:
    out = [v.strip() for v in arg.split(",") if v.strip()]
    bad = [v for v in out if v not in VARIANT_TO_FOLDER]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown variants: {bad}. Valid: {list(VARIANT_TO_FOLDER)}"
        )
    return out


def main() -> None:
    epilog = "Variant -> consumer:\n" + "\n".join(
        f"  {v:13s}  {VARIANT_USED_BY[v]}" for v in VARIANT_TO_FOLDER
    )
    parser = argparse.ArgumentParser(
        description="Download STRABLE datasets bundle from Google Drive.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=["default"],
        help=(
            "Comma-separated list of variants to install. "
            f"Choices: {sorted(VARIANT_TO_FOLDER)}. Default: 'default'."
        ),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path(path_configs["base_path"]) / "data",
        help="Where to install the variant folders. Default: <STRABLE_ROOT>/data.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing variant folders rather than skipping them.",
    )
    args = parser.parse_args()

    args.target_root.mkdir(parents=True, exist_ok=True)
    print(f"Target root  : {args.target_root}")
    print(f"Variants     : {', '.join(args.variants)}")

    with tempfile.TemporaryDirectory(prefix="strable_gdrive_") as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "strable_datasets.zip"

        download_archive(archive_path)
        extracted_root = extract_archive(archive_path, tmp_path)

        n = install(extracted_root, args.target_root, args.variants, args.force)

        if args.keep_archive:
            kept = args.target_root / archive_path.name
            shutil.copy2(archive_path, kept)
            print(f"Archive kept at {kept}")

    print(f"\nDone. {n} variants installed under {args.target_root}.")


if __name__ == "__main__":
    main()
