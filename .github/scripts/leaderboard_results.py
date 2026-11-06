"""Copy the leaderboard's latest released results into the package, when they are newer than the shipped copy.

The leaderboard (gblayer/strable-leaderboard) tags each results release vX.Y.Z. When its highest tag is newer than
``leaderboard_version`` in strable/data/leaderboard/manifest.json, this writes that tag's
site/results_per_fold.parquet, the manifest (version, build date, commit, SHA-256) and the next minor version of
the package, then prints ``tag=…`` and ``version=…`` for $GITHUB_OUTPUT. It prints nothing when the copy is
current. Standard library only: it runs before the package is installed.
"""

import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

LEADERBOARD = "gblayer/strable-leaderboard"
FILE = "site/results_per_fold.parquet"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "strable" / "data" / "leaderboard"
INIT = ROOT / "strable" / "__init__.py"
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")  # releases only, not pre-releases such as v0.1.0-pre
VERSION = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.M)


def fetch(url: str) -> bytes:
    headers = {"Accept": "application/vnd.github+json"}
    if url.startswith("https://api.github.com/") and os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    for attempt in range(3):  # GitHub sometimes fails a request; a daily run should not fail for that
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as reply:
                return reply.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(30 * (attempt + 1))


def key(tag: str) -> tuple[int, ...]:
    return tuple(int(x) for x in TAG.match(tag).groups())


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    tags = json.loads(fetch(f"https://api.github.com/repos/{LEADERBOARD}/tags?per_page=100"))
    tags = [t for t in tags if TAG.match(t["name"])]
    latest = max(tags, key=lambda t: key(t["name"]))
    tag, commit = latest["name"], latest["commit"]["sha"]
    if key(tag) <= key(manifest["leaderboard_version"]):
        return

    raw = f"https://raw.githubusercontent.com/{LEADERBOARD}/{commit}"
    info = json.loads(fetch(f"{raw}/site/results.json"))
    if info["version"] != tag:
        raise SystemExit(f"site/results.json at {tag} says version {info['version']}")
    rows = fetch(f"{raw}/{FILE}")
    (DATA / "results_per_fold.parquet").write_bytes(rows)
    manifest.update(leaderboard_version=tag, built=info["built"], commit=commit, file=FILE,
                    sha256=hashlib.sha256(rows).hexdigest())
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    major, minor, _ = (int(x) for x in VERSION.search(INIT.read_text()).groups())
    version = f"{major}.{minor + 1}.0"
    INIT.write_text(VERSION.sub(f'__version__ = "{version}"', INIT.read_text(), count=1))
    print(f"tag={tag}")
    print(f"version={version}")


if __name__ == "__main__":
    main()
