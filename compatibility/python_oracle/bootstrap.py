"""Fetch the exact manifest commit into an ignored, separate reference checkout."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE.parent / "manifest.json").read_text(encoding="utf-8-sig"))
target = HERE / "upstream"
ref = manifest["reference"]
if not target.exists():
    subprocess.run(["git", "clone", "--no-checkout", ref["repository"], str(target)], check=True)
else:
    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise SystemExit("Reference checkout has local changes; refusing to overwrite them.")
subprocess.run(["git", "-C", str(target), "fetch", "origin", ref["commit"]], check=True)
subprocess.run(["git", "-C", str(target), "checkout", "--detach", ref["commit"]], check=True)
print(f"Pinned oracle ready: {ref['commit']}. adapter.py imports only standard-library engine modules.")
