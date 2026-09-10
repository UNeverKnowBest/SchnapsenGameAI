"""Explicit golden-fixture update command; normal tests never regenerate their expectations."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compatibility/python_oracle"))
from adapter import replay, MANIFEST
from scenarios import scenarios, assert_expected

fixtures = []
for case in scenarios():
    records = replay(case["request"])
    assert_expected(records, case["assertions"])
    fixtures.append({"name":case["name"], "request":case["request"], "records":records})
output = {"reference":MANIFEST["reference"]["commit"], "fixtures":fixtures}
path = Path(__file__).with_name("handwritten.json")
path.write_text(json.dumps(output, separators=(",", ":"))+"\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} verified handwritten fixtures to {path}")
