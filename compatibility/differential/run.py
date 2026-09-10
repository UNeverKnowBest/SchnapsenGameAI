"""Separate privileged-state and public-observation comparisons with reproducible failures."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compatibility/python_oracle"))
sys.path.insert(0, str(ROOT / "compatibility/fixtures"))
from adapter import Oracle, MANIFEST
from scenarios import scenarios, assert_expected


def difference(a, b, path=""):
    if type(a) != type(b):
        return f"{path}: types {type(a).__name__} / {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {a.keys()} / {b.keys()}"
        for key in a:
            diff = difference(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: lengths {len(a)} / {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = difference(x, y, f"{path}[{i}]")
            if diff:
                return diff
    elif a != b:
        return f"{path}: Python={a!r}, Rust={b!r}"
    return None


def compare(records, actual, mode, coverage):
    if len(records) != len(actual):
        return f"record counts: {len(records)} / {len(actual)}"
    for index, (py, rs) in enumerate(zip(records, actual)):
        if mode in ("all", "states"):
            # Ordered equality below is stronger; explicitly check semantic sets too.
            for key in ["legal_moves"]:
                p = {json.dumps(m, sort_keys=True) for m in py["state"][key]}
                r = {json.dumps(m, sort_keys=True) for m in rs["state"][key]}
                if p != r:
                    return f"record {index}: normalized legal move sets differ"
            diff = difference(py["state"], rs["state"], f"record[{index}].state")
            if diff:
                return diff
            coverage["state_records_compared"] += 1
        if mode in ("all", "observations"):
            diff = difference(py["observations"], rs["observations"], f"record[{index}].observations")
            if diff:
                return diff
            coverage["observations_compared"] += len(py["observations"])
            coverage["history_views_compared"] += sum(len(o["history"]) for o in py["observations"])
    return None


class RustProcess:
    def __init__(self, executable):
        self.process = subprocess.Popen([str(executable), "replay"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding="utf-8")

    def replay(self, request):
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Rust replay stopped: {self.process.poll()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["records"]

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=10)
        self.process.stdout.close()
        if code != 0:
            raise RuntimeError(f"Rust exit code {code}")


def run(args):
    coverage = Counter()
    rust = RustProcess(args.engine)
    start = time.monotonic()
    def check(name, request, expected, assertions=()):
        try:
            actual = rust.replay(request)
            assert_expected(expected, assertions)
            assert_expected(actual, assertions)
            mismatch = compare(expected, actual, args.mode, coverage)
            if mismatch:
                raise AssertionError(mismatch)
        except Exception as exc:
            folder = ROOT / "compatibility/differential/failures"
            folder.mkdir(exist_ok=True)
            artifact = {"name":name, "request":request, "error":str(exc), "python":expected,
                        "rust":locals().get("actual"), "reference":MANIFEST["reference"]["commit"]}
            (folder / f"{name}.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            raise AssertionError(f"{name}: {exc}; saved replay in {folder}") from exc

    try:
        for case in scenarios():
            oracle = Oracle(case["request"])
            records = [oracle.record()]
            for action in case["request"]["actions"]:
                oracle.step(action)
                records.append(oracle.record())
            check(case["name"], case["request"], records, case["assertions"])
            coverage["handwritten_scenarios"] += 1

        for game_id in range(args.games):
            rng = random.Random(args.seed + game_id)
            deck = list(range(20))
            rng.shuffle(deck)  # Explicit permutation passed to both engines.
            request = {"deck":deck, "actions":[]}
            oracle = Oracle(request)
            records = [oracle.record()]
            while oracle.outcome() is None:
                legal = oracle.legal_moves()
                special = [m for m in legal if m["kind"] != "play"]
                # Mix uniform policies and deliberate special-move pressure.
                action = rng.choice(special if game_id % 3 == 0 and special else legal)
                request["actions"].append(action)
                coverage[action["kind"] + "_actions"] += 1
                oracle.step(action)
                record = oracle.record()
                records.append(record)
                coverage[f"phase_{record['state']['phase']}_records"] += 1
                if len(request["actions"]) > 40:
                    raise AssertionError("unexpectedly long game")
            outcome = oracle.outcome()
            coverage[f"award_{outcome['game_points']}"] += 1
            coverage[f"winner_{outcome['winner']}"] += 1
            if records[-1]["state"]["scores"][outcome["winner"]]["direct_points"] < 66:
                coverage["last_trick_below_66"] += 1
            check(f"seed-{args.seed+game_id}", request, records)
            coverage["complete_trajectories"] += 1
            if (game_id+1) % 250 == 0:
                print(f"{game_id+1}/{args.games}: zero mismatches; {time.monotonic()-start:.1f}s", flush=True)
    finally:
        rust.close()
    report = {"reference":MANIFEST["reference"]["commit"], "seed":args.seed, "mode":args.mode,
              "coverage":dict(coverage), "unexplained_mismatches":0, "seconds":time.monotonic()-start,
              "python":sys.version, "engine":str(args.engine)}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20250910)
    parser.add_argument("--mode", choices=["all","states","observations"], default="all")
    parser.add_argument("--engine", type=Path, default=ROOT / "engine/target/release" / ("schnapsen-engine.exe" if os.name=="nt" else "schnapsen-engine"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.games < 0:
        parser.error("--games must be nonnegative")
    run(args)
