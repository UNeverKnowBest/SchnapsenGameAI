import json
from nfsp.cli import main


def test_comparison_writes_reviewable_results_and_checkpoints(tmp_path):
    output = tmp_path / "comparison"
    main(["compare", "--games", "8", "--eval-every", "8", "--eval-games", "8", "--test-games", "8",
          "--seeds", "1,2", "--num-envs", "8", "--workers", "1", "--device", "cpu",
          "--torch-threads", "1", "--hidden", "16", "--replay-capacity", "512", "--output", str(output)])
    report = json.loads((output / "results.json").read_text())
    assert len(report["final_tests"]) == 16
    assert report["protocol"]["test_seed"] != report["protocol"]["cross_seed"]
    assert all(row['status'] == 'insufficient_data' for row in report['plateau'])
    assert (output / 'learning_statistics.csv').exists()
    assert (output / 'plateau.json').exists()
    live = (output / 'live.html').read_text(encoding='utf-8')
    assert '已完成' in live and 'http-equiv="refresh"' not in live
    for summary in report['summary']:
        values = [r['win_rate'] for r in report['final_tests']
                  if (r['algorithm'], r['baseline']) == (summary['algorithm'], summary['baseline'])]
        assert summary['win']['mean'] == sum(values) / len(values)
    eval_seeds = {report['protocol']['eval_seed'] + r['games'] * 101 for r in report['curves']}
    assert report['protocol']['test_seed'] not in eval_seeds
    assert report['protocol']['cross_seed'] not in eval_seeds
    assert len(report["summary"]) == 8
    assert len(report["final_runs"]) == 8
    assert all(row["games"] == 8 for row in report["final_runs"])
    assert len(report["cross_play"]) == 56
    assert (output / "he-ppo_seed1" / "policy.pt").exists()
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html and "data-baseline" in html
    assert "cdn" not in html


def test_performance_runs_matched_work_and_generates_report(tmp_path):
    output = tmp_path / "performance"
    main(["performance", "--games", "8", "--repeats", "2", "--env-counts", "1,4",
          "--worker-counts", "1,2", "--device", "cpu", "--hidden", "16", "--torch-threads", "1",
          "--output", str(output)])
    report = json.loads((output / "results.json").read_text())
    assert all(report["matched_workload_checks"].values())
    assert len(report["measurements"]) == 16
    training = [r for r in report["summary"] if r["mode"] == "training"]
    assert len({r["optimizer_steps"] for r in training}) == 1
    assert all(r["optimizer_steps"] > 0 for r in training)
    assert (output / "results.csv").exists()
