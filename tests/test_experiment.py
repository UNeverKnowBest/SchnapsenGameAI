import json

import numpy as np
import pytest
import torch

import experiment
from nfsp.ppo import PPOConfig, PPOTrainer


def tiny(**kwargs):
    values = dict(num_envs=8, hidden=(16,), device='cpu', torch_threads=1,
                  epochs=2, minibatch_size=32, adaptive_lr=True, deterministic=True)
    values.update(kwargs)
    return PPOConfig(**values)


def test_lr_bounds_kl_backoff_and_checkpoint_resume(tmp_path):
    trainer = PPOTrainer(tiny(lr_max=.00031))
    first = trainer.train_wave()
    assert first['losses']['lr_next'] == .00031
    trainer.save(tmp_path / 'state.pt')
    resumed = PPOTrainer.load(tmp_path / 'state.pt')
    a, b = trainer.train_wave(), resumed.train_wave()
    assert a['losses'] == b['losses']
    for left, right in zip(trainer.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    backoff = PPOTrainer(tiny(target_kl=1e-8))
    result = backoff.train_wave()['losses']
    assert result['early_stop_kl']
    assert result['lr_next'] == pytest.approx(.0003/1.5)
    fixed = PPOTrainer(tiny(adaptive_lr=False))
    assert fixed.train_wave()['losses']['lr_next'] == .0003


def test_packed_collection_preserves_behavior_log_probabilities():
    trainer = PPOTrainer(tiny())
    batch, _ = trainer.collect(8)
    from nfsp.ppo import legal_distribution
    with torch.no_grad():
        dist = legal_distribution(trainer.model(torch.from_numpy(batch['states'])), torch.from_numpy(batch['masks']))
        expected = dist.log_prob(torch.from_numpy(batch['actions'])).numpy()
    np.testing.assert_allclose(batch['old_log_probs'], expected, atol=5e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_rollout_uses_current_cuda_weights_after_update_and_resume(tmp_path):
    trainer = PPOTrainer(tiny(device='cuda', rollout_device='cpu'))
    trainer.train_wave()
    trainer.save(tmp_path / 'state.pt')
    trainer = PPOTrainer.load(tmp_path / 'state.pt')
    batch, _ = trainer.collect(8)
    for a, b in zip(trainer.model.parameters(), trainer.rollout_model.parameters()):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
    assert np.isfinite(batch['old_log_probs']).all()
    assert trainer.train_wave()['losses']['optimizer_steps'] > 0


def test_experiment_matches_budget_separates_test_and_renders(tmp_path):
    out = tmp_path / 'run'
    experiment.main(['--demo', '--skip-tuning', '--num-envs', '8', '--rounds', '3',
                     '--games-per-round', '16', '--max-decisions', '1', '--eval-games', '8',
                     '--test-games', '8', '--epochs', '1', '--latency-repeats', '2', '--output', str(out)])
    data = json.loads((out / 'results.json').read_text())
    assert [r['games'] for r in data['runs']] == [8, 8]
    assert all(r['stop_reason'] == 'decision_budget' for r in data['runs'])
    assert data['protocol']['test_seed'] != data['protocol']['validation_seed']
    assert all('decisions' in row for row in data['curves'])
    assert data['protocol']['learning_analysis']['window'] == 5
    assert (out / 'learning_statistics.csv').exists()
    assert (out / 'figures' / 'learning_random_seconds.svg').exists()
    assert (out / 'figures' / 'learning_random_decisions.svg').exists()
    assert len(data['matches']) == 5
    assert (out / 'adaptive-ppo_seed42' / 'best.pt').exists()
    assert (out / 'adaptive-ppo_seed42' / 'latest.pt').exists()
    for path in (out / 'figures').glob('*.svg'):
        import xml.etree.ElementTree as ET
        assert ET.parse(path).getroot().tag == '{http://www.w3.org/2000/svg}svg'
    assert len(list((out / 'figures').glob('*.svg'))) >= 8
    experiment.main(['--render-only', str(out / 'results.json')])
    with pytest.raises(FileExistsError):
        experiment.main(['--demo', '--skip-tuning', '--output', str(out)])


def test_plateau_stops_both_algorithms_and_keeps_initial_best(tmp_path, monkeypatch):
    real_evaluate = experiment.evaluate_match
    def flat_score(*a, **kw):
        result = real_evaluate(*a, **kw)
        result['win_rate'] = .5
        return result
    monkeypatch.setattr(experiment, 'evaluate_match', flat_score)
    experiment.main(['--demo', '--skip-tuning', '--num-envs', '4', '--rounds', '5',
                     '--games-per-round', '4', '--eval-games', '4', '--test-games', '4',
                     '--epochs', '1', '--latency-repeats', '2', '--min-rounds', '1',
                     '--patience', '1', '--output', str(tmp_path)])
    data = json.loads((tmp_path / 'results.json').read_text())
    assert all(r['games'] == 4 and r['best_round'] == 0 and r['stop_reason'] == 'validation_plateau' for r in data['runs'])


@pytest.mark.parametrize('args', [['--rounds','0'], ['--eval-games','3'], ['--test-games','1'],
                                  ['--seeds','1,1'], ['--seeds','2000042'], ['--seeds','7000042'], ['--max-decisions','0'], ['--tune-repeats','1']])
def test_invalid_experiment_budgets(args):
    with pytest.raises(ValueError):
        experiment.settings(experiment.parser().parse_args(args))


@pytest.mark.parametrize('kw', [dict(lr_min=0), dict(lr_max=float('nan')),
                               dict(lr_min=.001), dict(rollout_device='bad')])
def test_invalid_adaptive_config(kw):
    with pytest.raises(ValueError):
        tiny(**kw).validate()


def test_performance_assessment_reports_gpu_slowdown_without_hiding_it():
    data = dict(calibration=dict(summary=[
        dict(config=dict(device="cpu"), median_games_per_second=100.),
        dict(config=dict(device="cuda"), median_games_per_second=80.)]),
        latency=[dict(device="cpu", batch_size=1, p95_ms=.1),
                 dict(device="cuda", batch_size=1, p95_ms=.5),
                 dict(device="cpu", batch_size=256, p95_ms=2.),
                 dict(device="cuda", batch_size=256, p95_ms=.8)])
    result = experiment.assess_performance(data)
    assert result["cuda_slower_than_best_cpu"]
    assert result["cuda_over_cpu_best_config_ratio"] == .8
    assert [r["preferred_device"] for r in result["latency_recommendations"]] == ["cpu", "cuda"]
