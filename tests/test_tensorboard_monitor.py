import json
from pathlib import Path

import pytest

from nfsp.tensorboard import EventMirror
from nfsp.experiments import load_algorithm_configs


class Writer:
    def __init__(self):
        self.values = []
    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))
    def flush(self):
        pass
    def close(self):
        pass


def test_incremental_partial_lines_resume_and_final_test_step(tmp_path):
    events, logs = tmp_path / 'events.jsonl', tmp_path / 'tb'
    writer = Writer()
    row = dict(kind='train', algorithm='dqn', seed=42, total_games=5000,
               total_decisions=60000, seconds=2., epsilon=[.2,.19],
               losses=[dict(rl=.3), dict(rl=.4)])
    events.write_bytes(json.dumps(row).encode())
    mirror = EventMirror(events, logs, lambda path: writer)
    assert mirror.read_available() == 0
    with events.open('ab') as f:
        f.write(b'\n')
    assert mirror.read_available() == 1
    assert ('train/epsilon/player_0', .2, 5000) in writer.values
    final = dict(kind='test', algorithm='dqn', training_seed=42, games=4000,
                 opponent='heuristic', win_rate=.6)
    with events.open('a') as f:
        f.write(json.dumps(final) + '\n')
    assert mirror.read_available() == 1
    assert ('test/heuristic/win_rate', .6, 5000) in writer.values
    assert ('test_by_decisions/heuristic/win_rate', .6, 60000) in writer.values
    assert ('test_by_training_ms/heuristic/win_rate', .6, 2000) in writer.values
    mirror.close()
    reopened = EventMirror(events, logs, lambda path: Writer())
    assert reopened.read_available() == 0
    assert reopened.writers == {}
    reopened.close()


def test_sources_cannot_be_mixed_or_truncated(tmp_path):
    events = tmp_path / 'events.jsonl'
    events.write_text('{}\n')
    mirror = EventMirror(events, tmp_path / 'tb', lambda path: Writer())
    mirror.read_available()
    with pytest.raises(ValueError, match='another event source'):
        EventMirror(tmp_path / 'different.jsonl', tmp_path / 'tb', lambda path: Writer())
    events.write_text('')
    with pytest.raises(ValueError, match='truncated'):
        mirror.read_available()
    mirror.close()


def test_evaluation_axes_are_cumulative_and_match_arrays_not_logged(tmp_path):
    writer = Writer()
    mirror = EventMirror(tmp_path / 'events', tmp_path / 'tb', lambda path: writer)
    mirror.emit(dict(kind='evaluation', algorithm='ppo', training_seed=1, baseline='random',
                     games=10000, decisions=150000, seconds=10.25, win_rate=.7,
                     matches=[dict(pair_win_rates=[0,1])]))
    assert ('eval/random/win_rate', .7, 10000) in writer.values
    assert ('eval_by_training_ms/random/win_rate', .7, 10250) in writer.values
    assert len(writer.values) == 3
    mirror.close()


def test_recommended_configs_validate_without_training():
    configs = load_algorithm_configs(['dqn=configs/recommended_dqn.json',
                                     'ppo=configs/recommended_ppo.json'], ['dqn','ppo'])
    assert configs['dqn'].eta == 1 and configs['dqn'].double_dqn
    assert configs['dqn'].rl_lr == .0003
    assert configs['ppo'].entropy_coef == .01 and not configs['ppo'].adaptive_entropy
    with pytest.raises(ValueError):
        load_algorithm_configs(['dqn=configs/recommended_dqn.json']*2, ['dqn'])
