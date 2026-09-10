from argparse import Namespace
import xml.etree.ElementTree as ET

import pytest

from nfsp.learning import (curve_statistics, learning_chart, learning_protocol,
                           plateau_analysis, export_learning, write_live)


def row(seed, games, win, **kwargs):
    return dict(algorithm='ppo', baseline='random', training_seed=seed,
                games=games, win_rate=win, **kwargs)


def test_interval_uses_independent_seeds_and_single_seed_has_no_band():
    curves = [row(s, g, w) for s, w in [(1, .4), (2, .5), (3, .6)] for g in (0, 10)]
    stats = curve_statistics(curves, 'random')
    assert [r['seeds'] for r in stats] == [3, 3]
    assert stats[-1]['mean'] == pytest.approx(.5)
    assert stats[-1]['ci95'][1] == pytest.approx(.5 + 4.303*.1/(3**.5))
    assert 'class="seed-ci"' in learning_chart(curves, 'random')
    single = [r for r in curves if r['training_seed'] == 1]
    assert curve_statistics(single, 'random')[0]['ci95'] is None
    assert 'class="seed-ci"' not in learning_chart(single, 'random')


@pytest.mark.parametrize('axis', ['seconds', 'decisions'])
def test_axis_interpolates_each_seed_at_equal_budget_without_extrapolation(axis):
    curves = [row(1, 0, 0., **{axis:0}), row(1, 10, 1., **{axis:10}),
              row(2, 0, 0., **{axis:0}), row(2, 10, .5, **{axis:5})]
    stats = curve_statistics(curves, 'random', axis)
    assert [r['x'] for r in stats] == [0., 5.]
    assert stats[-1]['mean'] == pytest.approx(.5)
    assert stats[-1]['seeds'] == 2
    assert stats[-1]['ci95'] == pytest.approx([.5, .5])


def test_early_stop_keeps_common_seed_cohort_and_legacy_counts_are_not_invented():
    curves = [row(1, 0, .2), row(1, 10, .3), row(1, 20, .9),
              row(2, 0, .4), row(2, 10, .5)]
    assert [r['x'] for r in curve_statistics(curves, 'random')] == [0, 10]
    assert curve_statistics(curves, 'random', 'decisions') == []


def histories(values, seeds=(1, 2, 3)):
    return [row(seed, i*100, value) for seed in seeds for i, value in enumerate(values)]


def test_plateau_distinguishes_flat_improving_oscillating_and_inconclusive():
    protocol = dict(window=4, min_rounds=4, delta=.02)
    flat = plateau_analysis(histories([.5]*5), protocol)[0]
    assert flat['status'] == 'plateau_observed'
    assert flat['window_start_games'] == 100
    assert plateau_analysis(histories([.3, .4, .5, .6, .7]), protocol)[0]['status'] == 'plateau_not_established'
    assert plateau_analysis(histories([.5, .4, .6, .4, .6]), protocol)[0]['status'] == 'plateau_not_established'
    assert plateau_analysis(histories([.5]*5, (1,)), protocol)[0]['status'] == 'insufficient_data'
    assert plateau_analysis(histories([.5]*4), protocol)[0]['status'] == 'insufficient_data'
    # Opposing small seed trends have mean change zero, but uncertainty too wide.
    curves = []
    for seed, shift in [(1, -.019), (2, 0), (3, .019)]:
        curves += [row(seed, i*100, .5 + (shift if i >= 3 else 0)) for i in range(5)]
    result = plateau_analysis(curves, protocol)[0]
    assert result['change']['mean'] == pytest.approx(0)
    assert result['status'] == 'plateau_not_established'


def test_export_is_standalone_and_live_stops_refreshing(tmp_path):
    curves = [dict(r, decisions=r['games']*10, seconds=r['games']/10) for r in histories([.5]*5)]
    protocol = dict(window=4, min_rounds=4, delta=.02)
    export_learning(tmp_path, curves, protocol)
    assert len(list((tmp_path / 'figures').glob('*.svg'))) == 3
    for path in (tmp_path / 'figures').glob('*.svg'):
        assert ET.parse(path).getroot().tag == '{http://www.w3.org/2000/svg}svg'
    write_live(tmp_path, curves, protocol)
    assert 'http-equiv="refresh"' in (tmp_path / 'live.html').read_text(encoding='utf-8')
    write_live(tmp_path, curves, protocol, complete=True)
    assert 'http-equiv="refresh"' not in (tmp_path / 'live.html').read_text(encoding='utf-8')


@pytest.mark.parametrize('window,delta,minimum', [(3,.02,10), (5,0,10), (5,float('nan'),10), (5,.02,2)])
def test_invalid_plateau_protocol(window, delta, minimum):
    with pytest.raises(ValueError):
        learning_protocol(Namespace(plateau_window=window, plateau_delta=delta, plateau_min_rounds=minimum))
