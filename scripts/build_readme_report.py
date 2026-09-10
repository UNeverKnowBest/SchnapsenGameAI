"""Freeze local experiment evidence and reproduce README figures without training.

Default: render the committed snapshot (requires numpy and matplotlib).
--refresh: first read local CPU results and a fixed, complete-line CUDA log prefix.
Run from the repository root. Refreshing figures does not rewrite README prose.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'docs/report_data/results_snapshot.json'
FIGURES = ROOT / 'docs/report_figures'


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in ('pair_win_rates', 'pair_point_differences')}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def provenance(path, raw):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                file_modified_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())


def refresh():
    cpu_path = ROOT / 'runs/academic_comparison_20260910/results.json'
    cuda_path = ROOT / 'runs/day_cuda/events.jsonl'
    protocol_path = cuda_path.with_name('protocol.json')
    cpu_raw = cpu_path.read_bytes()
    # Fix the byte boundary before reading; ignore only an unfinished final line.
    limit = cuda_path.stat().st_size
    with cuda_path.open('rb') as stream:
        raw = stream.read(limit)
    raw = raw[:raw.rfind(b'\n') + 1]
    cpu = json.loads(cpu_raw)
    events = [json.loads(line) for line in raw.splitlines()]
    protocol_raw = protocol_path.read_bytes()
    progress, bins = {}, defaultdict(lambda: defaultdict(list))
    for row in events:
        if row['kind'] != 'train':
            continue
        algorithm, seed = row['algorithm'], row['seed']
        progress[(algorithm, seed)] = compact(row)
        bucket = (row['total_games'] - 1) // 50000 + 1
        values = bins[(algorithm, seed, bucket)]
        values['games'].append(row['total_games'])
        if algorithm == 'dqn':
            for player in (0, 1):
                loss = row['losses'][player]['rl']
                if loss is not None:
                    values[f'rl_p{player}'].append(loss)
                values[f'epsilon_p{player}'].append(row['epsilon'][player])
        elif algorithm == 'ppo':
            for name in ('normalized_entropy', 'approx_kl', 'value_loss'):
                values[name].append(row['losses'][name])
    diagnostics = [dict(algorithm=a, seed=s, bin_end_games=b*50000,
                        waves=len(v['games']), **{k: statistics.mean(x) for k, x in v.items()})
                   for (a,s,b), v in sorted(bins.items())]
    data = dict(schema_version=1, captured_utc=datetime.now(timezone.utc).isoformat(),
                sources=[provenance(cpu_path, cpu_raw), provenance(cuda_path, raw),
                         provenance(protocol_path, protocol_raw)],
                notes={'curves': 'Raw periodic evaluation, no smoothing or seed mixing.',
                       'diagnostics': 'Arithmetic mean of logged wave metrics in 50,000-game bins; last bin may be partial.',
                       'omissions': 'Per-deck pair arrays omitted; original per-policy uncertainty summaries retained.',
                       'cuda_status': 'Incomplete suite; final tests exist only for seed 42.'},
                cpu={k: compact(cpu[k]) for k in ('protocol', 'summary', 'curves', 'cross_play', 'plateau')},
                cuda=dict(protocol=json.loads(protocol_raw), event_counts=dict(Counter(e['kind'] for e in events)),
                          progress=list(progress.values()), diagnostics=diagnostics,
                          curves=[compact(e) for e in events if e['kind']=='evaluation'],
                          final_tests=[compact(e) for e in events if e['kind']=='final_test'],
                          cross_play=[compact(e) for e in events if e['kind']=='cross_play']))
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return data


def plot(data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'schnapsen-report'})
    colors = {'dqn':'#0072B2', 'ppo':'#D55E00', 'he-ppo':'#009E73', 'nfsp':'#CC79A7'}
    labels = {'dqn':'DQN', 'ppo':'PPO', 'he-ppo':'HE-PPO', 'nfsp':'NFSP'}
    FIGURES.mkdir(parents=True, exist_ok=True)
    def save(fig, name, caption):
        fig.text(.5, .015, caption, ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, .94))
        for extension in ('png', 'svg'):
            fig.savefig(FIGURES / f'{name}.{extension}', dpi=200,
                        metadata={'Date': None} if extension=='svg' else {})
        plt.close(fig)
    def win_axis(ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylabel('Evaluation win rate')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, baseline in zip(axes, ('random', 'heuristic')):
        for i, a in enumerate(('he-ppo','ppo','nfsp','dqn')):
            r = next(x for x in data['cpu']['summary'] if x['algorithm']==a and x['baseline']==baseline)['win']
            y, (low, high) = r['mean'], r['ci95']
            ax.bar(i, y, color=colors[a], width=.65)
            ax.errorbar(i, y, yerr=[[y-low],[high-y]], color='#222222', capsize=4)
            ax.text(i, high+.018, f'{y:.1%}', ha='center', fontsize=10)
        ax.set_xticks(range(4), ['HE-PPO','PPO','NFSP','DQN'])
        ax.set_title(f'Final test vs {baseline}')
        win_axis(ax)
    fig.suptitle('Completed CPU study: 100,000 games per algorithm and seed')
    save(fig, 'cpu_final', 'Mean and 95% Student-t interval across 3 training seeds; 4,000 test games per policy and opponent.')

    cuda = data['cuda']
    def rows(a, baseline):
        return sorted([r for r in cuda['curves'] if r['algorithm']==a and
                       r['training_seed']==42 and r['baseline']==baseline], key=lambda r:r['games'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, baseline in zip(axes, ('random','heuristic')):
        for a in ('dqn','ppo'):
            r=rows(a,baseline)
            ax.plot([x['games']/1e6 for x in r], [x['win_rate'] for x in r], color=colors[a], label=labels[a], lw=1.8)
        ax.set_title(f'Periodic evaluation vs {baseline}')
        ax.set_xlabel('Training games (millions)')
        ax.set_xlim(0,5)
        win_axis(ax)
        ax.legend(frameon=False, loc='lower right')
    fig.suptitle('CUDA learning curves: seed 42 only')
    save(fig, 'cuda_learning', 'Raw evaluations every 50,000 games; 10,000 games per policy and opponent. No across-seed confidence band.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, scale, label in zip(axes, ('decisions','seconds'), (1e6,60),
                                   ('Environment decisions (millions)','Pure training time (minutes)')):
        for a in ('dqn','ppo'):
            r=rows(a,'heuristic')
            ax.plot([x[key]/scale for x in r], [x['win_rate'] for x in r], color=colors[a], label=labels[a], lw=1.8)
        win_axis(ax)
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        ax.legend(frameon=False, loc='lower right')
    fig.suptitle('CUDA efficiency: win rate vs heuristic, seed 42 only')
    save(fig, 'cuda_efficiency', 'Time excludes evaluation, report writing and checkpoint I/O; algorithms have different update workloads.')

    policies = ['random','heuristic','dqn/p0','dqn/p1','ppo']
    matrix = np.full((5,5), np.nan)
    for r in cuda['cross_play']:
        if r['training_seed']!=42:
            continue
        i,j=policies.index(r['policy']),policies.index(r['opponent'])
        matrix[i,j],matrix[j,i]=r['win_rate'],1-r['win_rate']
    fig, ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(matrix, vmin=0,vmax=1,cmap='RdBu')
    display=['Random','Heuristic','DQN p0','DQN p1','PPO']
    ax.set_xticks(range(5),display)
    ax.set_yticks(range(5),display)
    ax.set_xlabel('Opponent (column)')
    ax.set_ylabel('Policy (row)')
    for i in range(5):
        for j in range(5):
            v=matrix[i,j]
            ax.text(j,i,'--' if np.isnan(v) else f'{v:.2%}',ha='center',va='center',
                    color='white' if not np.isnan(v) and abs(v-.5)>.28 else '#222222')
    fig.colorbar(im, ax=ax, format=PercentFormatter(1),label='Row policy win rate')
    fig.suptitle('Final CUDA cross-play: seed 42, 5 million training games')
    save(fig, 'cuda_crossplay', '20,000 paired-seat games per policy pair. DQN p0 and p1 are two policies from one training seed.')

    fig, axes=plt.subplots(2,2,figsize=(11,7))
    specs=[('dqn',['rl_p0','rl_p1'],'DQN replay loss'),
           ('dqn',['epsilon_p0','epsilon_p1'],'DQN exploration probability'),
           ('ppo',['normalized_entropy'],'PPO normalized entropy'),
           ('ppo',['approx_kl'],'PPO approximate KL')]
    for ax,(a,keys,title) in zip(axes.flat,specs):
        r=[x for x in cuda['diagnostics'] if x['algorithm']==a and x['seed']==42]
        for key in keys:
            ax.plot([x['games']/1e6 for x in r],[x[key] for x in r],label=key,lw=1.5)
        ax.set_title(title)
        ax.set_xlabel('Training games (millions)')
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('CUDA training diagnostics: seed 42 only')
    save(fig, 'cuda_diagnostics', 'Mean of logged wave metrics within each 50,000-game bin. Stable loss alone does not show convergence.')
    print(json.dumps({'snapshot':str(SNAPSHOT), 'figure_files':10,
                      'cuda_events':cuda['event_counts'], 'matplotlib':matplotlib.__version__}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Read local runs and replace the committed evidence snapshot')
    args=parser.parse_args()
    data=refresh() if args.refresh else json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    plot(data)


if __name__=='__main__':
    main()
