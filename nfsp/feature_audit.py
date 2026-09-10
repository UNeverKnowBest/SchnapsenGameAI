"""Measure the existing encoder without changing checkpoint/observation semantics."""
from pathlib import Path
import numpy as np
from ._native import BatchEnv, STATE_DIM
from .arena import counter_uniform
from .reports import bars, table, write_csv, write_report


BLOCKS = [("全局状态", 0, 13), ("20 张牌的区域", 13, 133), ("最近五条历史", 133, 243),
          ("填充的合法动作列表", 243, 443), ("派生标志", 443, 445), ("当前领先动作", 445, 465)]


def audit_features(games=4096, num_envs=256, workers=1, seed=70, output="runs/feature_audit"):
    if games <= 0 or num_envs <= 0 or workers <= 0:
        raise ValueError("games/envs/workers must be positive")
    nonzero = np.zeros(STATE_DIM, np.int64)
    low, high = np.full(STATE_DIM, np.inf), np.full(STATE_DIM, -np.inf)
    observations, action_histogram = 0, np.zeros(29, np.int64)
    for first in range(0, games, num_envs):
        count = min(num_envs, games-first)
        env = BatchEnv(count, workers)
        states, masks, players, _, _ = env.reset(seed, first)
        steps = np.zeros(count, np.int64)
        while (players >= 0).any():
            lanes = np.flatnonzero(players >= 0)
            rows = states[lanes]
            nonzero += (rows != 0).sum(axis=0)
            low, high = np.minimum(low, rows.min(0)), np.maximum(high, rows.max(0))
            observations += len(lanes)
            counts = masks[lanes].sum(axis=1).astype(int)
            action_histogram += np.bincount(counts, minlength=29)
            cumulative = masks[lanes].cumsum(axis=1)
            draws = counter_uniform(seed+17, first+lanes, steps[lanes]) * counts
            chosen = (draws[:, None] >= cumulative).sum(1)
            actions = np.full(count, -1, np.int16)
            actions[lanes] = chosen
            steps[lanes] += 1
            states, masks, players, _, _ = env.step(actions)
    rows = [dict(block=name, start=start, end_exclusive=end, dimensions=end-start,
                 observed_constant_dimensions=int((low[start:end] == high[start:end]).sum()),
                 observed_zero_dimensions=int((nonzero[start:end] == 0).sum()),
                 nonzero_fraction=float(nonzero[start:end].sum() / (observations*(end-start))))
            for name, start, end in BLOCKS]
    data = dict(games=games, seed=seed, observations=observations, state_dim=STATE_DIM,
                blocks=rows, nonzero_counts=nonzero.tolist(), minimum=low.tolist(), maximum=high.tolist(),
                legal_action_count_histogram=action_histogram.tolist(),
                structural_unused_rank_dimensions=128,
                recommendation="180-dimensional public snapshot + separate legal mask + GRU(64) over complete player-visible events; proposed, not enabled. Preserve current v2 for controlled algorithm comparison.")
    body = '<p>使用公开观测和随机合法策略审计现有 465 维输入。采样中未出现的维度不等于结构上永远无用；代码可证明的无效 rank 槽位为 128 维。</p>'
    body += bars("各特征块维度", [r["block"] for r in rows], [r["dimensions"] for r in rows], "dims")
    body += table(["特征块", "维度", "采样常量维", "采样全零维", "非零比例"],
                  [[r["block"], r["dimensions"], r["observed_constant_dimensions"], r["observed_zero_dimensions"], f'{r["nonzero_fraction"]:.1%}'] for r in rows])
    body += '<p>建议 20×6 卡牌区域 one-hot + 11 全局量 + 21 当前领先动作 + 28 合法动作，共 180 维当前快照；完整可见事件用 64 维 GRU 记忆。它是待消融验证的方案，不是已证明充分的信念状态。</p>'
    write_report(output, data, "Schnapsen · 卡牌特征审计", body)
    write_csv(output, rows)
    return data
