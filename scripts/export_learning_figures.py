"""Export report statistics as publication figures using optional Matplotlib.

python scripts/export_learning_figures.py runs/<experiment>/results.json
Uses saved statistics, never reruns training or changes the evaluation protocol.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter, MaxNLocator
    except ImportError as error:
        raise SystemExit("This optional exporter requires matplotlib; install it in the interpreter used to run this script.") from error

    data = json.loads(args.results.read_text(encoding="utf-8"))
    stats = json.loads((args.results.parent / "learning_statistics.json").read_text(encoding="utf-8"))
    output = args.output or args.results.parent / "publication_figures"
    output.mkdir(parents=True, exist_ok=True)
    algorithms = sorted({r["algorithm"] for r in stats})
    colors = dict(zip(algorithms, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    baselines = sorted({r["baseline"] for r in stats})
    paths = []
    for index, baseline in enumerate(baselines):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
        for ax, axis, label in zip(axes, ("games", "decisions", "seconds"),
                                   ("Training games", "Environment decisions", "Training time (s)")):
            for algorithm in algorithms:
                rows = sorted([r for r in stats if (r["algorithm"], r["baseline"], r["axis"]) ==
                               (algorithm, baseline, axis)], key=lambda r: r["x"])
                if not rows:
                    continue
                x, y = [r["x"] for r in rows], [r["mean"] for r in rows]
                color = colors[algorithm]
                raw = [r for r in data["curves"] if r["algorithm"] == algorithm and r["baseline"] == baseline]
                seeds = sorted({r.get("training_seed", r.get("seed")) for r in raw})
                for seed in seeds:
                    original = sorted([r for r in raw if r.get("training_seed", r.get("seed")) == seed and
                                       axis in r and x[0] <= r[axis] <= x[-1]], key=lambda r: r[axis])
                    ax.plot([r[axis] for r in original], [r["win_rate"] for r in original],
                            color=color, alpha=.25, linewidth=.7, marker=".", markersize=2)
                if rows[0]["ci95"] is not None:
                    ax.fill_between(x, [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows],
                                    color=color, alpha=.15, linewidth=0)
                ax.plot(x, y, color=color, linewidth=1.8, label=f"{algorithm} (n={rows[0]['seeds']})")
            ax.set_xlabel(label)
            ax.set_ylim(0, 1)
            ax.set_xlim(left=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
            ax.axhline(.5, color="gray", linewidth=.7, linestyle=":")
            ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Evaluation win rate")
        axes[0].legend(loc="lower right", fontsize=8, frameon=False)
        fig.suptitle(f"Learning curves vs {baseline}")
        fig.text(.5, .015, "Pointwise 95% t intervals across training seeds; thin lines are raw seed trajectories. "
                 "Time/decisions: within-support interpolation; training time excludes evaluation and I/O.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, .95))
        for extension in ("svg", "pdf", "png"):
            path = output / f"learning_{index}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    (output / "manifest.json").write_text(json.dumps(dict(source=str(args.results.resolve()),
        baselines=dict(enumerate(baselines)), files=paths, matplotlib=matplotlib.__version__), indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), figures=len(paths))))


if __name__ == "__main__":
    main()
