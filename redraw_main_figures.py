from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
VECTOR = FIGURES / "vector"


PALETTE = {
    "neural": "#0072B2",
    "mcafee": "#D55E00",
    "posted": "#009E73",
    "state_posted": "#CC79A7",
    "first_best": "#7F7F7F",
    "trade_reduction": "#E69F00",
    "grid": "#D9D9D9",
    "text": "#222222",
}

POLICY_LABELS = {
    "Neural dynamic mechanism": "Audited queue-aware",
    "Myopic first best": "First best",
    "Dynamic McAfee": "McAfee",
    "Dynamic posted price": "Posted price",
    "Optimized state posted price": "State posted",
    "Dynamic trade reduction": "Trade reduction",
    "No trade": "No trade",
    "Queue-aware McAfee": "Queue-aware McAfee",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "text.color": PALETTE["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    VECTOR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png")
    fig.savefig(VECTOR / f"{stem}.pdf")
    fig.savefig(VECTOR / f"{stem}.svg")
    plt.close(fig)


def as_float(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def soft_grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=PALETTE["grid"], linewidth=0.55)
    ax.set_axisbelow(True)


def plot_static_efficiency() -> None:
    df = as_float(
        pd.read_csv(TABLES / "main_results.csv"),
        ["neural_efficiency", "mcafee_efficiency", "posted_price_efficiency"],
    )
    labels = [fill(label.replace("Uniform-trained", "Uniform trained"), 22) for label in df["label"]]
    y = np.arange(len(df))
    height = 0.22

    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.barh(
        y + height,
        df["neural_efficiency"],
        height=height,
        label="Neural",
        color=PALETTE["neural"],
        linewidth=0.45,
        edgecolor="#1A1A1A",
    )
    ax.barh(
        y,
        df["mcafee_efficiency"],
        height=height,
        label="McAfee",
        color=PALETTE["mcafee"],
        hatch="//",
        linewidth=0.45,
        edgecolor="#1A1A1A",
    )
    ax.barh(
        y - height,
        df["posted_price_efficiency"],
        height=height,
        label="Posted price",
        color=PALETTE["posted"],
        hatch="..",
        linewidth=0.45,
        edgecolor="#1A1A1A",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.04)
    ax.set_xlabel("Efficiency relative to first best")
    soft_grid(ax, "x")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.52, 1.01))
    save_figure(fig, "main_efficiency")


def plot_regret_efficiency() -> None:
    df = as_float(
        pd.read_csv(TABLES / "main_results.csv"),
        ["neural_efficiency", "total_mean_regret", "mcafee_efficiency"],
    )
    beats = df["neural_efficiency"] >= df["mcafee_efficiency"]
    colors = np.where(beats, PALETTE["neural"], "#8C6D31")
    markers = np.where(beats, "o", "s")

    fig, ax = plt.subplots(figsize=(5.35, 3.85))
    for _, row in df.iterrows():
        marker = "o" if row["neural_efficiency"] >= row["mcafee_efficiency"] else "s"
        color = PALETTE["neural"] if marker == "o" else "#8C6D31"
        ax.scatter(
            row["total_mean_regret"],
            row["neural_efficiency"],
            s=42,
            marker=marker,
            color=color,
            edgecolor="#1A1A1A",
            linewidth=0.5,
            zorder=3,
        )

    short_labels = {
        "Uniform 2x2": "U2x2",
        "Uniform 3x3, ranked audit": "U3x3 ranked",
        "Uniform 3x3, augmented": "U3x3 augmented",
        "Uniform 5x5, ranked": "U5x5 ranked",
        "Uniform 7x7, stress": "U7x7 stress",
        "Uniform-trained on beta-easy": "OOD beta",
        "Uniform-trained on correlated": "OOD corr.",
    }
    offsets = {
        "Uniform 2x2": (5, 6),
        "Uniform 3x3, ranked audit": (5, 9),
        "Uniform 3x3, augmented": (6, -12),
        "Uniform 5x5, ranked": (-54, 8),
        "Uniform 7x7, stress": (-72, -14),
        "Uniform-trained on beta-easy": (-58, 9),
        "Uniform-trained on correlated": (5, -12),
    }
    for _, row in df.iterrows():
        dx, dy = offsets.get(row["label"], (5, 5))
        ax.annotate(
            short_labels.get(row["label"], row["label"]),
            (row["total_mean_regret"], row["neural_efficiency"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#777777"},
        )

    ax.set_xlabel("Hybrid ex-post regret")
    ax.set_ylabel("Neural efficiency")
    ax.set_ylim(0.53, 0.95)
    xmin = max(0.0, df["total_mean_regret"].min() - 0.0009)
    xmax = df["total_mean_regret"].max() + 0.0011
    ax.set_xlim(xmin, xmax)
    soft_grid(ax, "both")
    ax.scatter([], [], s=42, marker="o", color=PALETTE["neural"], edgecolor="#1A1A1A", linewidth=0.5, label="Above McAfee")
    ax.scatter([], [], s=42, marker="s", color="#8C6D31", edgecolor="#1A1A1A", linewidth=0.5, label="Below McAfee")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "regret_efficiency")


def plot_dynamic_efficiency() -> None:
    df = as_float(
        pd.read_csv(TABLES / "dynamic_results.csv"),
        ["efficiency", "efficiency_std"],
    )
    keep = [
        "Neural dynamic mechanism",
        "Myopic first best",
        "Dynamic McAfee",
        "Dynamic posted price",
        "Optimized state posted price",
        "Dynamic trade reduction",
    ]
    df = df[df["policy"].isin(keep)].copy()
    df["policy"] = pd.Categorical(df["policy"], categories=keep, ordered=True)
    df = df.sort_values("policy")
    colors = [
        PALETTE["neural"],
        PALETTE["first_best"],
        PALETTE["mcafee"],
        PALETTE["posted"],
        PALETTE["state_posted"],
        PALETTE["trade_reduction"],
    ]
    hatches = ["", "", "//", "..", "\\\\", "xx"]

    fig, ax = plt.subplots(figsize=(6.8, 3.85))
    y = np.arange(len(df))
    bars = ax.barh(
        y,
        df["efficiency"],
        xerr=df["efficiency_std"].fillna(0.0),
        capsize=2.5,
        color=colors,
        edgecolor="#1A1A1A",
        linewidth=0.45,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#333333"},
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_yticks(y)
    ax.set_yticklabels([POLICY_LABELS[p] for p in df["policy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Efficiency relative to dynamic first best")
    ax.set_xlim(0, 1.06)
    soft_grid(ax, "x")
    save_figure(fig, "dynamic_efficiency")


def plot_dynamic_robustness() -> None:
    df = as_float(
        pd.read_csv(TABLES / "dynamic_robustness.csv"),
        ["neural_efficiency", "neural_efficiency_std", "mcafee_efficiency", "posted_efficiency"],
    )
    labels = [fill(label, 16) for label in df["label"]]
    y = np.arange(len(df))
    height = 0.22

    fig, ax = plt.subplots(figsize=(7.15, 4.15))
    ax.barh(
        y + height,
        df["neural_efficiency"],
        xerr=df["neural_efficiency_std"],
        height=height,
        label="Audited queue-aware",
        color=PALETTE["neural"],
        edgecolor="#1A1A1A",
        linewidth=0.45,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#333333"},
        capsize=2.5,
    )
    ax.barh(
        y,
        df["mcafee_efficiency"],
        height=height,
        label="McAfee",
        color=PALETTE["mcafee"],
        hatch="//",
        edgecolor="#1A1A1A",
        linewidth=0.45,
    )
    ax.barh(
        y - height,
        df["posted_efficiency"],
        height=height,
        label="Posted price",
        color=PALETTE["posted"],
        hatch="..",
        edgecolor="#1A1A1A",
        linewidth=0.45,
    )
    ax.axvline(0, color="#555555", linewidth=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.08, 1.04)
    ax.set_xlabel("Efficiency relative to dynamic first best")
    soft_grid(ax, "x")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.53, 1.01))
    save_figure(fig, "dynamic_robustness")


def plot_dynamic_policy_trace() -> None:
    source = ROOT / "experiments" / "dynamic_policy_interpretation" / "dynamic_policy_trace_summary.csv"
    df = as_float(
        pd.read_csv(source),
        ["first_best_volume", "mcafee_volume", "posted_volume", "neural_volume"],
    )
    order = ["seller heavy", "balanced", "buyer heavy"]
    df = df.set_index("state").reindex(order).reset_index()
    x = np.arange(len(df))
    width = 0.19

    fig, ax = plt.subplots(figsize=(6.25, 3.65))
    series = [
        ("First best", "first_best_volume", PALETTE["first_best"], ""),
        ("McAfee", "mcafee_volume", PALETTE["mcafee"], "//"),
        ("Posted price", "posted_volume", PALETTE["posted"], ".."),
        ("Audited queue-aware", "neural_volume", PALETTE["neural"], ""),
    ]
    for idx, (label, col, color, hatch) in enumerate(series):
        pos = x + (idx - 1.5) * width
        ax.bar(
            pos,
            df[col],
            width=width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="#1A1A1A",
            linewidth=0.45,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Seller heavy", "Balanced", "Buyer heavy"])
    ax.set_ylabel("Mean trade volume on simulated paths")
    ax.set_ylim(0, max(df["first_best_volume"].max(), df["neural_volume"].max()) * 1.18)
    soft_grid(ax, "y")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save_figure(fig, "dynamic_policy_trace")


def write_manifest() -> None:
    def rel(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    rows = []
    for stem in [
        "main_efficiency",
        "regret_efficiency",
        "dynamic_policy_trace",
        "dynamic_efficiency",
        "dynamic_robustness",
    ]:
        png = FIGURES / f"{stem}.png"
        pdf = VECTOR / f"{stem}.pdf"
        svg = VECTOR / f"{stem}.svg"
        rows.append(
            {
                "figure": stem,
                "png": rel(png),
                "pdf": rel(pdf),
                "svg": rel(svg),
                "png_bytes": png.stat().st_size if png.exists() else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(FIGURES / "redraw_manifest.csv", index=False)


def main() -> None:
    configure_style()
    plot_static_efficiency()
    plot_regret_efficiency()
    plot_dynamic_policy_trace()
    plot_dynamic_efficiency()
    plot_dynamic_robustness()
    write_manifest()


if __name__ == "__main__":
    main()
