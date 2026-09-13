"""Графики для анализа выживаемости. Статика — GitHub не рендерит интерактив."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
# Категориальные слоты 1-5. Каждая кривая подписана прямо на графике:
# три из пяти цветов не добирают контраста к фону, и подпись это компенсирует.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "axes.edgecolor": GRID,
    "axes.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 0,
})


def _strip(ax, xgrid=True, ygrid=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _header(fig, title, subtitle, source, top=0.88):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(0.012, 0.925, subtitle, ha="left", va="top", fontsize=9.5,
             color=INK_SOFT, linespacing=1.45)
    fig.text(0.012, 0.012, source, ha="left", va="bottom", fontsize=8.5, color=INK_SOFT)
    fig.subplots_adjust(top=top)


def _step(ax, km, color, lw=2.0, label=None):
    """Ступенчатая кривая доли вернувшихся, начиная от нуля в момент травмы."""
    t = np.concatenate([[0], km.time.values])
    y = np.concatenate([[0], 1 - km.surv.values])
    ax.step(t, y, where="post", color=color, linewidth=lw, label=label, zorder=3)
    return t, y


def return_curves(curves: dict, out, horizon=400, title="", subtitle="", source=""):
    """Все кривые на одном поле: сравнение типов повреждения между собой."""
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=200)
    order = sorted(curves, key=lambda k: (1 - curves[k].surv.values[-1]))

    # Подпись ставится на уровне, которого кривая достигла к концу наблюдения.
    # Если подписи сходятся ближе чем на min_gap, разводим их по вертикали:
    # иначе пять названий сливаются в одну кашу у правого края.
    ends = {}
    for i, name in enumerate(order):
        color = SLOTS[i % len(SLOTS)]
        _step(ax, curves[name], color)
        ends[name] = (1 - curves[name].surv.values[-1], color)

    min_gap = 0.052
    placed = []
    for name in sorted(ends, key=lambda k: ends[k][0]):
        y, color = ends[name]
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((name, y, color))

    for name, y, color in placed:
        ax.text(horizon * 1.04, y, name, color=color, fontsize=9,
                va="center", fontweight="bold")

    _strip(ax, xgrid=False)
    ax.set_xlim(0, horizon * 1.36)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.25)])
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Игровых дней с момента травмы", fontsize=10, labelpad=10)
    ax.set_ylabel("Доля вернувшихся в состав", fontsize=10, labelpad=10)
    ax.axvline(365, color=GRID, linewidth=1.2, zorder=1)
    ax.text(367, 0.03, "год", fontsize=8.5, color=INK_SOFT)

    fig.tight_layout()
    _header(fig, title, subtitle, source, top=0.86)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def return_curves_panels(curves: dict, counts: dict, out, horizon=400,
                         title="", subtitle="", source=""):
    """По панели на тип: та же кривая, но с доверительной полосой."""
    order = sorted(curves, key=lambda k: (1 - curves[k].surv.values[-1]), reverse=True)
    fig, axes = plt.subplots(1, len(order), figsize=(13, 3.6), dpi=200, sharey=True)

    for i, (name, ax) in enumerate(zip(order, axes)):
        km = curves[name]
        color = SLOTS[order.index(name) % len(SLOTS)]

        for other in order:
            if other != name:
                o = curves[other]
                ax.step(np.concatenate([[0], o.time.values]),
                        np.concatenate([[0], 1 - o.surv.values]),
                        where="post", color=GRID, linewidth=1.0, zorder=1)

        t = np.concatenate([[0], km.time.values])
        ax.fill_between(t, np.concatenate([[0], 1 - km.hi.values]),
                        np.concatenate([[0], 1 - km.lo.values]),
                        step="post", color=color, alpha=0.18, linewidth=0, zorder=2)
        _step(ax, km, color)

        _strip(ax, xgrid=False)
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{name}\nn = {counts[name]}", fontsize=9, color=INK, pad=8)
        ax.set_xticks([0, 180, 365])
        ax.set_xticklabels(["0", "180", "365"], fontsize=8.5)
        if i == 0:
            ax.set_yticks(np.arange(0, 1.01, 0.25))
            ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.25)])

    fig.tight_layout()
    _header(fig, title, subtitle, source, top=0.70)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def logging_artifact(ratio: pd.Series, out, title="", subtitle="", source=""):
    """Полнота лога по сезонам: перелом в 2014 году видно невооружённым глазом."""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    colors = [SLOTS[1] if s < 2014 else SLOTS[0] for s in ratio.index]
    ax.bar(ratio.index.astype(str), ratio.values, color=colors, width=0.68)

    for x, v in enumerate(ratio.values):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK_SOFT)

    _strip(ax, xgrid=False)
    ax.set_ylim(0, max(ratio.values) * 1.22)
    ax.set_ylabel("Записей о возвращении на одно выбытие", fontsize=10, labelpad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOTS[1]),
               plt.Rectangle((0, 0), 1, 1, color=SLOTS[0])]
    ax.legend(handles, ["До 2014: часть возвращений не фиксировалась",
                        "С 2014: лог полный"],
              loc="upper left", frameon=False, fontsize=9.5)

    fig.tight_layout()
    _header(fig, title, subtitle, source, top=0.80)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def forest(table: pd.DataFrame, out, title="", subtitle="", source=""):
    """Отношения рисков с интервалами. Левее единицы — возвращается медленнее."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(t) + 2.4), dpi=200)
    y = np.arange(len(t))

    ax.axvline(1.0, color=INK_SOFT, linewidth=1.2, zorder=1)
    for i, (_, row) in enumerate(t.iterrows()):
        sig = row.CI_low > 1 or row.CI_high < 1
        color = SLOTS[0] if sig else INK_SOFT
        ax.plot([row.CI_low, row.CI_high], [i, i], color=color, linewidth=2.2,
                solid_capstyle="round", zorder=2)
        ax.scatter([row.HR], [i], s=70, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.text(ax.get_xlim()[1], i, f"  {row.HR:.2f} [{row.CI_low:.2f}–{row.CI_high:.2f}]",
                va="center", fontsize=9, color=INK_SOFT)

    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0,1", "0,25", "0,5", "1", "2"])
    _strip(ax, ygrid=False)
    ax.set_xlabel("Отношение рисков возвращения", fontsize=10, labelpad=10)
    ax.set_xlim(0.08, 2.6)

    fig.tight_layout()
    _header(fig, title, subtitle, source, top=0.82)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out
