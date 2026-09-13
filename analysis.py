"""
Возврат к прежнему уровню после тяжёлой травмы.

Два слоя измерения:
  минуты за игру      — доверяет ли тренер,
  продуктивность/36   — играет ли игрок так же, когда он на площадке.

Контрольная группа обязательна. Игрок, порвавший ахилл в 33 года, потерял бы
часть минут и без травмы: состав молодеет, роль сужается. Без сравнения
с ровесниками весь возрастной спад запишется на счёт травмы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FULL_SEASON_GP = 20      # меньше — это не сезон, а эпизодические выходы
RETURN_THRESHOLD = 0.90  # «вернулся» = не ниже 90% от своего же уровня
AGE_WINDOW = 1.0         # ±1 год для подбора контроля
MIN_WINDOW = 3.0         # ±3 минуты за игру


def add_metrics(stats: pd.DataFrame) -> pd.DataFrame:
    """Минуты и продуктивность, приведённая к 36 минутам на площадке."""
    st = stats.copy()
    st["min_pg"] = st.MIN
    played = st.MIN > 0
    st["pra36"] = np.where(
        played, (st.PTS + st.REB + st.AST) * 36.0 / st.MIN.where(played, np.nan), np.nan
    )
    return st


def first_full_season_after(stats: pd.DataFrame, name: str, season: int) -> pd.Series | None:
    """
    Первый сезон после травмы, в котором игрок реально играл.

    Сезон возвращения часто обрывочный: вышел в марте на четыре матча.
    Нас интересует момент, когда он снова в обойме, поэтому нужен порог по играм.
    """
    later = stats[
        (stats.name == name) & (stats.season > season) & (stats.GP >= FULL_SEASON_GP)
    ].sort_values("season")
    return later.iloc[0] if len(later) else None


def build_cases(episodes: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Одна строка — один эпизод тяжёлой травмы с базой до и исходом после.

    Игроки без полноценного сезона до травмы выбывают: сравнивать не с чем.
    Игроки без полноценного сезона после НЕ выбывают — это и есть результат
    «не вернулся», и выбрасывать их значило бы повторить вчерашнюю ошибку
    с ахиллами, когда фильтр съел ровно тот сигнал, который искали.
    """
    rows = []
    for ep in episodes.itertuples(index=False):
        before = stats[
            (stats.name == ep.name)
            & (stats.season == ep.season - 1)
            & (stats.GP >= FULL_SEASON_GP)
        ]
        if before.empty:
            continue
        b = before.iloc[0]
        a = first_full_season_after(stats, ep.name, ep.season)

        rows.append(
            {
                "player": ep.player,
                "name": ep.name,
                "injury": ep.injury,
                "season": ep.season,
                "days_out": ep.days,
                "age": b.AGE,
                "min_before": b.min_pg,
                "pra36_before": b.pra36,
                "returned": a is not None,
                "season_after": a.season if a is not None else np.nan,
                "seasons_lost": (a.season - ep.season) if a is not None else np.nan,
                "min_after": a.min_pg if a is not None else np.nan,
                "pra36_after": a.pra36 if a is not None else np.nan,
            }
        )

    cases = pd.DataFrame(rows)
    cases["min_ratio"] = cases.min_after / cases.min_before
    cases["pra36_ratio"] = cases.pra36_after / cases.pra36_before
    cases["back_minutes"] = cases.min_ratio >= RETURN_THRESHOLD
    cases["back_production"] = cases.pra36_ratio >= RETURN_THRESHOLD
    cases["back_fully"] = cases.back_minutes & cases.back_production
    return cases


def build_controls(cases: pd.DataFrame, stats: pd.DataFrame,
                   injured_names: set[str]) -> pd.DataFrame:
    """
    Для каждого случая — ровесники того же игрового веса, без тяжёлых травм.

    Контроль берётся из того же сезона, что и база пострадавшего, и
    прослеживается через тот же промежуток сезонов. Так возрастной спад
    и общий дрейф карьеры вычитаются сами собой.
    """
    clean = stats[~stats.name.isin(injured_names)]
    rows = []

    for c in cases.itertuples(index=False):
        if not c.returned:
            continue
        pool = clean[
            (clean.season == c.season - 1)
            & (clean.GP >= FULL_SEASON_GP)
            & (clean.AGE.sub(c.age).abs() <= AGE_WINDOW)
            & (clean.min_pg.sub(c.min_before).abs() <= MIN_WINDOW)
        ]
        if pool.empty:
            continue

        gap = int(c.season_after - c.season)
        later = clean[
            (clean.name.isin(pool.name)) & (clean.season == c.season - 1 + gap + 1)
        ].set_index("name")
        pool = pool.set_index("name")
        common = pool.index.intersection(later.index)
        if common.empty:
            continue

        rows.append(
            {
                "case": c.player,
                "injury": c.injury,
                "n_controls": len(common),
                "min_ratio": float(
                    (later.loc[common, "min_pg"] / pool.loc[common, "min_pg"]).median()
                ),
                "pra36_ratio": float(
                    (later.loc[common, "pra36"] / pool.loc[common, "pra36"]).median()
                ),
            }
        )

    ctrl = pd.DataFrame(rows)
    ctrl["back_minutes"] = ctrl.min_ratio >= RETURN_THRESHOLD
    ctrl["back_production"] = ctrl.pra36_ratio >= RETURN_THRESHOLD
    return ctrl
