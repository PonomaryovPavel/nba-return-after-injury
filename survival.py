"""
Анализ выживаемости: время до возвращения в состав после тяжёлой травмы.

Почему именно выживаемость, а не доли. Часть игроков не вернулась вообще —
у них исход не наблюдался, но это не «нет данных», а «событие ещё не наступило
к моменту, когда данные кончились». Такое наблюдение называется цензурированным.
Считать по ним долю нельзя: выбросишь — завысишь скорость возвращения, засчитаешь
как невозврат — занизишь. Оценка Каплана–Мейера обращается с ними корректно.

Горизонт наблюдения ограничен административно: 400 игровых дней, примерно два
с половиной сезона. Кто не вернулся к этому сроку, цензурируется на горизонте.
Без такого потолка в выборку попадают склейки, где игрок ушёл из лиги и вернулся
спустя годы, — формально это «возвращение», по смыслу другое событие.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

HORIZON_DAYS = 400          # горизонт наблюдения, игровых дней
DATA_END = pd.Timestamp("2020-10-06")

SEVERE = [
    "Разрыв ахиллова сухожилия",
    "Разрыв ПКС (ACL)",
    "Разрыв мениска",
    "Повреждение связок колена (MCL/PCL)",
    "Перелом",
]


def build_cohort(episodes: pd.DataFrame, stats: pd.DataFrame,
                 in_season_days) -> pd.DataFrame:
    """
    Когорта для анализа: одна строка — один эпизод тяжёлой травмы.

    time  — игровых дней от травмы до возвращения либо до конца наблюдения,
    event — 1, если возвращение состоялось в пределах горизонта, иначе 0.
    """
    c = episodes[episodes.injury.isin(SEVERE) & episodes.is_injury].copy()
    c["event"] = c.end.notna().astype(int)
    c["stop"] = c.end.fillna(DATA_END)
    c["time"] = [in_season_days(a, b) for a, b in zip(c.start, c.stop)]
    c = c[c.time > 0].copy()

    # Административное цензурирование на горизонте.
    beyond = c.time > HORIZON_DAYS
    c.loc[beyond, "event"] = 0
    c.loc[beyond, "time"] = HORIZON_DAYS

    # Ковариаты из сезона, предшествующего травме.
    base = stats[["name", "season", "AGE", "min_pg"]].copy()
    base["season"] = base.season + 1          # сезон травмы
    c = c.merge(base, on=["name", "season"], how="left")
    c = c.rename(columns={"AGE": "age", "min_pg": "min_before"})
    return c.reset_index(drop=True)


def km_curve(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Каплан–Мейер с 95% доверительным интервалом (лог-лог преобразование)."""
    sf = SurvfuncRight(np.asarray(time, float), np.asarray(event, int))
    s = np.asarray(sf.surv_prob, float)
    se = np.asarray(sf.surv_prob_se, float)

    # Доверительный интервал строится на шкале log(-log S) и переносится обратно:
    # так границы гарантированно остаются в пределах от нуля до единицы.
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(-np.log(s))
        se_ll = se / (s * np.abs(np.log(s)))
        lo = np.exp(-np.exp(ll + 1.96 * se_ll))
        hi = np.exp(-np.exp(ll - 1.96 * se_ll))

    return pd.DataFrame({
        "time": np.asarray(sf.surv_times, float),
        "surv": s,
        "lo": np.clip(np.nan_to_num(lo, nan=0.0), 0, 1),
        "hi": np.clip(np.nan_to_num(hi, nan=1.0), 0, 1),
        "at_risk": np.asarray(sf.n_risk, float),
    })


def median_return_time(time: np.ndarray, event: np.ndarray) -> float:
    """Медиана по Каплану–Мейеру: момент, когда доля невернувшихся падает до 0.5."""
    km = km_curve(time, event)
    reached = km[km.surv <= 0.5]
    return float(reached.time.iloc[0]) if len(reached) else np.nan


def share_returned_by(time: np.ndarray, event: np.ndarray, day: float) -> tuple:
    """Доля вернувшихся к указанному дню, с доверительным интервалом."""
    km = km_curve(time, event)
    upto = km[km.time <= day]
    if upto.empty:
        return 0.0, 0.0, 0.0
    row = upto.iloc[-1]
    return 1 - row.surv, 1 - row.hi, 1 - row.lo


def logrank(cohort: pd.DataFrame, group_col: str = "injury") -> tuple:
    """Проверка гипотезы, что кривые возвращения у групп одинаковы."""
    chi, p = survdiff(cohort.time.values, cohort.event.values,
                      cohort[group_col].values)
    return float(chi), float(p)


def cox(cohort: pd.DataFrame, formula: str) -> PHReg:
    """Регрессия Кокса. Отношение рисков выше единицы = возвращение быстрее."""
    d = cohort.dropna(subset=["age", "min_before"]).copy()
    return PHReg.from_formula(formula, d, status="event").fit()


def cox_table(model: PHReg) -> pd.DataFrame:
    """Коэффициенты в читаемом виде: отношение рисков и интервал."""
    params = np.asarray(model.params, float)
    ci = np.asarray(model.conf_int(), float)
    return pd.DataFrame({
        "HR": np.exp(params).round(3),
        "CI_low": np.exp(ci[:, 0]).round(3),
        "CI_high": np.exp(ci[:, 1]).round(3),
        "p": np.asarray(model.pvalues, float).round(4),
    }, index=list(model.model.exog_names))


def ph_assumption_check(cohort: pd.DataFrame, group_col: str = "injury") -> pd.DataFrame:
    """
    Проверка допущения пропорциональных рисков.

    Если оно выполняется, кривые log(-log S(t)) у групп идут параллельно.
    Возвращает точки этих кривых, чтобы посмотреть глазами, а не поверить на слово.
    """
    rows = []
    for name, g in cohort.groupby(group_col):
        km = km_curve(g.time.values, g.event.values)
        km = km[(km.surv > 0.01) & (km.surv < 0.99) & (km.time > 0)]
        rows.append(pd.DataFrame({
            "group": name,
            "log_time": np.log(km.time),
            "loglog_surv": np.log(-np.log(km.surv)),
        }))
    return pd.concat(rows, ignore_index=True)
