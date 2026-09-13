"""
Подготовка данных о травмах NBA (2010-2020).

Исходные данные — лог транзакций Pro Sports Transactions: каждая строка это
событие «игрок выбыл» (Relinquished) или «игрок вернулся» (Acquired).
Медицинских записей здесь нет — есть только движение игроков по составу.

Модуль собирает из этих событий *эпизоды отсутствия* и оценивает,
сколько игровых дней стоила каждая травма.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Окно активного сезона NBA: регулярка стартует во второй половине октября,
# плей-офф заканчивается в середине июня. Дни между 15.06 и 15.10 не считаем —
# травма, полученная в апреле, не «стоит» команде июльских каникул.
SEASON_START = (10, 15)
SEASON_END = (6, 15)

# Признаки структурного повреждения. Нужны, чтобы отделить разрыв ахиллова
# сухожилия от тендинита: в исходном тексте и то и другое — просто "achilles".
SEVERE = r"torn|tear|ruptur|surgery|repair"

# Порядок важен: правила проверяются сверху вниз, первое совпадение выигрывает.
# Поэтому «разрыв мениска» стоит выше generic-правила «травма колена».
INJURY_RULES: list[tuple[str, str, bool]] = [
    ("Разрыв ахиллова сухожилия",       r"achilles",                    True),
    ("Разрыв ПКС (ACL)",                r"\bacl\b|anterior cruciate",   False),
    ("Разрыв мениска",                  r"meniscus",                    False),
    # Внимание: сюда попадают и разрывы, и растяжения. Проверка по тексту
    # показала, что признак разрыва есть лишь у 4 эпизодов из 45, поэтому
    # категория называется «повреждение», а не «разрыв».
    ("Повреждение связок колена (MCL/PCL)", r"\bmcl\b|\bpcl\b|collateral", False),
    ("Перелом",                         r"fractur|broken",              False),
    ("Плантарный фасциит",              r"plantar",                     False),
    ("Сотрясение мозга",                r"concussion",                  False),
    ("Травма подколенного сухожилия",   r"hamstring",                   False),
    ("Травма паха / приводящих",        r"groin|adductor",              False),
    ("Травма икроножной мышцы",         r"\bcalf\b",                    False),
    ("Травма квадрицепса",              r"quad(?:riceps)?\b",           False),
    ("Травма плеча",                    r"shoulder",                    False),
    ("Растяжение голеностопа",          r"ankle",                       False),
    ("Травма спины",                    r"\bback\b|lumbar|spasm",       False),
    ("Травма бедра / таза",             r"\bhip\b",                     False),
    ("Травма колена (прочее)",          r"knee",                        False),
    ("Травма кисти / пальца",           r"hand|finger|thumb|wrist",     False),
    ("Травма стопы",                    r"foot|toe|metatars",           False),
]

# Пропуски не по причине травмы. В исходнике они лежат вперемешку с травмами.
NOT_INJURY = (
    r"illness|\brest\b|personal|flu\b|virus|suspend|"
    r"conditioning|g league|assign|maternity|bereav"
)

# Эпизоды длиннее этого — почти всегда склейка двух разных историй
# (игрок ушёл из лиги и вернулся через год). Отсекаем.
MAX_INSEASON_DAYS = 400


def load_events(path: str | Path) -> pd.DataFrame:
    """Читает сырой CSV и приводит его к виду «одно событие — одна строка»."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[~(df.Acquired.isna() & df.Relinquished.isna())].copy()

    # Имя игрока лежит в одной из двух колонок; заодно чистим хвосты вида "(C)".
    name = df.Acquired.fillna(df.Relinquished).astype(str)
    df["player"] = (
        name.str.replace(r"\s*\(.*?\)\s*$", "", regex=True).str.strip().str.strip('"')
    )
    df["event"] = np.where(df.Acquired.notna(), "back", "out")
    return df.sort_values(["player", "Date"]).reset_index(drop=True)


def build_episodes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Склеивает события в эпизоды отсутствия.

    Логика простая: для каждого игрока идём по времени. Первое «выбыл»
    открывает эпизод, следующее «вернулся» — закрывает. Повторные «выбыл»
    внутри открытого эпизода не создают новый, а дописывают описание —
    так и бывает в реальности: сначала «day-to-day», через неделю
    «placed on IL», потом «surgery».

    Эпизод без возврата (игрок так и не вернулся в пределах данных)
    помечается end = NaT и в расчёт длительности не идёт.
    """
    episodes: list[dict] = []

    for player, group in events.groupby("player", sort=False):
        current: dict | None = None
        for row in group.itertuples(index=False):
            if row.event == "out":
                if current is None:
                    current = {"player": player, "start": row.Date,
                               "notes": [str(row.Notes)]}
                else:
                    current["notes"].append(str(row.Notes))
            elif current is not None:
                episodes.append({**current, "end": row.Date, "team": row.Team})
                current = None
        if current is not None:
            episodes.append({**current, "end": pd.NaT, "team": None})

    ep = pd.DataFrame(episodes)
    ep["text"] = ep.notes.map(lambda notes: " | ".join(notes).lower())
    ep["calendar_days"] = (ep.end - ep.start).dt.days
    return ep


def in_season_days(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Сколько дней отсутствия пришлось на активный сезон.

    Календарная разница завышает всё, что задевает межсезонье: разрыв связок
    в марте и такой же разрыв в октябре дают одинаковые 9 месяцев по календарю,
    но пропущенных матчей в них совсем разное количество.
    """
    if pd.isna(start) or pd.isna(end) or end < start:
        return np.nan

    total = 0
    for year in range(start.year - 1, end.year + 1):
        window_start = pd.Timestamp(year, *SEASON_START)
        window_end = pd.Timestamp(year + 1, *SEASON_END)
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).days
    return total


def classify(text: str) -> str | None:
    """Достаёт тип травмы из свободного текста. None — не распознали."""
    for label, pattern, needs_severe in INJURY_RULES:
        if re.search(pattern, text):
            if needs_severe and not re.search(SEVERE, text):
                return None
            return label
    return None


def prepare(path: str | Path) -> pd.DataFrame:
    """Полный путь от сырого CSV до таблицы размеченных эпизодов."""
    ep = build_episodes(load_events(path))
    ep = ep[ep.calendar_days.notna() & (ep.calendar_days >= 0)].copy()

    ep["days"] = [in_season_days(a, b) for a, b in zip(ep.start, ep.end)]
    # days == 0 — служебные записи «выбыл и вернулся в тот же день»,
    # реального пропуска за ними нет.
    ep = ep[(ep.days > 0) & (ep.days <= MAX_INSEASON_DAYS)].copy()

    ep["injury"] = ep.text.map(classify)
    ep["is_injury"] = ~ep.text.str.contains(NOT_INJURY, regex=True)

    ep["season"] = ep.start.map(lambda d: d.year if d.month >= 10 else d.year - 1)
    return ep.reset_index(drop=True)


def summary(ep: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Сводка по типам травм. Категории меньше min_n эпизодов отбрасываем."""
    clean = ep[ep.injury.notna() & ep.is_injury]
    table = clean.groupby("injury").days.agg(
        n="size",
        median="median",
        mean="mean",
        p90=lambda s: s.quantile(0.90),
    )
    return table[table.n >= min_n].sort_values("median", ascending=False).round(1)
