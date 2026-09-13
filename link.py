"""
Связывание эпизодов травм с посезонной статистикой игроков NBA.

Два источника пишут имена по-разному: Pro Sports Transactions ведёт их так,
как они появлялись в новостях клуба, официальная статистика — так, как игрок
записан в лиге. Отсюда «Ron Artest» против «Metta World Peace».
"""

from __future__ import annotations

import re

import pandas as pd

# Разрывы и переломы. Лёгкие повреждения на уровень игры не влияют,
# и смешивать их с тяжёлыми нельзя — получится каша.
SEVERE_INJURIES = [
    "Разрыв ахиллова сухожилия",
    "Разрыв ПКС (ACL)",
    "Разрыв мениска",
    "Разрыв связок колена (MCL/PCL)",
    "Перелом",
]

# Слева — как игрок записан в логе транзакций, справа — как в статистике лиги.
# Смена имени (Artest, Kanter), полное имя вместо краткого, диакритика.
ALIASES = {
    "alex ajinca": "alexis ajinca",
    "enes kanter": "enes freedom",
    "jeff taylor": "jeffery taylor",
    "jose barea": "jj barea",
    "malcom lee": "malcolm lee",
    "nene hilario": "nene",
    "ognen kuzmic": "ognjen kuzmic",
    "ron artest": "metta world peace",
    "wes matthews": "wesley matthews",
}

_DIACRITICS = str.maketrans("çččćšžōéáíúñ", "ccccszoeaiun")


def normalize(name: str) -> str:
    """Приводит имя к виду, в котором два источника сравнимы."""
    s = str(name).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\.?$", "", s).strip()
    s = re.sub(r"[.'`\-]", "", s)
    s = s.translate(_DIACRITICS)
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)


def load_stats(path) -> pd.DataFrame:
    """Посезонная статистика: одна строка — игрок в одном сезоне."""
    st = pd.read_csv(path)
    st["season"] = st.SEASON.astype(str).str.slice(0, 4).astype(int)
    st["name"] = st.PLAYER_NAME.map(normalize)
    return st


def severe_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Только тяжёлые повреждения, с нормализованным именем и сезоном травмы."""
    sev = episodes[
        episodes.injury.isin(SEVERE_INJURIES) & episodes.is_injury
    ].copy()
    sev["name"] = sev.player.map(normalize)
    return sev


def match_report(sev: pd.DataFrame, stats: pd.DataFrame) -> dict:
    """Сколько эпизодов нашло своего игрока в статистике."""
    known = set(stats.name)
    hit = sev.name.isin(known)
    return {
        "episodes": len(sev),
        "matched": int(hit.sum()),
        "rate": float(hit.mean()),
        "unmatched": sorted(sev.loc[~hit, "player"].unique()),
    }
