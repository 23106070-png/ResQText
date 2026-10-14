"""Analytics over stored reports. Shared by the API and the dashboard."""
from __future__ import annotations

import pandas as pd

import config

CAVEAT = ("More reports does not prove a worse disaster: it can also reflect better phone coverage, a "
          "published helpline, media attention or people repeating the same request. Use these trends to "
          "decide where to look first, then verify on the ground.")


def explode_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["label"])
    return df.explode("labels").rename(columns={"labels": "label"}).dropna(subset=["label"])


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    e = explode_labels(df)
    counts = e["label"].value_counts() if not e.empty else pd.Series(dtype=int)
    return pd.DataFrame({"label": config.LABELS, "count": [int(counts.get(l, 0)) for l in config.LABELS]})


def by_location(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    known = df[~df["location_text"].str.lower().isin(["unknown", "needs verification"])] if not df.empty else df
    if known.empty:
        return pd.DataFrame(columns=["location", "count"])
    return known["location_text"].value_counts().head(top).rename_axis("location").reset_index(name="count")


def top_resources(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["resource", "count"])
    r = df.explode("resources").dropna(subset=["resources"])
    r = r[r["resources"] != "Unknown"]
    return r["resources"].value_counts().head(top).rename_axis("resource").reset_index(name="count")


def volume_over_time(df: pd.DataFrame, freq: str = "6h") -> pd.DataFrame:
    if df.empty or df["ts"].isna().all():
        return pd.DataFrame(columns=["ts", "count"])
    s = df.dropna(subset=["ts"]).set_index("ts").resample(freq).size()
    return s.rename("count").reset_index()


def category_trend(df: pd.DataFrame, freq: str = "6h") -> pd.DataFrame:
    e = explode_labels(df.dropna(subset=["ts"])) if not df.empty else df
    if e.empty:
        return pd.DataFrame(columns=["ts", "label", "count"])
    g = e.groupby([pd.Grouper(key="ts", freq=freq), "label"]).size().rename("count").reset_index()
    return g


def window_comparison(df: pd.DataFrame, hours: int = 12, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compare the latest `hours` of reports with the `hours` before it, per category.
    `now` defaults to the newest report time, so stored/simulated scenarios compare correctly."""
    cols = ["label", "previous", "current", "change", "change_pct"]
    d = df.dropna(subset=["ts"]) if not df.empty else df
    if d.empty:
        return pd.DataFrame(columns=cols)
    now = now or d["ts"].max()
    cur_start, prev_start = now - pd.Timedelta(hours=hours), now - pd.Timedelta(hours=2 * hours)
    e = explode_labels(d)
    cur = e[(e["ts"] > cur_start) & (e["ts"] <= now)]["label"].value_counts()
    prev = e[(e["ts"] > prev_start) & (e["ts"] <= cur_start)]["label"].value_counts()
    rows = []
    for l in config.LABELS:
        c, p = int(cur.get(l, 0)), int(prev.get(l, 0))
        rows.append({"label": l, "previous": p, "current": c, "change": c - p,
                     "change_pct": None if p == 0 else round(100 * (c - p) / p, 1)})
    return pd.DataFrame(rows, columns=cols)


def summary(df: pd.DataFrame, counts: dict) -> dict:
    return {
        "counts": counts,
        "by_category": by_category(df).to_dict("records"),
        "top_locations": by_location(df).to_dict("records"),
        "top_resources": top_resources(df).to_dict("records"),
        "window_comparison_12h": window_comparison(df, 12).to_dict("records"),
        "caveat": CAVEAT,
    }


def location_window_comparison(df: pd.DataFrame, hours: int = 24, top: int = 8) -> pd.DataFrame:
    """Reports per location: latest window vs the window before. Counts collected reports only."""
    cols = ["location", "previous", "current", "change"]
    d = df.dropna(subset=["ts"]) if not df.empty else df
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d[~d["location_text"].str.lower().isin(["unknown", "needs verification"])]
    if d.empty:
        return pd.DataFrame(columns=cols)
    now = d["ts"].max()
    cs, ps = now - pd.Timedelta(hours=hours), now - pd.Timedelta(hours=2 * hours)
    cur = d[(d["ts"] > cs) & (d["ts"] <= now)]["location_text"].value_counts()
    prev = d[(d["ts"] > ps) & (d["ts"] <= cs)]["location_text"].value_counts()
    rows = [{"location": l, "previous": int(prev.get(l, 0)), "current": int(cur.get(l, 0)), "change": int(cur.get(l, 0) - prev.get(l, 0))}
            for l in set(cur.index) | set(prev.index)]
    return pd.DataFrame(rows, columns=cols).sort_values(["current", "change"], ascending=False).head(top)
