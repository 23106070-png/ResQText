"""Dataset loading, cleaning, label normalisation and leakage-safe splitting.

CSV schema (see data/README.md): text, labels ("A|B"), optional timestamp, message_id, source, split.
Leakage protections: (1) exact/normalised duplicate texts are dropped BEFORE splitting, (2) a pre-assigned
`split` column (template-pool split) is respected, (3) the TF-IDF vectoriser is fitted inside the pipeline on
training data only, (4) val is for model selection, the test set is used once for the final report, and
(5) assert_no_leakage() fails loudly if any text appears in two splits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from nlp.preprocessing import normalize_text

R, M, F, S, I, G = config.LABELS
LABEL_ALIASES = {  # normalise spelling variants and other taxonomies onto our six classes
    "rescue": R, "rescue assistance": R, "rescue_volunteering_or_donation_effort": R, "missing_or_found_people": R,
    "medical": M, "medical assistance": M, "injured_or_dead_people": M,
    "food": F, "water": F, "food and water": F, "food and drinking water": F, "food_water": F,
    "shelter": S, "evacuation": S, "shelter and evacuation": S, "displaced_people_and_evacuations": S,
    "infrastructure": I, "infrastructure damage": I, "infrastructure_and_utility_damage": I,
    "general": G, "general information": G, "other_relevant_information": G, "caution_and_advice": G,
    "sympathy_and_support": G, "not_humanitarian": G, "requests_or_urgent_needs": None,  # too vague: skip
}
CANON = {l.lower(): l for l in config.LABELS}


class DatasetError(ValueError):
    pass


def normalize_labels(raw) -> list[str]:
    out = []
    for part in str(raw).replace(";", "|").replace(",", "|").split("|"):
        key = part.strip().lower()
        if not key:
            continue
        lab = CANON.get(key) or LABEL_ALIASES.get(key)
        if lab and lab not in out:
            out.append(lab)
    return out


def load_labelled_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read CSV: {exc}") from exc
    missing = {"text", "labels"} - set(df.columns)
    if missing:
        raise DatasetError(f"Missing required column(s): {sorted(missing)}. Required: text, labels")
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["label_list"] = df["labels"].map(normalize_labels)
    df = df[df["label_list"].map(len) > 0]
    df["norm"] = df["text"].map(normalize_text)
    df = df.drop_duplicates(subset="norm").reset_index(drop=True)  # leakage guard #1
    if df.empty:
        raise DatasetError("No usable rows after cleaning.")
    return df


def split_dataset(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    if "split" in df.columns and set(df["split"].dropna()) >= {"train", "val", "test"}:
        parts = {k: df[df["split"] == k].reset_index(drop=True) for k in ("train", "val", "test")}
    else:
        strat = df["label_list"].map(lambda l: l[0])
        try:
            train, rest = train_test_split(df, test_size=0.3, random_state=seed, stratify=strat)
            val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["label_list"].map(lambda l: l[0]))
        except ValueError:  # a class too small to stratify
            train, rest = train_test_split(df, test_size=0.3, random_state=seed)
            val, test = train_test_split(rest, test_size=0.5, random_state=seed)
        parts = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}
    assert_no_leakage(parts)
    return parts


def assert_no_leakage(parts: dict[str, pd.DataFrame]) -> None:
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(parts[a]["norm"]) & set(parts[b]["norm"])
            if shared:
                raise DatasetError(f"Data leakage: {len(shared)} identical text(s) in both {a} and {b}.")


def load_humaid(directory: Path | str) -> pd.DataFrame:
    """Load a manually downloaded HumAID event folder (tsv files with tweet_text + class_label).
    HumAID is NOT downloaded automatically: see data/README.md for access and licence checks."""
    directory = Path(directory)
    files = sorted(directory.rglob("*.tsv"))
    if not files:
        raise DatasetError(f"No .tsv files in {directory}. Download HumAID from https://crisisnlp.qcri.org/humaid_dataset first.")
    frames = []
    for f in files:
        d = pd.read_csv(f, sep="\t")
        cols = {c.lower(): c for c in d.columns}
        tcol, lcol = cols.get("tweet_text") or cols.get("text"), cols.get("class_label") or cols.get("label")
        if not tcol or not lcol:
            continue
        frames.append(pd.DataFrame({"text": d[tcol], "labels": d[lcol], "source": f.stem}))
    if not frames:
        raise DatasetError("TSV files found but none have tweet_text/class_label columns.")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else config.SAMPLE_CSV
    parts = split_dataset(load_labelled_csv(src))
    out = config.DATA_DIR / "processed"
    out.mkdir(exist_ok=True)
    for k, v in parts.items():
        v[["text", "labels"]].to_csv(out / f"{k}.csv", index=False)
        print(f"{k}: {len(v)} rows")
    print("No leakage detected. Wrote", out)
