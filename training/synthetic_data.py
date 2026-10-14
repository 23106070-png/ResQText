"""Generate the SYNTHETIC development dataset (sample_messages.csv + duplicate_pairs.csv).

Everything here is invented template text for demos and tests. Results measured on it say NOTHING about
real-world performance. To avoid template leakage, templates are divided into train/val/test pools BEFORE
messages are generated, so a test sentence pattern never appears in training.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

import config

R, M, F, S, I, G = config.LABELS
TEMPLATES = {
    R: [
        "We are trapped on the first floor in {loc}. Water is rising, please send a boat.",
        "{n} people stuck on the roof of a building near {loc}. Need rescue urgently.",
        "Our family is stranded in {loc}, the water has reached the second floor. Please help us get out.",
        "Elderly couple trapped in their house at {loc}, they cannot walk. Need a rescue team.",
        "Help! Several people are cut off by flood water in {loc}, no way out.",
        "Rescue needed at {loc}, {n} persons stuck in a car surrounded by water.",
        "Boat needed in {loc}, {n} villagers stuck on the terrace since {when}.",
        "Someone is drowning near the river bank at {loc}. Send NDRF immediately.",
        "Landslide has buried a house at {loc}, people may be trapped under the debris.",
        "Kids are alone and trapped in a flooded building in {loc}, please send rescuers.",
    ],
    M: [
        "A pregnant woman in {loc} is in labour and we cannot reach any hospital. Need an ambulance.",
        "{n} people injured after a wall collapsed in {loc}. Need doctors and first aid.",
        "My father needs insulin and we have run out. We are in {loc}.",
        "Patient on oxygen support at {loc}, the cylinder will finish soon. Urgent.",
        "Child has high fever and is vomiting, need a doctor in {loc}.",
        "Elderly man had a heart attack in {loc}, the ambulance cannot get through the water.",
        "We need medicines for diabetes and blood pressure at the {loc} relief point.",
        "Several people have deep cuts and fractures near {loc}, need a medical team.",
        "Many children have fever and diarrhoea in {loc}, please send a medical camp.",
        "Injured man bleeding badly at {loc}, please send an ambulance.",
    ],
    F: [
        "No drinking water since {when} in {loc}. {n} families are thirsty.",
        "We need food packets for {n} people, nothing to eat since yesterday in {loc}.",
        "Children are hungry in {loc}, we need rice and milk.",
        "Clean water is not available at {loc}, tap water is contaminated.",
        "Please send bottled water and biscuits to {loc}, {n} people are starving.",
        "Food supplies finished in our building at {loc}, {n} elderly residents need meals.",
        "Need baby food and drinking water in {loc}.",
        "Ration shortage at {loc}, families have had no meal today.",
        "Requesting water tankers at {loc}, no safe water for two days.",
        "Hungry since morning, {n} people at {loc} need food urgently.",
    ],
    S: [
        "{n} families lost their homes in {loc} and need a place to stay tonight.",
        "Please open a relief camp near {loc}, people are sleeping on the road.",
        "We need blankets and tents at {loc}, it is cold and raining.",
        "The school in {loc} is being used as a shelter, it is full. Need more space.",
        "Evacuation needed for low-lying houses in {loc}, water is entering.",
        "Need a bus to evacuate {n} people from {loc} to a safe place.",
        "Our house is flooded, we need temporary shelter in {loc}.",
        "People at {loc} want to move to higher ground, need transport and shelter.",
        "Homeless families at {loc} need dry clothes and mattresses at the relief camp.",
        "Please evacuate the old age home in {loc}, residents cannot stay.",
    ],
    I: [
        "The bridge near {loc} has collapsed, the road is blocked.",
        "No electricity in {loc} since {when}, power lines fell.",
        "Main road at {loc} is submerged, vehicles cannot pass.",
        "Landslide blocked the highway near {loc}.",
        "Mobile network is down in {loc}, towers damaged.",
        "Water pipeline burst in {loc}, the road caved in.",
        "Electric pole fallen on the road at {loc}, dangerous.",
        "Drainage overflowed and the underpass at {loc} is completely flooded.",
        "A tree fell on power lines in {loc}, the street is blocked.",
        "Railway tracks at {loc} are under water, trains cancelled.",
    ],
    G: [
        "Water level near the {loc} bridge is rising slowly, sharing for awareness.",
        "Is the road to {loc} open for traffic?",
        "The flood relief helpline number is active, please share.",
        "Heavy rain continues in {loc} since {when}.",
        "Volunteers are gathering at {loc} to help with distribution. Contact the coordinator.",
        "The weather department has issued a red alert for {loc} today.",
        "Please confirm if schools in {loc} will be closed tomorrow.",
        "Rain has reduced in {loc}, the situation is improving.",
        "We are safe in {loc}, thank you everyone for the support.",
        "Relief trucks have reached {loc} and distribution has started.",
    ],
}
NS = ["3", "five", "8", "12", "20", "two", "6", "40", "15", "four"]
WHENS = ["6 pm", "yesterday", "last night", "this morning", "midnight", "2 pm"]
LOCS = ["Thane", "Kalwa", "Mumbra", "Mulund", "Bhandup", "Vikhroli", "Powai", "Ghatkopar", "Kurla", "Chembur",
        "Sion", "Dadar", "Andheri", "Vashi", "Panvel", "Dombivli", "Kalyan", "Ulhasnagar", "Bhiwandi", "Badlapur"]
POOL_OF = lambda i: "train" if i < 6 else ("val" if i < 8 else "test")  # noqa: E731
START = datetime(2024, 7, 26, 6, 0)  # fictional scenario start, clearly simulated


def add_typos(text: str, rng: random.Random, p: float = 0.12) -> str:
    words = text.split()
    for i, w in enumerate(words):
        if len(w) >= 5 and w.isalpha() and rng.random() < p:
            k = rng.randrange(1, len(w) - 1)
            words[i] = w[:k] + w[k + 1:] if rng.random() < 0.5 else w[:k] + w[k] + w[k:]
    return " ".join(words)


def _fill(t: str, rng: random.Random, loc: str | None = None) -> str:
    return t.format(loc=loc or rng.choice(LOCS), n=rng.choice(NS), when=rng.choice(WHENS))


def make_messages(n_per_split: dict | None = None, seed: int = 7) -> pd.DataFrame:
    n_per_split = n_per_split or {"train": 900, "val": 180, "test": 300}
    rng, rows = random.Random(seed), []
    for split, n in n_per_split.items():
        pool = {lab: [t for i, t in enumerate(ts) if POOL_OF(i) == split] for lab, ts in TEMPLATES.items()}
        for _ in range(n):
            k = 2 if rng.random() < 0.3 else 1
            labs = rng.sample(config.LABELS, k)
            loc = rng.choice(LOCS)  # multi-need messages share one location
            text = " ".join(_fill(rng.choice(pool[l]), rng, loc) for l in labs)
            if rng.random() < 0.18:
                text = add_typos(text, rng)
            # later hours get more medical / shelter reports so the trend charts have something to show
            hours = rng.random() * 72
            if (M in labs or S in labs) and rng.random() < 0.5:
                hours = 30 + rng.random() * 42
            rows.append({"text": text, "timestamp": (START + timedelta(hours=hours)).isoformat(timespec="minutes"),
                         "labels": "|".join(labs), "split": split})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "message_id", [f"SIM-{i + 1:04d}" for i in range(len(df))])
    df["source"] = "synthetic"
    df["is_simulated"] = 1
    return df


HAND_PAIRS = [  # hand-written paraphrases of the same incident (harder positives)
    ("Five people stuck on a rooftop in Kalwa, need a boat urgently.", "Boat required at Kalwa, 5 persons stranded on the terrace.", "Kalwa", "Kalwa"),
    ("Pregnant woman in labour at Mulund, no ambulance reachable.", "Need ambulance in Mulund for a woman about to deliver a baby.", "Mulund", "Mulund"),
    ("No drinking water in Kurla since last night.", "Kurla has had no clean water since yesterday evening.", "Kurla", "Kurla"),
    ("Bridge near Dadar has collapsed and the road is closed.", "Dadar bridge broke down, traffic cannot pass.", "Dadar", "Dadar"),
    ("We need blankets and tents at the Thane relief camp.", "Thane camp requires tents and blankets, it is cold.", "Thane", "Thane"),
    ("Elderly man had a heart attack in Sion, need a doctor.", "Old man with chest pain and heart attack in Sion, send a doctor.", "Sion", "Sion"),
    ("Need food packets for 20 people in Vashi.", "Twenty people in Vashi have nothing to eat, send food.", "Vashi", "Vashi"),
    ("Electric pole fallen on the road at Powai.", "A power pole has fallen across the road in Powai.", "Powai", "Powai"),
]
HAND_NEG = [
    ("Five people stuck on a rooftop in Kalwa, need a boat urgently.", "Five people stuck on a rooftop in Mumbra, need a boat urgently.", "Kalwa", "Mumbra"),
    ("No drinking water in Kurla since last night.", "Pregnant woman in labour at Kurla, no ambulance reachable.", "Kurla", "Kurla"),
    ("Need food packets for 20 people in Vashi.", "Need food packets for 20 people in Panvel.", "Vashi", "Panvel"),
    ("Bridge near Dadar has collapsed.", "Is the road to Dadar open for traffic?", "Dadar", "Dadar"),
    ("Need blankets at the Thane relief camp.", "Need blankets at the Kalyan relief camp.", "Thane", "Kalyan"),
    ("Elderly man had a heart attack in Sion.", "Elderly couple trapped in their house in Sion.", "Sion", "Sion"),
]


def make_pairs(seed: int = 11) -> pd.DataFrame:
    rng, rows = random.Random(seed), []
    for split in ("train", "val", "test"):
        for lab, ts in TEMPLATES.items():
            pool = [t for i, t in enumerate(ts) if POOL_OF(i) == split]
            for _ in range(6):
                t, loc = rng.choice(pool), rng.choice(LOCS)
                a = _fill(t, rng, loc)
                b = add_typos("Repeat report: " + a, rng, 0.2)
                rows.append((a, b, loc, loc, 1, "easy_positive", split))  # near-copy
                other = rng.choice([x for x in LOCS if x != loc])
                rows.append((a, a.replace(loc, other), loc, other, 0, "same_text_other_location", split))
                lab2 = rng.choice([x for x in config.LABELS if x != lab])
                t2 = rng.choice([x for i, x in enumerate(TEMPLATES[lab2]) if POOL_OF(i) == split])
                rows.append((a, _fill(t2, rng, loc), loc, loc, 0, "other_need_same_location", split))
    for i, (a, b, la, lb) in enumerate(HAND_PAIRS):
        rows.append((a, b, la, lb, 1, "paraphrase", ("val", "test")[i % 2]))
    for i, (a, b, la, lb) in enumerate(HAND_NEG):
        rows.append((a, b, la, lb, 0, "hand_negative", ("val", "test")[i % 2]))
    return pd.DataFrame(rows, columns=["text_a", "text_b", "location_a", "location_b", "is_duplicate", "pair_type", "split"])


if __name__ == "__main__":
    config.DATA_DIR.mkdir(exist_ok=True)
    msgs, pairs = make_messages(), make_pairs()
    msgs.to_csv(config.SAMPLE_CSV, index=False)
    pairs.to_csv(config.PAIRS_CSV, index=False)
    print(f"Wrote {len(msgs)} synthetic messages -> {config.SAMPLE_CSV}")
    print(f"Wrote {len(pairs)} synthetic pairs    -> {config.PAIRS_CSV}")
