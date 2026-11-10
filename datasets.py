"""
Real datasets, encoded as Boolean truth-table fragments: {input_tuple: label}.

MONK's problems (Thrun et al., 1991) — UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/70/monk+s+problems (CC BY 4.0).
Files live in data/monks/ exactly as downloaded.
"""

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Attribute arities a1..a6, from data/monks/monks.names.
MONKS_ARITIES = (3, 3, 2, 3, 4, 2)

# Ground-truth concepts (monks.names §9), on the raw attribute values.
MONKS_CONCEPTS = {
    1: lambda a: int(a[0] == a[1] or a[4] == 1),
    2: lambda a: int(sum(v == 1 for v in a) == 2),
    3: lambda a: int((a[4] == 3 and a[3] == 1) or (a[4] != 4 and a[1] != 3)),
}


def one_hot(attrs: tuple, arities=MONKS_ARITIES) -> tuple:
    """Categorical values (1-based) → concatenated one-hot bits."""
    bits = []
    for v, k in zip(attrs, arities):
        bits.extend(int(v == c) for c in range(1, k + 1))
    return tuple(bits)


def feature_names(arities=MONKS_ARITIES) -> list:
    """Name of each one-hot bit, e.g. 'a1=2' — for reading learned formulas."""
    return [f"a{i+1}={c}" for i, k in enumerate(arities) for c in range(1, k + 1)]


def _read_monks_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            label = int(parts[0])
            attrs = tuple(int(v) for v in parts[1:7])
            rows.append((attrs, label))
    return rows


def load_monks(problem: int, data_dir: str = None):
    """
    Returns (train, test) dicts mapping a 17-bit one-hot input tuple to its
    label, for MONK's problem 1, 2 or 3, using the official split. The test
    set is the full 432-instance domain. MONK-3's TRAINING labels contain
    5% deliberate noise (test labels are clean).
    """
    if problem not in (1, 2, 3):
        raise ValueError("MONK's problem must be 1, 2 or 3")
    d = os.path.join(data_dir or DATA_DIR, "monks")
    out = []
    for part in ("train", "test"):
        rows = _read_monks_file(os.path.join(d, f"monks-{problem}.{part}"))
        data = {}
        for attrs, label in rows:
            x = one_hot(attrs)
            if x in data and data[x] != label:
                raise ValueError(f"conflicting labels for {attrs} in {part}")
            data[x] = label
        out.append(data)
    return tuple(out)
