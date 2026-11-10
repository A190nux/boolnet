"""
Simple reference learners for {input_tuple: label} Boolean data. Each is
built from a training dict and is callable on an input tuple, so
bool_net.evaluate(model, train, test, verbose=False) works on them.

No dependencies beyond the standard library (sklearn isn't assumed).
"""

import math


class MajorityLabel:
    """Always predict the most common training label (ties → 0)."""

    def __init__(self, data: dict):
        self.y = int(2 * sum(data.values()) > len(data))

    def __call__(self, x):
        return self.y


class NearestNeighbour:
    """Label of the nearest training row in Hamming distance (majority vote
    among equally near rows; a tie in that vote → the most common label)."""

    def __init__(self, data: dict):
        self.data = data
        self.default = MajorityLabel(data).y

    def __call__(self, x):
        if x in self.data:
            return self.data[x]
        best, votes = None, []
        for xt, y in self.data.items():
            d = sum(a != b for a, b in zip(x, xt))
            if best is None or d < best:
                best, votes = d, [y]
            elif d == best:
                votes.append(y)
        s = 2 * sum(votes)
        return 1 if s > len(votes) else 0 if s < len(votes) else self.default


def _entropy(pos, n):
    if n == 0 or pos == 0 or pos == n:
        return 0.0
    p = pos / n
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class DecisionTree:
    """
    ID3-style binary decision tree over 0/1 features: split on the feature
    with the highest information gain, recurse until a node is pure, no
    feature separates the rows, or max_depth is reached. Leaves predict their
    majority label (a tie falls back to the parent's majority).

    With one-hot inputs, a split on bit "a5=1" is the usual categorical
    test "a5 == 1 vs a5 != 1".
    """

    def __init__(self, data: dict, max_depth: int = None):
        self.max_depth = max_depth
        rows = list(data.items())
        self.n_features = len(rows[0][0]) if rows else 0
        default = MajorityLabel(data).y
        self.root = self._build(rows, 0, default)

    def _build(self, rows, depth, parent_label):
        n = len(rows)
        pos = sum(y for _, y in rows)
        if n == 0:
            return ('leaf', parent_label)
        label = 1 if 2 * pos > n else 0 if 2 * pos < n else parent_label
        if pos == 0 or pos == n or (self.max_depth is not None and depth >= self.max_depth):
            return ('leaf', label)
        h = _entropy(pos, n)
        # Any feature that actually separates the rows is allowed, even at
        # zero gain: an impure node can have every split at zero gain (an
        # XOR-like pattern), and stopping there would leave training rows
        # misclassified. Ties keep the lowest feature index.
        best_f, best_gain = None, -1.0
        for f in range(self.n_features):
            n1 = p1 = 0
            for x, y in rows:
                if x[f]:
                    n1 += 1
                    p1 += y
            n0, p0 = n - n1, pos - p1
            if n1 == 0 or n0 == 0:
                continue
            gain = h - (n1 / n) * _entropy(p1, n1) - (n0 / n) * _entropy(p0, n0)
            if gain > best_gain:
                best_f, best_gain = f, gain
        if best_f is None:
            return ('leaf', label)
        left = [(x, y) for x, y in rows if not x[best_f]]
        right = [(x, y) for x, y in rows if x[best_f]]
        return ('split', best_f,
                self._build(left, depth + 1, label),
                self._build(right, depth + 1, label))

    def __call__(self, x):
        node = self.root
        while node[0] == 'split':
            node = node[3] if x[node[1]] else node[2]
        return node[1]

    def n_leaves(self, node=None) -> int:
        node = node or self.root
        if node[0] == 'leaf':
            return 1
        return self.n_leaves(node[2]) + self.n_leaves(node[3])
