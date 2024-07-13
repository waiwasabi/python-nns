from collections import defaultdict
import numpy as np


class Node:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)


# p (array-like): array of classes resulting corresponding to a partition.
def entr(p):
    b = np.bincount(p) / p.size
    return - np.sum((n := b[b > 0]) * np.log2(n))


# p (array-like): array of classes resulting corresponding to a partition.
def gini(p): return 1 - np.sum((np.bincount(p) / p.size)**2)


def split_num(c, y, metric=entr):
    c = np.sort(c)
    splits = [(a + b) / 2 for a, b in zip(c[:-1], c[1:])]
    entropies = [metric(y[c <= s]) + metric(y[c > s]) for s in splits]
    return np.min(entropies) / 2


def split_cat(c, y, metric=entr):
    return np.mean([metric(y[c == s]) for s in np.unique(c)])


def split_col(c, y, is_numeric, metric=entr):
    return split_num(c, y, metric) if is_numeric else split_cat(c, y, metric)


class InfoGain:
    @staticmethod
    def select(X, y, numeric):
        # for each column c: compute the entropy of each partition resulting from splitting y by c.
        parts = [split_col(c, y, numeric[i]) for i, c in enumerate(X.T)]
        return np.argmin(parts), np.min(parts)

    @staticmethod
    def get_split(c, y):
        splits = [np.mean([a, b]) for a, b in zip(c[:-1], c[1:])]
        entropies = [entr(y[c <= s]) + entr(y[c > s]) for s in splits]
        return splits[np.argmin(entropies)]


class Gini:
    @staticmethod
    def select(X, y, numeric):
        parts = [split_col(c, y, numeric[i], gini) for i, c in enumerate(X.T)]
        return np.argmin(parts), np.min(parts)

    @staticmethod
    def get_split(c, y):
        splits = [np.mean([a, b]) for a, b in zip(c[:-1], c[1:])]
        ginis = [gini(y[c <= s]) + gini(y[c > s]) for s in splits]
        return splits[np.argmin(ginis)]


def decision_tree_classifier(X, y, numeric=None, criterion=InfoGain, m=None, maj_stop=1, depth_stop=None):
    numeric = np.ones(X.shape[1], dtype=bool) if numeric is None else numeric
    y_maj = np.bincount(y).argmax()  # majority class in dataset
    default_node = Node(lambda _: y_maj)
    m = m if m is not None else lambda x: x.shape[1]

    def tree(X, y, n, depth=0):
        """
        Args:
            X (array-like): a slice of the original dataset
            y (array-like): labels corresponding to the rows of X. Assumed to be binary.
            n (array-like): boolean mask indicating which columns of X are numeric
        """
        if not y.size:
            return default_node

        root_node = Node(lambda _: part_maj)
        part_maj, n_maj = (b := np.bincount(y)).argmax(), b.max()
        if n_maj / y.size >= maj_stop or not X.size:
            return root_node

        if depth_stop is not None and depth >= depth_stop:
            return root_node

        mask = np.random.choice(np.arange(X.shape[1]), m(X), replace=False)
        idx, crit = criterion.select(X[:, mask], y, n[mask])

        c = mask[idx]
        col = X.T[c]

        if n[c]:
            cut = criterion.get_split(col, y)
            dl = tree(X[col <= cut], y[col <= cut], n, depth=depth + 1)
            dr = tree(X[col > cut], y[col > cut], n, depth=depth + 1)

            def f(x): return (dl if x[c] <= cut else dr)(x)
        else:
            cmask = np.arange(X.shape[1]) != c
            d = {i: tree(X[col == i][:, cmask], y[col == i], n[cmask], depth=depth + 1)
                 for i in np.unique(col)}
            d = defaultdict(lambda: root_node, d)
            def f(x): return d[x[c]](x[cmask])
        return Node(f)

    return tree(X, y, numeric)
