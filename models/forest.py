from dt import *
import numpy as np
from tqdm import tqdm


def bootstrap(X, y):
    indices = np.random.choice(np.arange(nrows := X.shape[0]), size=nrows)
    return np.take(X, indices, axis=0), np.take(y, indices, axis=0)


def random_forest(X, y, n_trees=1, *args, **kwargs):
    def m(X): return np.floor(np.sqrt(X.shape[1])).astype(int)
    trees = [decision_tree_classifier(*bootstrap(X, y), m=m, *args, **kwargs)
             for _ in range(n_trees)]

    return lambda x: np.bincount([t(x) for t in trees]).argmax()


def binary_prec(y_pred, y_true, _):
    s = np.sum(y_pred == 1)
    return np.sum([p == a for p, a in zip(y_pred, y_true) if p == 1]) / s if s != 0 else 0

def binary_recall(y_pred, y_true, _):
    s = np.sum(y_true == 1)
    return np.sum([p == a for p, a in zip(y_pred, y_true) if p == 1]) / s if s != 0 else 0


def multiclass_prec(y_pred, y_true, i):
    s = np.sum(y_pred == i)
    return np.sum([p == a for p, a in zip(y_pred, y_true) if p == i]) / s if s != 0 else 0


def multiclass_recall(y_pred, y_true, i):
    s = np.sum(y_true == i)
    return np.sum([p == a for p, a in zip(y_pred, y_true) if p == i]) / s if s != 0 else 0


def rf_eval(X_tr, y_tr, X_te, y_te, pr=multiclass_prec, re=multiclass_recall, *args, **kwargs):
    rf = random_forest(X_tr, y_tr, *args, **kwargs)
    y_pred = [rf(v) for v in X_te]

    prec, recall, f1 = [], [], []
    for i in np.unique(y_te):
        prec.append(pr(y_pred, y_te, i))
        recall.append(re(y_pred, y_te, i))
        f1.append(2 * prec[-1] * recall[-1] / (prec[-1] + recall[-1])
                  ) if prec[-1] + recall[-1] != 0 else f1.append(0)

    acc = np.sum([p == a for p, a in zip(y_pred, y_te)]) / y_te.size

    return [acc, np.mean(prec), np.mean(recall), np.mean(f1)]


def stratified_k_fold_eval(X, y, k, n_trees, *args, **kwargs):
    idx = np.arange(y.size)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    y_groups = [np.array_split(idx[y == i], k) for i in np.unique(y)]
    [np.random.shuffle(l) for l in y_groups]
    folds = [np.concatenate(k) for k in zip(*y_groups)]

    metrics = []
    for fold in tqdm(folds):
        mask = np.in1d(idx, fold)
        X_te, y_te = X[mask], y[mask]
        X_tr, y_tr = X[~mask], y[~mask]
        metrics.append([rf_eval(X_tr, y_tr, X_te, y_te, n_trees=n,
                                *args, **kwargs) for n in n_trees])

    return np.mean(metrics, axis=0)
