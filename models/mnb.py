from collections import Counter, defaultdict
from itertools import chain
import numpy as np


def init_classifier(p_train, n_train):
	p_freq, n_freq = Counter(chain(*p_train)), Counter(chain(*n_train))
	p_len, n_len = sum([len(l) for l in p_train]), sum(
		[len(l) for l in n_train])

	N = len(p_train) + len(n_train)
	V = len(set(p_train) | set(n_train)) + 1

	p, n = len(p_train) / N, len(n_train) / N
	p_probs, n_probs = None, None

	def calc_probs(alpha=0):
		nonlocal p_probs, n_probs
		p_probs = defaultdict(lambda: alpha/V, {k: (v + alpha) / (p_len + alpha * V)
												for k, v in p_freq.items()})
		n_probs = defaultdict(lambda: alpha/V, {k: (v + alpha) / (n_len + alpha * V)
												for k, v in n_freq.items()})

	def classify(doc, log_probs=False):
		agg = np.sum if log_probs else np.prod
		def f(x): return np.log(max(1e-8, x)) if log_probs else max(1e-8, x)
		p_prob = agg([f(p)] + [f(p_probs[w]) for w in set(doc)])
		n_prob = agg([f(n)] + [f(n_probs[w]) for w in set(doc)])
		# 1 is pos; 0 is neg
		return 1 if p_prob > n_prob else 0 if p_prob < n_prob else np.random.choice([0, 1])


	def multinomial_naive_bayes(p_test, n_test, *args, **kwargs):
		p_pred = [classify(doc, *args, **kwargs) for doc in p_test]
		n_pred = [classify(doc, *args, **kwargs) for doc in n_test]
		return eval(p_pred, n_pred)

	calc_probs()
	return multinomial_naive_bayes, calc_probs


def eval(p_pred, n_pred):
	tp, fn = sum(p_pred), len(p_pred) - sum(p_pred)
	tn, fp = len(n_pred) - sum(n_pred), sum(n_pred)
	return np.array([[tp, fp], [fn, tn]])


def calc_metrics(confusion_matrix):
	tp, fp, fn, tn = confusion_matrix.ravel()
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return accuracy, precision, recall

