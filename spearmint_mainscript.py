import os
import time
import numpy as np
import json
from hyper_optimizer import HyperBaseEstimator


class NegatedBranin(HyperBaseEstimator):

    def __init__(self, a=0, b=1):
        super(NegatedBranin, self).__init__(a=a, b=b)

    def predict(self, X, y=None):
        pass

    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        import math
        import numpy as np

        return -(np.square(self.b - (5.1 / (4 * np.square(math.pi))) * np.square(self.a) +
                           (5 / math.pi) * self.a - 6) + 10 * (1 - (1. / (8 * math.pi))) * np.cos(self.a) + 10)


def main(job_id, params):

    for k in params.keys():
        v = params[k]
        if v.shape == (1,):
            params[k] = v[0]

    # load dataset
    current_folder = os.path.split(__file__)[0]
    with open(os.path.join(current_folder, 'data.npz')) as f:
        data = np.load(f)
        X, y, cv = data['X'], data['y'], data['cv'].tolist()
        if len(y.shape) == 0:
            y = None

    result_dir = os.path.join(current_folder, 'results')
    os.makedirs(result_dir, exist_ok=True)

    stats = {'test_scores': [],
             'fit_times': [],
             'score_times': [],
             'params': params}

    for train_idx, test_idx in cv:
        x_split, y_split = X[train_idx], None if y is None else y[train_idx]
        x_test_split, y_test_split = X[test_idx], None if y is None else y[test_idx]

        obj = NegatedBranin(**params)
        t = time.time()
        obj.fit(X=x_split, y=y_split)
        stats['fit_times'].append(time.time() - t)
        t = time.time()
        test_score = obj.score(X=x_test_split, y=y_test_split)
        stats['score_times'].append(time.time() - t)
        stats['test_scores'].append(test_score)

    stats['train_scores'] = [np.nan] * len(stats['test_scores'])
    with open(os.path.join(result_dir, '{}.json'.format(job_id))) as f:
        json.dump(stats, f)

    return np.mean(stats['test_scores'])
