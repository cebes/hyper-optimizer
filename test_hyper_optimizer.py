import math
import unittest

import numpy as np

from hyper_optimizer import Parameter, HyperBaseEstimator, RandomOptimizer


class NegatedBranin(HyperBaseEstimator):
    def __init__(self, a=0, b=1):
        super(NegatedBranin, self).__init__(a=a, b=b)

    def predict(self, X, y=None):
        pass

    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        return -(np.square(self.b - (5.1 / (4 * np.square(math.pi))) * np.square(self.a) +
                           (5 / math.pi) * self.a - 6) + 10 * (1 - (1. / (8 * math.pi))) * np.cos(self.a) + 10)


class TestHyperOptimizer(unittest.TestCase):
    def test_random_optimizer(self):
        opt = RandomOptimizer(estimator=NegatedBranin(),
                              params=[Parameter('a', Parameter.DOUBLE, min_bound=-5, max_bound=10),
                                      Parameter('b', Parameter.DOUBLE, min_bound=0, max_bound=15)],
                              max_trials=20)

        opt.fit(np.arange(100), np.arange(100))
        self.assertIsInstance(opt.best_estimator_, NegatedBranin)
        self.assertIsInstance(opt.cv_results_, dict)
        self.assertIsInstance(opt.best_params_, dict)


if __name__ == '__main__':
    unittest.main()