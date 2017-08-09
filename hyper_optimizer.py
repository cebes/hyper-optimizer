from __future__ import print_function
from __future__ import unicode_literals

import copy
from collections import OrderedDict

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit


def _require(condition, msg):
    if not condition:
        raise ValueError(msg)


class Parameter(object):
    CATEGORICAL = 'categorical'
    INT = 'int'
    DOUBLE = 'double'
    SCIKIT_DISTRIBUTION = 'scikit-distribution'

    def __init__(self, name='param', param_type=CATEGORICAL, min_bound=0, max_bound=100, values=None,
                 distribution=None):
        """

        :param name:
        :param param_type:
        :param min_bound:
        :param max_bound:
        :param values: list of discrete values to take. Only matter if ``param_type=CATEGORICAL``
        :param distribution: a random distribution to take values from, typically from ``scipy.stats.distributions``.
            Only matter if ``param_type=SCIKIT_DISTRIBUTION``
        """
        _require(name not in ['X', 'y'], 'Parameter named {} is forbidden'.format(name))

        self.name = name
        self.param_type = param_type
        if self.param_type == Parameter.CATEGORICAL:
            _require(values is not None, 'Categorical variables must have a list of values')
            self.values = values
        elif self.param_type == Parameter.SCIKIT_DISTRIBUTION:
            _require(distribution is not None, 'Scikit distribution variables must provide a distribution')
            self.distribution = distribution
        else:
            _require(min_bound < max_bound,
                     'Minimum bound {} must be smaller than maximum bound {}'.format(min_bound, max_bound))
            self.min_bound = min_bound
            self.max_bound = max_bound


class HyperBaseEstimator(BaseEstimator):
    """
    Base class for any estimator to be optimized by this library.
    Subclasses need to provide a constructor without using **kwargs or *args,
    implement fit, predict and score functions

    It is recommended that in the fit() function, the trained model is stored in `self._model`
    """

    def __init__(self, **parameters):
        for k, v in parameters.items():
            setattr(self, k, v)
        self._model = None

    def fit(self, X, y=None):
        raise NotImplementedError()

    def predict(self, X, y=None):
        raise NotImplementedError()

    def score(self, X, y=None):
        raise NotImplementedError()


class HistoryEntry(object):
    def __init__(self, train_scores=(), test_scores=(), fit_times=(), score_times=(), params=None):
        _require(len(train_scores) == len(test_scores) == len(fit_times) == len(score_times),
                 'arguments must have the same length, i.e. the number of splits')
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.fit_times = fit_times
        self.score_times = score_times
        self.params = params
        self.n_splits = len(train_scores)


class History(object):
    def __init__(self):
        self.entries = []
        self.n_splits = None

    def append(self, entry):
        """

        :type entry: HistoryEntry
        """
        if self.n_splits is None:
            self.n_splits = entry.n_splits
        _require(self.n_splits == entry.n_splits, 'Must have the same number of splits')
        self.entries.append(entry)
        return self

    def to_dict(self):
        """
        Convert the history into a dict (similar to sklearn cv_results_)
        """
        results = OrderedDict()

        # preserve the order of the keys
        for i in range(self.n_splits):
            results['split{}_test_score'.format(i)] = []
        for k in ['mean_test_score', 'std_test_score', 'rank_test_score']:
            results[k] = []
        for i in range(self.n_splits):
            results['split{}_train_score'.format(i)] = []
        for k in ['mean_train_score', 'std_train_score', 'mean_fit_time', 'std_fit_time',
                  'mean_score_time', 'std_score_time', 'params']:
            results[k] = []

        for entry in self.entries:
            for i in range(self.n_splits):
                results['split{}_test_score'.format(i)].append(entry.test_scores[i])
                results['split{}_train_score'.format(i)].append(entry.train_scores[i])
            for k in ['test_score', 'train_score', 'fit_time', 'score_time']:
                results['mean_{}'.format(k)].append(np.mean(getattr(entry, '{}s'.format(k))))
                results['std_{}'.format(k)].append(np.std(getattr(entry, '{}s'.format(k))))
            results['params'].append(entry.params)
        ls = results['mean_test_score']
        results['rank_test_score'] = (len(ls) - np.argsort(ls)).tolist()
        return results

    @property
    def best_entry_(self):
        best_entry = None
        for entry in self.entries:
            if best_entry is None or np.mean(best_entry.test_scores) < np.mean(entry.test_scores):
                best_entry = entry
        return best_entry

    @property
    def best_test_score_(self):
        return np.mean(self.best_entry_.test_scores)

    @property
    def best_params_(self):
        return copy.deepcopy(self.best_entry_.params)


class Optimizer(object):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None,
                 refit=True, expr_name='unnamed', verbose=0, random_state=None, error_score='raise'):
        """

        :param estimator: an object of class :ref:`HyperBaseEstimator`
        :param params: a list of :ref:`Parameter` objects
        :param max_trials: maximum number of iterations when doing parameter search
        :param cv:
            - None: Use standard 3-fold cross validation, with 10% test set
            - a scikit-learn object for cross-validation, i.e. ShuffleSplit or KFold
            - tuple (X, y=None): use this separated validation set instead

        :param refit: Refit the best estimator with the entire dataset
        :param expr_name:
        :param verbose: Controls the verbosity: the higher, the more messages.
        :param random_state: int, pseudo random number generator state used for random uniform
        :param error_score: Value to assign to the score if an error occurs in estimator fitting.
                If set to ‘raise’, the error is raised. If a numeric value is given,
                FitFailedWarning is raised. This parameter does not affect the refit step,
                which will always raise the error.
        """
        self.estimator = estimator
        self.params = params
        self.max_trials = max_trials
        self.cv = cv
        self.refit = refit
        self.expr_name = expr_name
        self.verbose = verbose
        self.random_state = random_state
        self.error_score = error_score
        self.history_ = History()

    def fit(self, X, y=None):
        raise NotImplementedError()

    @property
    def best_estimator_(self):
        raise NotImplementedError()

    @property
    def best_test_score_(self):
        return self.history_.best_test_score_

    @property
    def best_params_(self):
        return self.history_.best_params_

    @property
    def cv_results_(self):
        return self.history_.to_dict()

    """
    Helpers
    """

    def _check_cv(self):
        """
        Return a wrapped object for cross-validation
        """
        if self.cv is None:
            self.cv = ShuffleSplit(n_splits=3, test_size=0.1, random_state=self.random_state)

        if isinstance(self.cv, tuple):
            _require(len(self.cv) == 2, 'If you pass a tuple to "cv", it has to be a 2-tuple, containing (X, y)')


class RandomOptimizer(Optimizer):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None,
                 refit=True, expr_name='unnamed', verbose=0, n_jobs=1):
        """

        :param n_jobs: number of jobs running in parallel
        """
        super(RandomOptimizer, self).__init__(estimator=estimator, params=params, max_trials=max_trials,
                                              cv=cv, refit=refit, expr_name=expr_name, verbose=verbose)
        self.n_jobs = n_jobs
        self.optimizer = None

    def fit(self, X, y=None):
        self._check_cv()

        if isinstance(self.cv, tuple):
            n_train = X.shape[0]
            X = np.concatenate((X, self.cv[0]), axis=0)
            if y is not None:
                y = np.concatenate((y, self.cv[1]), axis=0)

            cv_obj = [(np.arange(0, n_train), np.arange(n_train, X.shape[0]))]
        else:
            cv_obj = self.cv

        self.optimizer = RandomizedSearchCV(estimator=self.estimator, param_distributions=self._parse_params(),
                                            n_iter=self.max_trials, n_jobs=self.n_jobs, refit=self.refit,
                                            cv=cv_obj, verbose=self.verbose, random_state=self.random_state,
                                            error_score=self.error_score)
        self.optimizer.fit(X=X, y=y)

        for i in range(self.max_trials):
            train_scores = []
            test_scores = []
            fit_times = []
            score_times = []
            for s in range(self.optimizer.n_splits_):
                train_scores.append(self.optimizer.cv_results_['split{}_train_score'.format(s)][i])
                test_scores.append(self.optimizer.cv_results_['split{}_test_score'.format(s)][i])
                fit_times.append(self.optimizer.cv_results_['mean_fit_time'][i])
                score_times.append(self.optimizer.cv_results_['mean_score_time'][i])

            entry = HistoryEntry(train_scores=train_scores, test_scores=test_scores, fit_times=fit_times,
                                 score_times=score_times, params=self.optimizer.cv_results_['params'][i])
            self.history_.append(entry)

    @property
    def best_estimator_(self):
        _require(self.optimizer is not None, 'Call fit() before you can get the best estimator')
        return self.optimizer.best_estimator_

    def _parse_params(self):
        """Helper to parse the list of params into a dict of param for sklearn Optimizer"""
        params = {}
        for p in self.params:
            if p.param_type == Parameter.CATEGORICAL:
                vals = p.values[:]
            elif p.param_type == Parameter.SCIKIT_DISTRIBUTION:
                vals = p.distribution
            elif p.param_type == Parameter.INT:
                vals = stats.randint(p.min_bound, p.max_bound)
            else:
                class _WrappedUniform(object):
                    def __init__(self, a, b):
                        self.a, self.len = a, (b - a)
                        self.d = stats.uniform()

                    def rvs(self, size=None, random_state=None):
                        return self.d.rvs(size=size, random_state=random_state) * self.len + self.a

                vals = _WrappedUniform(p.min_bound, p.max_bound)
            params[p.name] = vals
        return params


class SpearmintOptimizer(Optimizer):
    pass


class SigOptOptimizer(Optimizer):
    pass
