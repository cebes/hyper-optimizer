from __future__ import print_function
from __future__ import unicode_literals

import copy
import glob
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict

import numpy as np
import six
from bayes_opt import BayesianOptimization
from scipy import stats
from sigopt_sklearn.search import SigOptSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from spearmint import main as spearmint_main
from spearmint.resources.resource import parse_resources_from_config, print_resources_status
from spearmint.utils.database.mongodb import MongoDB


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


@six.python_2_unicode_compatible
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

    def __repr__(self):
        return '{}(params={},test_scores={})'.format(self.__class__.__name__, self.params, self.test_scores)


class History(object):
    def __init__(self):
        self.entries = []
        self.n_splits = None

    def __repr__(self):
        return '{}(entries={})'.format(self.__class__.__name__, self.entries)

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
        arg_sorted = np.argsort(results['mean_test_score'])

        ls = [0] * len(arg_sorted)
        for i, v in enumerate(arg_sorted):
            ls[v] = i + 1
        results['rank_test_score'] = ls

        return results

    def plot(self, style='seaborn-dark-palette', figsize=(20, 10), **fig_kw):
        import matplotlib.pyplot as plt

        plt.style.use(style)
        fig, axs = plt.subplots(2, 1, sharex='all', figsize=figsize, **fig_kw)

        x = range(len(self.entries))
        avg_train_score = [np.mean(e.train_scores) for e in self.entries]
        avg_train_score = [np.max(avg_train_score[:i+1]) for i in range(len(avg_train_score))]
        avg_test_score = [np.mean(e.test_scores) for e in self.entries]
        avg_test_score = [np.max(avg_test_score[:i + 1]) for i in range(len(avg_test_score))]
        axs[0].plot(x, avg_train_score, label='Best average training score', linewidth=2)
        axs[0].plot(x, avg_test_score, label='Best average test score', linewidth=2)
        axs[0].set_title('Average train and test scores across folds')
        axs[0].set_xlabel('Trial')
        axs[0].set_ylabel('Score')
        axs[0].grid(which='major', alpha=0.5)
        axs[0].legend()

        avg_fit_time = [np.mean(e.fit_times) for e in self.entries]
        std_fit_time = [np.std(e.fit_times) for e in self.entries]
        avg_score_time = [np.mean(e.score_times) for e in self.entries]
        std_score_time = [np.std(e.score_times) for e in self.entries]
        axs[1].errorbar(x, avg_fit_time, yerr=std_fit_time, label='Average training time', linewidth=2)
        axs[1].errorbar(x, avg_score_time, yerr=std_score_time, label='Average scoring time', linewidth=2)
        axs[1].set_xlabel('Trial')
        axs[1].set_ylabel('Time (seconds)')
        axs[1].set_xticks(x)
        axs[1].grid(which='major', alpha=0.5)
        axs[1].legend()

        return fig


class Optimizer(object):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None,
                 refit=True, verbose=0, random_state=None, error_score='raise'):
        """

        :param estimator: an object of class :ref:`HyperBaseEstimator`
        :param params: a list of :ref:`Parameter` objects
        :param max_trials: maximum number of iterations when doing parameter search
        :param cv:
            - None: Use standard 3-fold cross validation, with 10% test set
            - a scikit-learn object for cross-validation, i.e. ShuffleSplit or KFold
            - tuple (X, y=None): use this separated validation set instead

        :param refit: Refit the best estimator with the entire dataset
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

    def _check_cv(self, X, y=None, extract_to_list=False):
        """
        Return a wrapped object for cross-validation
        """
        if self.cv is None:
            self.cv = ShuffleSplit(n_splits=3, test_size=0.1, random_state=self.random_state)

        cv_obj = self.cv
        if isinstance(self.cv, tuple):
            _require(len(self.cv) == 2, 'If you pass a tuple to "cv", it has to be a 2-tuple, containing (X, y)')
            n_train = X.shape[0]
            X = np.concatenate((X, self.cv[0]), axis=0)
            if y is not None:
                y = np.concatenate((y, self.cv[1]), axis=0)

            cv_obj = [(np.arange(0, n_train), np.arange(n_train, X.shape[0]))]
        elif extract_to_list:
            cv_obj = list(cv_obj.split(X))

        return X, y, cv_obj


class RandomOptimizer(Optimizer):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None,
                 refit=True, verbose=0, random_state=None, error_score='raise', n_jobs=1):
        """

        :param n_jobs: number of jobs running in parallel
        """
        super(RandomOptimizer, self).__init__(estimator=estimator, params=params, max_trials=max_trials,
                                              cv=cv, refit=refit, verbose=verbose, random_state=random_state,
                                              error_score=error_score)
        self.n_jobs = n_jobs
        self.optimizer = None

    def fit(self, X, y=None):
        X, y, cv_obj = self._check_cv(X, y)
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


class SigOptOptimizer(Optimizer):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None,
                 refit=True, verbose=0, random_state=None, error_score='raise', api_token='', n_jobs=1):
        """

        :param estimator:
        :param params:
        :param max_trials:
        :param cv:
        :param refit:
        :param verbose:
        :param api_token:
        :param n_jobs:
        """
        super(SigOptOptimizer, self).__init__(estimator=estimator, params=params, max_trials=max_trials,
                                              cv=cv, refit=refit, verbose=verbose,
                                              random_state=random_state, error_score=error_score)
        self.api_token = api_token
        self.n_jobs = n_jobs
        self.optimizer = None

    def fit(self, X, y=None):
        X, y, cv_obj = self._check_cv(X, y, extract_to_list=True)
        self.optimizer = SigOptSearchCV(estimator=self.estimator, param_domains=self._parse_params(),
                                        n_iter=self.max_trials, n_jobs=self.n_jobs, refit=self.refit,
                                        cv=cv_obj, verbose=self.verbose, error_score=self.error_score,
                                        client_token=self.api_token)
        self.optimizer.fit(X=X, y=y)

        for obs in self.optimizer.sigopt_connection.experiments(
                self.optimizer.experiment.id).observations().fetch().iterate_pages():
            entry = HistoryEntry(train_scores=[np.nan], test_scores=[obs.value],
                                 fit_times=[np.nan], score_times=[np.nan], params=obs.assignments.to_json())
            self.history_.append(entry)

    @property
    def best_estimator_(self):
        return self.optimizer.best_estimator_

    def _parse_params(self):
        params = {}
        for p in self.params:
            if p.param_type == Parameter.CATEGORICAL:
                vals = p.values[:]
            elif p.param_type == Parameter.SCIKIT_DISTRIBUTION:
                raise ValueError('Parameter {} of type {} is not supported by {}'.format(
                    p.name, p.param_type, self.__class__.__name__))
            else:
                cast_func = float if p.param_type == Parameter.DOUBLE else int
                vals = (cast_func(p.min_bound), cast_func(p.max_bound))
            params[p.name] = vals
        return params


class BayesOptimizer(Optimizer):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None,
                 refit=True, verbose=0, random_state=None, error_score='raise', acquisition_func='ucb'):
        """
        Does not support categorical and integer variables

        :param estimator:
        :param params:
        :param max_trials:
        :param cv:
        :param refit:
        :param verbose:
        :param random_state:
        :param error_score:
        :param acquisition_func:
        """
        if not inspect.isclass(estimator):
            estimator = estimator.__class__

        super(BayesOptimizer, self).__init__(estimator=estimator, params=params, max_trials=max_trials,
                                             cv=cv, refit=refit, verbose=verbose, random_state=random_state,
                                             error_score=error_score)
        self.acquisition_func = acquisition_func
        self.optimizer_ = None
        self._x_all = None
        self._y_all = None
        self._cv_obj = None
        self.history_ = History()
        self._best_estimator_ = None

    @property
    def best_estimator_(self):
        return self._best_estimator_

    def fit(self, X, y=None):
        self._x_all, self._y_all, self._cv_obj = self._check_cv(X, y, extract_to_list=True)
        self.history_ = History()

        self.optimizer_ = BayesianOptimization(self._objective_func, pbounds=self._parse_params(), verbose=self.verbose)
        self.optimizer_.maximize(n_iter=self.max_trials, acq=self.acquisition_func)

        if self.refit:
            self._best_estimator_ = self.estimator(**self.best_params_)
            self._best_estimator_.fit(X=X, y=y)

    def _objective_func(self, **kwargs):
        estimator = self.estimator(**kwargs)

        entry = HistoryEntry(train_scores=[], test_scores=[], fit_times=[], score_times=[], params=kwargs)
        for train_idx, test_idx in self._cv_obj:
            x_split, y_split = self._x_all[train_idx], None if self._y_all is None else self._y_all[train_idx]
            x_test_split, y_test_split = self._x_all[test_idx], None if self._y_all is None else self._y_all[test_idx]

            t = time.time()
            estimator.fit(X=x_split, y=y_split)
            entry.fit_times.append(time.time() - t)
            t = time.time()
            test_score = estimator.score(X=x_test_split, y=y_test_split)
            entry.score_times.append(time.time() - t)
            entry.test_scores.append(test_score)
            entry.train_scores.append(np.nan)

        self.history_.append(entry)
        return np.mean(entry.test_scores)

    def _parse_params(self):
        params = {}
        for p in self.params:
            if p.param_type == Parameter.CATEGORICAL or p.param_type == Parameter.SCIKIT_DISTRIBUTION:
                raise ValueError('Parameter {} of type {} is not supported by {}'.format(
                    p.name, p.param_type, self.__class__.__name__))
            else:
                cast_func = float if p.param_type == Parameter.DOUBLE else int
                vals = (cast_func(p.min_bound), cast_func(p.max_bound))
            params[p.name] = vals
        return params


"""
Spearmint optimizer
"""


class SpearmintOptimizer(Optimizer):
    def __init__(self, estimator=None, params=None, max_trials=40, cv=None, refit=True, verbose=0,
                 noisy_likelihood=True, db_address='localhost', expr_name='unnamed', overwrite_expr=True,
                 polling_time=0, n_jobs=1):
        """
        The Free plan doesn't support Categorical variables and more than 4 variables.

        :param estimator:
        :param params:
        :param max_trials:
        :param cv:
        :param refit:
        :param verbose:
        :param noisy_likelihood:
        :param db_address:
        :param expr_name:
        :param overwrite_expr:
        :param polling_time:
        :param n_jobs:
        """
        if not inspect.isclass(estimator):
            estimator = estimator.__class__
        super(SpearmintOptimizer, self).__init__(estimator=estimator, params=params, max_trials=max_trials,
                                                 cv=cv, refit=refit, verbose=verbose,
                                                 random_state=None, error_score='raise')
        self.noisy_likelihood = noisy_likelihood
        self.db_address = db_address
        self.expr_name = expr_name
        self.overwrite_expr = overwrite_expr
        self.polling_time = polling_time
        self.n_jobs = n_jobs
        self._best_estimator_ = None

    def fit(self, X, y=None):

        script_file = 'branin_noisy.py'
        options = {'chooser': 'default_chooser',
                   'language': 'PYTHON',
                   'main-file': script_file,
                   'experiment-name': self.expr_name,
                   'tasks': {'main': {'type': 'OBJECTIVE',
                                      'likelihood': 'GAUSSIAN' if self.noisy_likelihood else 'NOISELESS'}},
                   'database': {'name': 'spearmint', 'address': self.db_address},
                   'variables': self._parse_params(),
                   'max-concurrent': self.n_jobs}
        expt_dir = tempfile.mkdtemp(prefix='hyper_optimizer', suffix=self.expr_name)

        print('Experiment directory: {}'.format(expt_dir))

        # bump data and cv to a file in expt_dir
        X_all, y_all, cv_obj_list = self._check_cv(X, y, extract_to_list=True)
        with open(os.path.join(expt_dir, 'data.npz'), 'wb') as f:
            np.savez(f, X=X_all, y=y_all, cv=cv_obj_list)

        # create the main script, write it to script_file
        with open(os.path.join(expt_dir, script_file), 'w') as f:
            f.write(self._create_script())

        resources = parse_resources_from_config(options)
        # Load up the chooser.
        chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
        chooser = chooser_module.init(options)
        experiment_name = options.get("experiment-name", 'unnamed-experiment')

        # Connect to the database
        db_address = options['database']['address']
        sys.stderr.write('Using database at %s.\n' % db_address)
        db = MongoDB(database_address=db_address)

        if self.overwrite_expr:
            db.remove(experiment_name, 'jobs')

        while True:

            # clean up
            jobs = spearmint_main.load_jobs(db, experiment_name)
            for job in jobs:
                if job['status'] == 'pending':
                    if not resources[job['resource']].isJobAlive(job):
                        job['status'] = 'broken'
                        spearmint_main.save_job(job, db, experiment_name)
                        # always raise if job is broken
                        raise ValueError('Broken job {} detected. Experiment folder: {}'.format(job['id'], expt_dir))

            # break if more than max_trials jobs have been completed
            trials = sum(job['status'] == 'complete' for job in jobs)
            if trials >= self.max_trials:
                break

            for resource_name, resource in resources.items():

                jobs = spearmint_main.load_jobs(db, experiment_name)

                while resource.acceptingJobs(jobs):

                    # Load jobs from DB
                    # (move out of one or both loops?) would need to pass into load_tasks
                    jobs = spearmint_main.load_jobs(db, experiment_name)

                    # Get a suggestion for the next job
                    suggested_job = spearmint_main.get_suggestion(chooser, resource.tasks, db,
                                                                  expt_dir, options, resource_name)

                    # Submit the job to the appropriate resource
                    process_id = resource.attemptDispatch(experiment_name, suggested_job, db_address, expt_dir)

                    # Set the status of the job appropriately (successfully submitted or not)
                    if process_id is None:
                        suggested_job['status'] = 'broken'
                        spearmint_main.save_job(suggested_job, db, experiment_name)
                        raise ValueError('Failed to dispatch job {}. Experiment folder: {}'.format(
                            suggested_job['id'], expt_dir))
                    else:
                        suggested_job['status'] = 'pending'
                        suggested_job['proc_id'] = process_id
                        spearmint_main.save_job(suggested_job, db, experiment_name)

                    # Print out the status of the resources
                    if self.verbose > 0:
                        jobs = spearmint_main.load_jobs(db, experiment_name)
                        print_resources_status(resources.values(), jobs)

            # If no resources are accepting jobs, sleep
            # (they might be accepting if suggest takes a while and so some jobs already
            # finished by the time this point is reached)
            if spearmint_main.tired(db, experiment_name, resources):
                time.sleep(self.polling_time)

        for result_file in sorted(glob.glob(os.path.join(expt_dir, 'results/*.json'))):
            with open(result_file, 'r') as f:
                result = json.load(f)
            entry = HistoryEntry(train_scores=result['train_scores'],
                                 test_scores=result['test_scores'],
                                 fit_times=result['fit_times'],
                                 score_times=result['score_times'],
                                 params=result['params'])
            self.history_.append(entry)

        if self.verbose > 0:
            print('Done. Experiment folder: {}'.format(expt_dir))

        if self.refit:
            self._best_estimator_ = self.estimator(**self.best_params_)
            self._best_estimator_.fit(X=X, y=y)

    @property
    def best_estimator_(self):
        return self._best_estimator_

    def _parse_params(self):
        params = {}
        for p in self.params:
            if p.param_type == Parameter.CATEGORICAL:
                vals = {'type': 'ENUM',
                        'size': 1,
                        'options': p.values[:]}
            elif p.param_type == Parameter.SCIKIT_DISTRIBUTION:
                raise ValueError('Parameter {} of type {} is not supported by {}'.format(
                    p.name, p.param_type, self.__class__.__name__))
            else:
                cast_func = float if p.param_type == Parameter.DOUBLE else int
                vals = {'type': 'FLOAT' if p.param_type == Parameter.DOUBLE else 'INT',
                        'size': 1,
                        'min': cast_func(p.min_bound),
                        'max': cast_func(p.max_bound)}
            params[p.name] = vals
        return params

    def _create_script(self):

        return """import os
import time
import numpy as np
import json

from sklearn.base import BaseEstimator

%s

%s

def main(job_id, params):
    print(job_id, params)
    
    for k in params.keys():
        v = params[k]
        if isinstance(v, list):
            if len(v) == 1:
                params[k] = v[0]
        elif v.shape == (1,):
            params[k] = v[0]

    # load dataset
    current_folder = os.path.split(__file__)[0]
    with open(os.path.join(current_folder, 'data.npz'), 'rb') as f:
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

        obj = %s(**params)
        t = time.time()
        obj.fit(X=x_split, y=y_split)
        stats['fit_times'].append(time.time() - t)
        t = time.time()
        test_score = obj.score(X=x_test_split, y=y_test_split)
        stats['score_times'].append(time.time() - t)
        stats['test_scores'].append(test_score)

    stats['train_scores'] = [np.nan] * len(stats['test_scores'])
    with open(os.path.join(result_dir, '{:09d}.json'.format(job_id)), 'w') as f:
        json.dump(stats, f)

    # returns the negative score because spearmint minimizes the objective function
    return -np.mean(stats['test_scores'])

        """ % (inspect.getsource(HyperBaseEstimator), inspect.getsource(self.estimator), self.estimator.__name__)
