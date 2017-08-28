# Convenient classes for optimizing hyper-parameters

## 1. Overview

This is an attempt to benchmark different implementation of BayesOpt for hyper-parameter tuning for a real-world
problem. We do that by writing a simple wrapper for different implementations of Bayesian Optimizers. As a result,
the `hyper_optimizer.py` script can be used separately in your projects.

1. Setup environment. We recommend python3 and its `venv` module:

        python3 -m venv ./venv3
        source venv3/bin/activate
        pip install -r requirements.txt
        
2. Implement an Estimator by inheriting from HyperBaseEstimator:

        from hyper_optimizer import HyperBaseEstimator
        
        class MyEstimator(HyperBaseEstimator):
            def __init__(self, ...):
                super(NegatedBranin, self).__init__(...)
        
            def fit(self, X, y=None):
                # implement the fit() function, this is where you train the model given 
                # all the hyper-parameters received in the constructor
                pass
                
            def predict(self, X, y=None):
                # the trained model is used to predict the outcomes for X here
                pass            
        
            def score(self, X, y=None):
                # implement a custom scorer. Normally you will use the predict() function
                # to compute the outcomes z, and then use some score function to compare z to y
                # This function has to return a scalar value
                # All the optimizers in this project will try to *maximize* the return value
                # of this function.

    As you can see, this follows `sklearn`'s convention for [custom estimators](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator),
    but we don't force you to implement `set_params` and `get_params` as long as all the hyper-parameters
    are initialized in the constructor.
    
3. You can then use any implementation of BayesOpt to optimize the Estimator like so:

        from hyper_optimizer import Parameter, SkOptOptimizer
        
        opt = SkOptOptimizer(estimator=MyEstimator(),
                             params=[Parameter('a', Parameter.DOUBLE, min_bound=-5, max_bound=10),
                                     Parameter('b', Parameter.DOUBLE, min_bound=0, max_bound=15)],
                             max_trials=20)
        
        # prepare X and y, then call opt.fit()
        opt.fit(X, y)
        
    Once this is done, you can access the best score and configuration using `opt.best_test_score_`, `opt.best_params_`,
    and `opt.best_estimator_`. The history of tried configurations can be accessed at `opt.history_`.
    
## 2. Optimizers and parameters

The following optimizers are implemented:

| Optimizer            | Description                                                                                                                                                                                                                       | Limitation                                   |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| `RandomOptimizer`    | A light wrapper of [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html),  which performs random search on the search space. Can be used as a (strong) baseline. |                                              |
| `SkOptOptimizer`     | A light wrapper of [skopt BayesSearchCV](https://scikit-optimize.github.io/#skopt.BayesSearchCV)                                                                                                                                  |                                              |
| `SigOptOptimizer`    | A light wrapper of [sigopt_sklearn SigOptSearchCV](https://sigopt.com/docs/overview/scikit_learn)                                                                                                                                 | Limited support in the free version          |
| `BayesOptimizer`     | A light wrapper of [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)                                                                                                                                           | Does not support categorical variables (yet) |
| `SpearmintOptimizer` | A light wrapper of [Spearmint](https://github.com/HIPS/Spearmint)                                                                                                                                                                 | Restrictive license                          |

The most common parameters of those optimizers are:

- **estimator**: an object of class :ref:`HyperBaseEstimator`. This class should implement a `fit` and `score` function.
- **params**: a list of :ref:`Parameter` objects, describing the search space
- **max_trials**: maximum number of iterations when doing parameter search
- **cv**: how to do cross validation, can be one of the following:
    - `None`: Use standard 3-fold cross validation, with 10% test set
    - a scikit-learn object for cross-validation, i.e. `ShuffleSplit` or `KFold`
    - tuple (X, y=None): use this separated validation set instead. This is more preferred when working with medium
    and large datasets.
- **refit**: Refit the best estimator with the entire dataset
- **verbose**: Controls the verbosity: the higher, the more messages.
- **random_state**: int, pseudo random number generator state used for random uniform
- **error_score**: Value to assign to the score if an error occurs in estimator fitting.
        If set to ‘raise’, the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit step,
        which will always raise the error.
        
The optimizers work to figure out the best values of the `Parameter`s that **maximizes** the scoring function for the given
dataset. We support continuous, integer and categorical parameters (although not all optimizers support those parameter types):

    # a categorical parameter
    Parameter(name='booster', param_type=Parameter.CATEGORICAL, values=['gbtree', 'gblinear', 'dart'])
    
    # integer parameter
    Parameter(name='max_depth', param_type=Parameter.INT, min_bound=0, max_bound=100)
    
    # continuous parameter
    Parameter(name='learning_rate', param_type=Parameter.DOUBLE, min_bound=0.01, max_bound=0.5)
    
    # a Scipy random distribution (only supported in RandomOptimizer)
    Parameter(name='learning_rate', param_type=Parameter.SCIKIT_DISTRIBUTION, distribution=scipy.stats.cauchy)
    
## 3. Example

This is an implementation of a custom Estimator for optimizing an XGBoost regressor

    from hyper_optimizer import Parameter, HyperBaseEstimator

    class XGBoostEstimator(HyperBaseEstimator):
        def __init__(self, learning_rate=0.1, gamma=0, colsample_bytree=1, reg_lambda=1):
            super(XGBoostEstimator, self).__init__(learning_rate=learning_rate, gamma=gamma,
                                                   colsample_bytree=colsample_bytree, reg_lambda=reg_lambda)
            self.model_ = None
            
        def predict(self, X, y=None):
            import xgboost as xgb
            return self.model_.predict(xgb.DMatrix(X, label=y))
    
        def fit(self, X, y=None):
            import xgboost as xgb
            dtrain = xgb.DMatrix(X, label=y)
            watchlist = [(dtrain, 'train')]
    
            xgb_pars = dict(learning_rate=self.learning_rate, gamma=self.gamma, subsample=1,
                            colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                            base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            max_delta_step=0, max_depth=18, min_child_weight=1, 
                            n_estimators=180, reg_alpha=0, scale_pos_weight=1, 
                            n_jobs=2, eval_metric='rmse', objective='reg:linear', random_state=42, missing=None)
                
            self.model_ = xgb.train(xgb_pars, dtrain, num_boost_round=60, evals=watchlist, 
                                    early_stopping_rounds=50, maximize=False, verbose_eval=50)
    
        def score(self, X, y=None):
            # the library will maximize the return value of this function,
            # so we are gonna return the negated score of the regressor
            from sklearn.metrics import mean_squared_error
            import math
            y_pred = self.predict(X, y)
            return -math.sqrt(mean_squared_error(y_pred, y))


    params = [Parameter('learning_rate', Parameter.DOUBLE, min_bound=0.001, max_bound=0.4),
              Parameter('gamma', Parameter.DOUBLE, min_bound=0.0, max_bound=3.0),
              Parameter('colsample_bytree', Parameter.DOUBLE, min_bound=0.5, max_bound=1.0),
              Parameter('reg_lambda', Parameter.DOUBLE, min_bound=0.0, max_bound=1.0)]
              
              
Note that we tune 4 hyper-parameters, and the names of those hyper-parameters are defined in the constructor of the 
estimator. We do the main work in the `fit()` function,  and the `score()` function conveniently call the `predict()`
function before computing the mean squared error. The `score()` function returns the negated mean squared error because
the optimizers will **maximize** whatever it returns. By maximizing the negated MSE, we practically minimize the MSE.

For a more serious example, see the notebook in `nyc_taxi_duration/main.ipynb`.
