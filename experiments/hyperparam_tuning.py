from audioop import mul
import numpy as np
import pandas as pd
import json

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, TimeSeriesSplit

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost

from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as metrics

from tqdm import tqdm

def print_classification_report(opt, X_train, X_test, y_train, y_test):
    print('Train:')
    y_hat = opt.predict(X_train)
    print(metrics.classification_report(y_train, y_hat))
    print('Score:', opt.score(X_train, y_train))

    print('Test:')
    y_hat = opt.predict(X_test)
    print(metrics.classification_report(y_test, y_hat))
    print('Score:', opt.score(X_test, y_test))

def bayes_opt(X_train, X_test, y_train, y_test,
              pipeline, model_search_space, imbalanced_search_space, n_iter, scoring='f1', n_jobs=-1, n_points=1):
    search_spaces = {
        **imbalanced_search_space,
        **model_search_space
    }
    print('Search Spaces:')
    for name, values in search_spaces.items():
        print(f'{name}: {values}')
    print()
                      
    # time splits same proportions as train/test
    y_test_proportion = len(y_test) / (len(y_train) + len(y_test))
    tscv = TimeSeriesSplit(n_splits=5, test_size=round(len(y_train) * y_test_proportion))

    print(f'Sample {n_points=} with {n_jobs=}.')
    with tqdm(total=n_iter) as pbar:
        opt = BayesSearchCV(
            pipeline,
            search_spaces,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            n_points=n_points, # number of models evaluated simultaneously
            random_state=0,
            verbose=0
        )

        def on_step(optim_result):
            pbar.update(1)

        opt.fit(X_train, y_train, callback=on_step)

        print(f'best val. {scoring}: {opt.best_score_}')

        print_classification_report(opt, X_train, X_test, y_train, y_test)
        
        print('Best parameters:')
        print(opt.best_params_)
        
        return opt


def save_cv_results(opt, path):
    res = opt.cv_results_.copy()
    res['param_model'] = list(map(lambda m: str(type(m)).split('.')[-1][:-2], res['param_model'])) # remove '>
    res = pd.DataFrame(res)
    params = [c for c in res.columns if 'param_' in c]
    res = res[params + ['mean_fit_time', 'std_fit_time', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    res = res.sort_values(by='rank_test_score')
    res = res.convert_dtypes()

    res.to_csv(path)

def tpot_opt(X_train, X_test, y_train, y_test, n_iter, scoring='f1', n_jobs=-1):
    import multiprocessing
    multiprocessing.set_start_method('forkserver')
    from tpot import TPOTClassifier

    y_test_proportion = len(y_test) / (len(y_train) + len(y_test))
    tscv = TimeSeriesSplit(n_splits=5, test_size=round(len(y_train) * y_test_proportion))
    opt = TPOTClassifier(generations=n_iter, population_size=100,
                            cv=tscv, scoring=scoring,
                            random_state=0, verbosity=3,
                            n_jobs=n_jobs)
    opt.fit(X_train, y_train)

    cv_scores = [(name, info['internal_cv_score']) for name, info in opt.evaluated_individuals_.items()]
    cv_scores = sorted(cv_scores, key=lambda p: -p[1])

    print(f'best val. {scoring}: {cv_scores[0][1]}')

    print_classification_report(opt, X_train, X_test, y_train, y_test)

    return opt

def save_tpot_resuls(opt, path):
    def write_json_to_file(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    write_json_to_file(opt.evaluated_individuals_, path + '_tpot_evaluated_individuals.json')
    write_json_to_file({k: str(v) for k,v in opt.pareto_front_fitted_pipelines_.items()}, path + '_tpot_pareto_front.json')
    opt.export(path + '_tpot_exported_pipeline.py')


def default_pipeline():
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('sampler', None),
        ('model', None)   
    ])

def imbalanced_search_space():
    return {
        'sampler': Categorical([
            None,
            RandomUnderSampler(random_state=0),
            RandomOverSampler(random_state=0),
            SMOTE(random_state=0)
        ])
        #'model__class_weight': Categorical([None, 'balanced']) MLP does not have, xgboost has scale_pos_weight
    }
        
def logistic_regression_search_space():
    return {
        'model': Categorical([LogisticRegression(random_state=0, solver='saga')]),
        'model__C': (1e-4, 1e+3, 'log-uniform'),
        'model__penalty': Categorical(['l1', 'l2']),
    }


def svm_pipeline():
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('sampler', None),
        ('kernel', Nystroem(random_state=0, n_components=500)),
        ('model', None)   
    ])
        
def svm_search_space():
    return {
        'model': Categorical([LinearSVC(random_state=0)]),
        'model__C': Real(1e-4, 1e+3, 'log-uniform'),
        'kernel__kernel': Categorical(['linear', 'rbf', 'poly']),
        'kernel__gamma': Categorical([None]),
        'kernel__degree': Categorical([3, 5])
    }


def mlp_search_space():
    return {
        'model': Categorical([MLPClassifier(random_state=0)]),
        'model__alpha': Real(1e-4, 1e+3, 'log-uniform'),
        'model__learning_rate_init': Real(1e-4, 1e-1, 'log-uniform'),
        'model__hidden_layer_sizes': Categorical([10, 20, 50, 100, 200]),
        'model__activation': Categorical(['relu', 'logistic', 'tanh'])
    }


def random_forest_search_space():
    return {
        'model': Categorical([RandomForestClassifier(random_state=0)]),
        'model__max_depth': Categorical([None, 3, 5, 10, 15, 20]),
        'model__min_samples_split': Categorical([2, 5]),
        'model__n_estimators': Integer(5,150)
    }


def xgboost_search_space():
    return {
        'model': Categorical([xgboost.XGBClassifier(random_state=0, n_jobs=4, use_label_encoder=False, eval_metric='logloss')]),
        'model__max_depth': Integer(3, 10),
        'model__min_child_weight': Categorical([1, 2, 5]),
        'model__max_delta_step': Categorical([0, 1]),
        'model__n_estimators': Integer(5,150),
        'model__gamma': Real(0, 1),
        
    }


import argparse
import sys
import os
from experiments.data_utils import *

if __name__ == '__main__':
    output_dir = 'experiments/results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, dest='n_iter', default=100,
        help='Number of samples per model.')

    parser.add_argument('--n_jobs', type=int, dest='n_jobs', default=5,
        help='Number of parallel jobs. Should be a multiple of 5 (# of CV splits) for BayesOpt.')

    parser.add_argument('--target', type=str, dest='target', default='performance',
        help='Classification target.')

    parser.add_argument('--model', type=str, dest='model', default='lr',
        help='Select model to perform hyperparameter tuning.')

    parser.add_argument('--data', type=str, dest='data', default='bugbug_buglevel',
        help='Choice of labeling and data.')
        
    parser.add_argument('--scoring', type=str, dest='scoring', default='roc_auc',
        help='Scoring function to be optimized.')


    args = parser.parse_args(sys.argv[1:])
    print(f'\n{args=}\n')

    data_map = {
        'bugbug_buglevel': lambda target: get_bugbug_data(target, kind='buglevel'),
        'bugbug_szz': lambda target: get_szz_commitlevel('bugbug_szz', target),
        'fixed_defect_szz': lambda target: get_szz_commitlevel('fixed_defect_szz', target)
    }
    assert args.data in data_map.keys(), f'Unknown data and labeling {args.data=}.'

    X, y, features = data_map[args.data](args.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    search_space_map = {
        'lr': logistic_regression_search_space,
        'svm': svm_search_space,
        'mlp': mlp_search_space,
        'rf': random_forest_search_space,
        'xgb': xgboost_search_space,
    }
    
    if args.model == 'tpot':
        opt = tpot_opt(X_train, X_test, y_train, y_test,
                n_iter=args.n_iter,
                scoring=args.scoring,
                n_jobs=args.n_jobs)
        
        save_tpot_resuls(opt, os.path.join(output_dir, f'{args.data}_{args.target}'))

    else:
        args.model in search_space_map.keys(), f'Invalid model {args.model}.'

        pipeline = default_pipeline if args.model != 'svm' else svm_pipeline

        # %%
        opt = bayes_opt(X_train, X_test, y_train, y_test,
                        pipeline=pipeline(),
                        model_search_space=search_space_map[args.model](),
                        imbalanced_search_space=imbalanced_search_space(),
                        n_iter=args.n_iter,
                        scoring=args.scoring,
                        n_jobs=args.n_jobs,
                        n_points=max(args.n_jobs // 5, 1))

        save_cv_results(opt, os.path.join(output_dir, f'{args.data}_{args.target}_{args.scoring}_{args.model}.csv'))
