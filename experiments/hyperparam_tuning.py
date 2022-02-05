import numpy as np
import pandas as pd

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

def default_pipeline():
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('sampler', None),
        ('model', None)   
    ])

def bayes_opt(X_train, X_test, y_train, y_test,
              pipeline, model_search_space, imbalanced_search_space, n_iter, scoring='f1', n_jobs=-1):
    search_spaces = {
        **imbalanced_search_space,
        **model_search_space
    }
    print('Search Spaces:')
    for name, values in search_spaces.items():
        print(f'{name}: {values}')
                      
    # time splits same proportions as train/test
    y_test_proportion = len(y_test) / (len(y_train) + len(y_test))
    tscv = TimeSeriesSplit(n_splits=5, test_size=round(len(y_train) * y_test_proportion))

    with tqdm(total=n_iter) as pbar:
        opt = BayesSearchCV(
            pipeline,
            search_spaces,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=0,
            verbose=0
        )

        def on_step(optim_result):
            pbar.update(1)

        opt.fit(X_train, y_train, callback=on_step)

        print(f'best val. score {opt.best_score_}')
        
        print('Train:')
        y_hat = opt.predict(X_train)
        print(metrics.classification_report(y_train, y_hat))
        print('Test:')
        y_hat = opt.predict(X_test)
        print(metrics.classification_report(y_test, y_hat))
        
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

def imbalanced_search_space():
    return {
        'sampler': Categorical([None, RandomUnderSampler(), RandomOverSampler(), SMOTE()])
        #'model__class_weight': Categorical([None, 'balanced']) MLP does not have, xgboost has scale_pos_weight
    }
        
def logistic_regression_search_space():
    return {
        'model': Categorical([LogisticRegression(random_state=0, max_iter=1000, solver='saga')]),
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
        'model__penalty': Categorical(['l1', 'l2']),
        'kernel__kernel': Categorical(['linear', 'rbf', 'poly']),
        'kernel__gamme': Categorical(['scale']),
        'kernel__degree': Categorical([3, 5])
    }


def mlp_search_space():
    return {
        'model': Categorical([MLPClassifier(random_state=0)]),
        'model__alpha': Real(1e-4, 1e+3, 'log-uniform'),
        'model__learning_rate': Real(1e-4, 1e-1, 'log-uniform'),
        'model__hidden_layer_sizes': Categorical([(10,), (20,), (50,), (100,), (200,)]),
        'model__activation': Categorical(['relu', 'logistic', 'tanh'])
    }


def random_forest_search_space():
    return {
        'model': Categorical([RandomForestClassifier(random_state=0)]),
        'model__max_depth': Categorical([None, 5, 10, 20, 50]),
        'model__min_samples_split': Categorical([2, 5]),
        'model__n_estimators': Integer(5,150)
    }


def xgboost_search_space():
    return {
        'model': Categorical([xgboost.XGBClassifier(random_state=0, n_jobs=4, use_label_encoder=False, eval_metric='logloss')]),
        'model__max_depth': Integer([3, 10]),
        'model__min_child_weight': Categorical([1, 2, 5]),
        'model__max_delta_step': Categorical([0, 1]),
        'model__n_estimators': Integer(5,150),
        'model__gamma': Real(0, 1),
        
    }


def get_bugbug_buglevel(target):
    features = pd.read_csv('data/feature_extractor/features_buglevel.csv')
    features['revision'] = features['first_revision']
    features.set_index('revision', inplace=True)
    features

    # %%
    labeling = pd.read_csv('data/labeling/bugbug.csv')
    labeling.set_index('revision', inplace=True)
    labeling

    # %%
    features['target'] = labeling[target]
    assert features['first_id'].is_monotonic_increasing
    features = features.drop(['first_revision', 'first_id', 'revisions', 'ids'], axis=1)

    pos = features['target'].sum()
    neg = (1-features['target']).sum()
    print(f'{target}: {pos} positive {pos/(pos+neg)*100:.2f}% - negative {neg} {neg/(pos+neg)*100:.2f}% ')
    # %%
    y = np.array(features['target'])
    X = features.fillna(0).drop('target', axis=1)
    X = X.drop([c for c in X.columns if 'delta' in c], axis=1, errors='ignore')
    X = np.array(X)
    print(f'{X.shape=}')
    return X, y


import argparse
import sys
import os

if __name__ == '__main__':
    output_dir = 'experiments/results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, dest='n_iter', default=10,
        help='Number of samples per model.')

    parser.add_argument('--n_jobs', type=int, dest='n_jobs', default=-1,
        help='Number of parallel jobs.')

    parser.add_argument('--target', type=str, dest='target', default='performance',
        help='Classification target.')

    args = parser.parse_args(sys.argv[1:])
    print(f'{args.n_iter=}, {args.n_jobs=}, {args.target=}')

    X, y = get_bugbug_buglevel(args.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # %%
    opt = bayes_opt(X_train, X_test, y_train, y_test,
                    pipeline=default_pipeline(),
                    model_search_space=logistic_regression_search_space(),
                    imbalanced_search_space=imbalanced_search_space(),
                    n_iter=args.n_iter,
                    scoring='roc_auc',
                    n_jobs=args.n_jobs)

    save_cv_results(opt, os.path.join(output_dir, 'logistic_regression.csv'))
