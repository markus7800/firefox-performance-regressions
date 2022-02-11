
import numpy as np
import pandas as pd

def get_bugbug_data(target, kind='buglevel'):
    features = pd.read_csv(f'data/feature_extractor/features_{kind}.csv')

    if kind == 'buglevel':
        # labeling is based on bugnumber, that's why it is ok to index at
        # first revision of a commit group in case of kind=='buglevel'
        features['revision'] = features['first_revision']

    features.set_index('revision', inplace=True)

    labeling = pd.read_csv('data/labeling/bugbug.csv')
    labeling.set_index('revision', inplace=True)

    features['target'] = labeling[target] # works because index is revision hash
    if kind == 'buglevel':
        assert features['first_id'].is_monotonic_increasing
        features = features.drop(['first_revision', 'first_id', 'revisions', 'ids'], axis=1)
    else: # kind == 'commitlevel'
        assert features['id'].is_monotonic_increasing
        features = features.drop(['id'], axis=1)

    pos = features['target'].sum()
    neg = (1-features['target']).sum()
    print(f'{target}: {pos} positive {pos/(pos+neg)*100:.2f}% - negative {neg} {neg/(pos+neg)*100:.2f}% ')

    y = np.array(features['target'])
    X = features.fillna(0).drop('target', axis=1)
    # X = X.drop([c for c in X.columns if 'delta' in c], axis=1, errors='ignore')
    X = np.array(X)
    print(f'{X.shape=}\n')

    return X, y, features

def get_szz_commitlevel(name, target):
    features = pd.read_csv('data/feature_extractor/features_commitlevel.csv')
    features.set_index('revision', inplace=True)

    labeling = pd.read_csv(f'data/labeling/{name}.csv')
    labeling.set_index('revision', inplace=True)

    features['target'] = labeling[target] # works because index is revision hash
    assert features['id'].is_monotonic_increasing
    features = features.drop(['id'], axis=1)

    pos = features['target'].sum()
    neg = (1-features['target']).sum()
    print(f'{target}: {pos} positive {pos/(pos+neg)*100:.2f}% - negative {neg} {neg/(pos+neg)*100:.2f}% ')

    y = np.array(features['target'])
    X = features.fillna(0).drop('target', axis=1)
    # X = X.drop([c for c in X.columns if 'delta' in c], axis=1, errors='ignore')
    X = np.array(X)
    print(f'{X.shape=}\n')

    return X, y, features