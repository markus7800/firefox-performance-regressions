
import numpy as np
import pandas as pd
import pickle

def get_ml_data_traditional(labeling_name, target, kind='buglevel'):
    assert ('szz' not in labeling_name) or kind == 'commitlevel', 'SZZ labeling only allows commitlevel.'
    assert ('szz' not in labeling_name) or target == 'performance', 'SZZ labeling only allows performance target.'

    features = pd.read_csv(f'data/feature_extractor/features_{kind}.csv')

    if kind == 'buglevel':
        # labeling is based on bugnumber, that's why it is ok to index at
        # first revision of a commit group in case of kind=='buglevel'
        features['revision'] = features['first_revision']

    features.set_index('revision', inplace=True)

    labeling = pd.read_csv(f'data/labeling/{labeling_name}.csv')
    labeling.set_index('revision', inplace=True)

    features['target'] = labeling[target] # works because index is revision hash
    assert features['target'].isna().sum() == 0
    if kind == 'buglevel':
        assert features['first_id'].is_monotonic_increasing
        features = features.drop(['first_revision', 'first_id', 'revisions', 'ids'], axis=1)
    else: # kind == 'commitlevel'
        assert features['id'].is_monotonic_increasing
        features = features.drop(['id'], axis=1)

    pos = features['target'].sum()
    neg = (1-features['target']).sum()
    print(f'{target}: {pos} positive {pos/(pos+neg)*100:.2f}% - negative {neg} {neg/(pos+neg)*100:.2f}%')

    y = np.array(features['target'])
    X = features.fillna(0).drop('target', axis=1)
    # X = X.drop([c for c in X.columns if 'delta' in c], axis=1, errors='ignore')
    X = np.array(X)
    print(f'{X.shape=}\n')

    return X, y, features


def get_ml_data_bow(labeling_name, target, kind='buglevel'):
    assert ('szz' not in labeling_name) or kind == 'commitlevel', 'SZZ labeling only allows commitlevel.'
    assert ('szz' not in labeling_name) or target == 'performance', 'SZZ labeling only allows performance target.'

    with open(f'data/bow/bow_{kind}.pickle', 'rb') as f:
        X, feature_names = pickle.load(f)

    revisions = pd.read_csv(f'data/bow/revisions_{kind}.csv')
    if kind == 'buglevel':
        revisions['revision'] = revisions['first_revision']
    revisions.set_index('revision', inplace=True)

    labeling = pd.read_csv(f'data/labeling/{labeling_name}.csv')
    labeling.set_index('revision', inplace=True)
    
    revisions['target'] = labeling[target]
    assert revisions['target'].isna().sum() == 0
    
    if kind == 'buglevel':
        assert revisions['first_id'].is_monotonic_increasing
    else: # kind == 'commitlevel'
        assert revisions['id'].is_monotonic_increasing
    
    y = np.array(revisions['target'])

    
    pos = y.sum()
    neg = (1-y).sum()
    print(f'{target}: {pos} positive {pos/(pos+neg)*100:.2f}% - negative {neg} {neg/(pos+neg)*100:.2f}%')

    print(f'{X.shape=}\n')
    
    return X, y, feature_names
