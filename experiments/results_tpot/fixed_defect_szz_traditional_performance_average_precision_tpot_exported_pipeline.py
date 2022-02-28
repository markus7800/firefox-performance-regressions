import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=0)

# Average CV score on the training set was: 0.07963642855362951
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=1, min_samples_leaf=5, min_samples_split=19)),
    MinMaxScaler(),
    StackingEstimator(estimator=GaussianNB()),
    VarianceThreshold(threshold=0.0001),
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.1, fit_intercept=True, l1_ratio=1.0, learning_rate="constant", loss="modified_huber", penalty="elasticnet", power_t=0.1)),
    XGBClassifier(learning_rate=0.1, max_depth=2, min_child_weight=16, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
