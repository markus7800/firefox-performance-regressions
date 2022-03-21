import os
import pandas as pd
import numpy as np


from src.utils import *
from experiments.data_utils import *
from experiments.plot_utils import *
from experiments.hyperparam_tuning import *

output_dir = 'experiments/results'

def get_results(output_dir, data, feature_type, target, scoring, model):
    path = os.path.join(output_dir, f'{data}_{feature_type}_{target}_{scoring}_{model}.csv')
    results = pd.read_csv(path, index_col=0)
    results = results.replace({np.nan: None})
    return results


data_map = {}

data_map['traditional'] = {
    'bugbug_buglevel': lambda target, drop_columns=False: get_ml_data_traditional('bugbug', target, kind='buglevel', drop_columns=drop_columns),
    'fixed_defect_szz': lambda target, drop_columns=False: get_ml_data_traditional('fixed_defect_szz', target, kind='commitlevel', drop_columns=drop_columns)
}
data_map['bow'] = {
    'bugbug_buglevel': lambda target, _=False: get_ml_data_bow('bugbug', target, kind='buglevel'),
    'fixed_defect_szz': lambda target, _=False: get_ml_data_bow('fixed_defect_szz', target, kind='commitlevel')
}

def get_params(results):
    params = results[[c for c in results.columns if 'param' in c]]
    params = params.rename(lambda c: c[6:], axis=1) # remove param_
    return params


from sklearn.dummy import DummyClassifier

model_map = {
    'dummy': lambda: DummyClassifier(constant=1),
    'lr': lambda: LogisticRegression(random_state=0, solver='saga'),
    'svm': lambda: LinearSVC(random_state=0),
    'mlp': lambda: MLPClassifier(random_state=0),
    'rf': lambda: RandomForestClassifier(random_state=0),
    'xgb': lambda: xgboost.XGBClassifier(random_state=0, n_jobs=4, use_label_encoder=False, eval_metric='logloss')
}
model_names = {
    'dummy': 'Dummy Classifier',
    'lr': 'Logistic Regression',
    'svm': 'Support Vector Machine',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'mlp': 'Multi-Layer Perceptron',
    'tpot': 'TPOT'
}
models = list(model_names.keys())


def get_best_params(model, params):
    best_params = dict(params.iloc[0])

    best_params['model'] = model_map[model]()
    try:
        best_params['sampler'] = eval(best_params['sampler'])
    except:
        best_params['sampler'] = None
    
    return best_params

def get_best_f1_threshold(clf, X, y):
    y_score = get_y_score(clf, X)
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_score)
    precision = precision[:-1] # last one is 0
    recall = recall[:-1] # last one is 1

    D = precision + recall
    P = precision[D != 0]
    R = recall[D != 0]
    F1 = 2 * P * R / D[D != 0]
    T = thresholds[D != 0]

    amax = F1.argmax()
    return T[amax], F1[amax]

def get_tpot_pipeline(output_dir, data, feature_type, target, scoring):
    import re
    from tpot.export_utils import set_param_recursive
    path = os.path.join(output_dir, f'{data}_{feature_type}_{target}_{scoring}_tpot_exported_pipeline.py')

    with open(path,'r') as f:
        exported_pipeline_str = f.read()
        import_lines = []
        pipeline_lines = []
        state = 0
        for line in exported_pipeline_str.splitlines():
            if len(line) == 0:
                continue
            if line[0] == '#':
                state += 1
                continue
                
            if state == 0:
                import_lines.append(line)
            if state == 2:
                pipeline_lines.append(line)

        import_statements = '\n'.join(import_lines)
        exec(import_statements)
        pipeline_statements = '\n'.join(pipeline_lines)

        exported_pipeline = eval(pipeline_statements[20:]) 
        set_param_recursive(exported_pipeline.steps, 'random_state', 0)

        score = float(re.findall('Average CV score on the training set was: ([\d.]+)', exported_pipeline_str)[0])
        return exported_pipeline, None, score


def get_pipeline(output_dir, data, feature_type, target, scoring, model):
    if model == 'dummy':
        return model_map[model](), None, None
    if model == 'tpot':
        return get_tpot_pipeline(output_dir, data, feature_type, target, scoring)
        
    results = get_results(output_dir, data, feature_type, target, scoring, model)
    best_result = results.iloc[0]
    params = get_params(results)
    best_params = get_best_params(model, params)

    pipeline = default_pipeline() if model != 'svm' else svm_pipeline()
    pipeline.set_params(**best_params)
    if feature_type == 'bow':
        pipeline.set_params(scaler=None)

    return pipeline, best_params, best_result['mean_test_score']


from matplotlib.lines import Line2D
def plot_roc_auc_rec_prec_for_all_models(target, data, feature_type, scoring,
    fitted_pipelines, X_train, X_test, y_train, y_test, split='test',
    save=False, show=True, figsize=(12,8), ylim=[0,1], path=None, output_dir=None,
    roc_fig=None, roc_ax=None, pr_fig=None, pr_ax=None, colors=None, linestyles=None):

    if split == 'test':
        _X = X_test
        _y = y_test
    elif split == 'train':
        _X = X_train
        _y = y_train

    if not roc_fig or not roc_ax:
        roc_fig, roc_ax = plt.subplots(figsize=figsize)
        roc_ax.axline([0,0], [1,1], color='black', linestyle='dashed')
        roc_ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')

    if not pr_fig or not pr_ax:
        pr_fig, pr_ax = plt.subplots(figsize=figsize)
        pr_ax.set_ylim(ylim)
        R, P = np.meshgrid(np.linspace(0,1,100), np.linspace(ylim[0], ylim[1],100))
        F1 = 2 * R * P / (R + P)
        CS = pr_ax.contour(R, P, F1, levels=np.arange(0.1, 1, 0.1), alpha=0.5, colors='gray')
        pr_ax.clabel(CS, CS.levels, inline=1)
        random_guess_precision = _y.sum() / len(_y)
        pr_ax.axline([0,random_guess_precision], [1,random_guess_precision], color='black', linestyle='dashed')
        pr_ax.set(xlabel='Recall', ylabel='Precision', title='Recall-Precision Curve')

    for i, (model, pipeline) in enumerate(fitted_pipelines):
        y_score = get_y_score(pipeline, _X)

        color = colors[i] if colors else None
        linestyle = linestyles[i] if linestyles else None

        fpr, tpr, thresholds = metrics.roc_curve(_y, y_score)
        roc_ax.plot(fpr, tpr, label=model, color=color, linestyle=linestyle)

        precision, recall, thresholds = metrics.precision_recall_curve(_y, y_score)
        pr_ax.plot(recall, precision, label=model, color=color, linestyle=linestyle)

        # marking best threshold
        # threshold_train, f1_train = get_best_f1_threshold(pipeline, X_train, y_train)
        # y_score = get_y_score(pipeline, _X)
        # y_pred = y_score >= threshold_train

        # R = metrics.recall_score(_y, y_pred)
        # P =  metrics.precision_score(_y, y_pred)
        
        #pr_ax.plot([R, R], [0, P], linestyle="dashed", color=plt.gca().lines[-1].get_color(), linewidth=1)
        #pr_ax.plot([0, R], [P, P], linestyle="dashed", color=plt.gca().lines[-1].get_color(), linewidth=1)
        # pr_ax.annotate("",
        #     xy=(R, 0), xycoords='data',
        #     xytext=(R, -0.1*(ylim[1]-ylim[0])), textcoords='data',
        #     arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color=plt.gca().lines[-1].get_color()),
        #     )
        # pr_ax.annotate("",
        #     xy=(0, P), xycoords='data',
        #     xytext=(-0.03, P), textcoords='data',
        #     arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color=plt.gca().lines[-1].get_color()),
        #     )

    random_line = Line2D([0], [0], label='Random guessing', color='black', linestyle='dashed')

    handles, _ = roc_ax.get_legend_handles_labels() 
    handles.append(random_line)
    roc_ax.legend(handles=handles)

    
    handles, _ = pr_ax.get_legend_handles_labels()  
    f1_line = Line2D([0], [0], label='F1 isolines', color='gray', alpha=0.5)
    handles.extend([random_line, f1_line])
    pr_ax.legend(handles=handles)

    roc_fig.tight_layout()
    
    output_dir = "experiments/plots/" if not output_dir else output_dir
    if save:
        if not path:
            path = os.path.join(output_dir, f'plots/{data}_{feature_type}_{target}_{scoring}_roc_{split}.pdf')
        roc_fig.savefig(path + f'_roc_{split}.pdf')

    pr_fig.tight_layout()
    if save:
        if not path:
            path = os.path.join(output_dir, f'plots/{data}_{feature_type}_{target}_{scoring}_pr_{split}.pdf')
        pr_fig.savefig(path + f'_pr_{split}.pdf')
    
    if show:
        plt.show()

    return roc_fig, roc_ax, pr_fig, pr_ax