import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline

def get_y_score(clf, X):
    if (type(clf) is Pipeline and hasattr(clf._final_estimator, 'predict_proba')) or hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X)[:,1]
    else:
        y_score = clf.decision_function(X)
    return y_score

def plot_precision_recall_curve_with_f1(clf, X, y, path=None):
    y_score = get_y_score(clf, X)

    precision, recall, thresholds = metrics.precision_recall_curve(y, y_score)
    display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    display.ax_.set_title('Precision-Recall Curve')
    # fig = plt.gcf()
    # if path:
    #     fig.savefig(path + 'precrec_curve.pdf')

    R, P = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))

    F1 = 2 * R * P / (R + P)
    CS = plt.contour(R, P, F1, levels=np.arange(0.1, 1, 0.1), alpha=0.5, colors='gray')
    display.ax_.clabel(CS, inline=1)

    D = precision + recall
    P = precision[D != 0]
    R = recall[D != 0]
    F1 = 2 * P * R / D[D != 0]
    a = np.argmax(F1)
    print(f'best F1: {F1[a]} at precision={P[a]} recall={R[a]}')
    plt.show()

def plot_roc_curve(clf, X, y, path=None):
    y_score = get_y_score(clf, X)


    fpr, tpr, thresholds = metrics.roc_curve(y, y_score) # specificity TN/N, recall (sensitivity) TP/P
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                  estimator_name='example estimator')
    display.plot()
    display.ax_.axline([0,0], [1,1], color='tab:Red')
    display.ax_.set_title('ROC Curve')
    # fig = plt.gcf()
    # fig.savefig(path + 'roc_curve.pdf')
    plt.show()
