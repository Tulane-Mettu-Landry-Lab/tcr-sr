from scipy.special import softmax
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score
)
from .eval_metrics import EvalMetrics

def get_pred_scores(y_pred):
    return softmax(y_pred, axis=-1)[:, -1]

def get_pred_labels(y_pred, threshold=None):
    if threshold is None:
        return np.argmax(y_pred, axis=-1)
    else:
        return y_pred > threshold
 
@EvalMetrics.register('roc_curve')   
def metric_roc_curve(y_true, y_pred, datamodule, configs:dict={}):
    y_pred = get_pred_scores(y_pred)
    fpr, tpr, threholds = roc_curve(y_true, y_pred)
    return dict(fpr=fpr.tolist(), tpr=tpr.tolist(), threholds=threholds.tolist())

@EvalMetrics.register('roc_auc')  
def metric_roc_auc(y_true, y_pred, datamodule, configs:dict={}):
    y_pred = get_pred_scores(y_pred)
    return roc_auc_score(y_true, y_pred, max_fpr=configs.get('max_fpr'))

@EvalMetrics.register('accuracy')  
def metric_accuracy(y_true, y_pred, datamodule, configs:dict={}):
    y_pred = get_pred_labels(y_pred, threshold=None)
    return accuracy_score(y_true, y_pred)

@EvalMetrics.register('f1')  
def metric_f1(y_true, y_pred, datamodule, configs:dict={}):
    y_pred = get_pred_labels(y_pred, threshold=None)
    return f1_score(y_true, y_pred)

@EvalMetrics.register('precision')  
def metric_precision(y_true, y_pred, datamodule, configs:dict={}):
    y_pred = get_pred_labels(y_pred, threshold=None)
    return precision_score(y_true, y_pred)

@EvalMetrics.register('recall')  
def metric_recall(y_true, y_pred, datamodule, configs:dict={}):
    y_pred = get_pred_labels(y_pred, threshold=None)
    return recall_score(y_true, y_pred)

@EvalMetrics.register('epiwise_roc_auc')  
def metric_epiwise_roc_auc(y_true, y_pred, datamodule, configs:dict={}):
    df = datamodule.dataset.df
    scores = {}
    for epi in df['epi.aa'].unique():
        _sel = (df['epi.aa'] == epi)
        scores[epi] = roc_auc_score(y_true[_sel], get_pred_scores(y_pred[_sel]), max_fpr=configs.get('max_fpr'))
    return scores

@EvalMetrics.register('epiwise_accuracy')  
def metric_epiwise_roc_auc(y_true, y_pred, datamodule, configs:dict={}):
    df = datamodule.dataset.df
    scores = {}
    for epi in df['epi.aa'].unique():
        _sel = (df['epi.aa'] == epi)
        scores[epi] = accuracy_score(y_true[_sel], get_pred_labels(y_pred[_sel]))
    return scores

@EvalMetrics.register('forward')  
def metric_outputs(y_true, y_pred, datamodule, configs:dict={}):
    return dict(y_true=y_true.tolist(), y_pred=y_pred.tolist())