import colorlover as cl
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from sklearn import metrics


def serve_roc_curve(model, X_test, y_test):
    if hasattr(model, "decision_function"):
        decision_test = model.decision_function(X_test)
    else:
        decision_test = model.predict_proba(X_test)[:,1]
    label = np.unique(y_test)[1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test, pos_label=label)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace0 = go.Scatter(
        x=fpr, y=tpr, mode="lines", name="Test Data", marker={"color": "#13c6e9"},
    )
    trace1 = go.Scatter(
        x=[0,0.2,0.4,0.6,0.8,1.0], y=[0,0.2,0.4,0.6,0.8,1.0], mode="lines", name="Random", marker={"color": "#ff0000"}
    )

    layout = go.Layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        height=500,
        margin=dict(l=100, r=100, t=25, b=100),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"color": "#a5b1cd"},
    )

    data = [trace0, trace1]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_pie_confusion_matrix(model, X_test, y_test, threshold):
    # Compute threshold
    # scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    labels = np.unique(y_test)
    if hasattr(model, "decision_function"):
        y_pred_test = model.decision_function(X_test)
    else:
        y_pred_test = model.predict_proba(X_test)[:,1]
    y_pred_test = [labels[1] if x > threshold else labels[0] for x in y_pred_test]
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    metric = [{
        "Accuracy"   : round(accuracy,3),
        "Precision"  : round(precision,3),
        "Recall"     : round(recall,3),
        "F1-Score"   : round(f1,3),
        "Specificity": round(specificity,3)
    }]
    x=np.unique(y_test).tolist()
    y=np.unique(y_test).tolist()
    z=[[tp,fn],[fp,tn]]
    figure = ff.create_annotated_heatmap(z,x=x,y=y,colorscale='Magma')
    figure['layout'].update(
        title="Confusion Matrix",
        margin=dict(l=100, r=100, t=25, b=50),
        height=300,
        xaxis=dict(title="PREDICTED VALUE"),
        yaxis=dict(title="ACTUAL VALUE"),
        legend=dict(bgcolor="#ffffff", font={"color": "#a5b1cd"}, orientation="h"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"color": "#a5b1cd", "size":12},
    )


    return metric, figure