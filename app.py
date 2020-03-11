import time
import importlib
import base64
import io
import os
import flask
import pandas as pd
import socket
from joblib import dump, load

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import utils.dash_reusable_components as drc
import utils.figures as figs

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Pipelining"
server = app.server

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                html.Div(
                    className="container scalable",
                    children=[
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Machine Learning Pipelining",
                                    style={
                                        "textDecoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    children=[
                        html.Div(
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        html.Label("Load Dataset"),
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div([
                                                dcc.Input(
                                                    id='names',
                                                    placeholder='Select File...',
                                                    disabled=True),
                                                html.Button('Browse')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'textAlign': 'center',
                                                'margin': '20px'
                                            },
                                        ),
                                        html.Div(id='output-data-upload'),
                                        drc.NamedDropdown(
                                            name="Target Variable",
                                            id="input-target-variable",
                                            placeholder="Choose target variable...",
                                            style={
                                                'width': '100%',
                                            }
                                        ),
                                        drc.NamedSlider(
                                            name="Test Size",
                                            id="slider-dataset-test-size",
                                            min=0.1,
                                            max=0.5,
                                            step=0.1,
                                            marks={
                                                str(i): str(i)
                                                for i in [0.1, 0.2, 0.3, 0.4, 0.5]
                                            },
                                            value=0.2,
                                        ),
                                        drc.NamedDropdown(
                                            name="Feature Selection/Extraction",
                                            id="dropdown-feature-selection",
                                            options=[
                                                {
                                                    "label": "PCA",
                                                    "value": "pca"
                                                },
                                                {
                                                    "label": "Chi2",
                                                    "value": "chi2",
                                                },
                                                {
                                                    "label": "ANOVA F-value",
                                                    "value": "anova",
                                                },
  {
                                                    "label": "Mutual Information (Entropy)",
                                                    "value": "mi",
                                                },
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="pca",
                                        ),
                                        drc.NamedSlider(
                                            name="Number of Features",
                                            id="slider-num-feat",
                                            min=1,
                                            max=5,
                                            step=1,
                                            value=1,
                                        ),
                                    ],
                                ),
                        drc.Card(
                                    id="last-model-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Classifier",
                                            id="dropdown-classifier-selection",
                                            options=[
                                                {
                                                    "label": "Logistic Regression",
                                                    "value": "lr",
                                                },
                                                {
                                                    "label": "Gaussian Naive Bayes",
                                                    "value": "nb"
                                                },
                                                {
                                                    "label": "Decision Tree",
                                                    "value": "dt"
                                                },
                                                {
                                                    "label": "SVM Classifier",
                                                    "value": "svc",
                                                },
                                                {
                                                    "label": "Random Forest Classifier",
                                                    "value": "rf",
                                                },
                                                {
                                                    "label": "MLP Classifier",
                                                    "value": "mlp",
                                                },
                                                {
                                                    "label": "KNN Classifier",
                                                    "value": "knn",
                                                },
                                            ],
                                            value="lr",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="K-neighbors",
                                            id="slider-parameter-k",
                                            min=1,
                                            max=5,
                                            value=3,
                                            marks={
                                                str(i) : "{}".format(i)
                                                for i in range(1, 6)
                                            }
                                        ),
                                        drc.NamedSlider(
                                            name="Tol",
                                            id="slider-parameter-tol",
                                            min=-4,
                                            max=1,
                                            value=-4,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-4, 1)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="C",
                                            id="slider-parameter-C-coef",
                                            min=0,
                                            max=10,
                                            step=2,
                                            value=10,
                                            marks={
                                                str(i) : "{}".format(i / 10)
                                                for i in range(0, 12, 2)
                                            }
                                        ),
                                        drc.NamedSlider(
                                            name="Max Depth",
                                            id="slider-parameter-max-depth",
                                            min=4,
                                            max=32,
                                            value=32,
                                            step=4,
                                            marks={
                                                str(i): str(i) for i in range(4, 36, 4)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="Number of Estimators",
                                            id="slider-parameter-n-estimators",
                                            min=100,
                                            max=400,
                                            value=100,
                                            step=50,
                                            marks={
                                                str(i): str(i) for i in range(100, 450, 50)
                                            },
                                        ),
                                        drc.Card(
                                            id="button-card",
                                            children=[
                                                    drc.NamedSlider(
                                                        name="Threshold",
                                                        id="slider-threshold",
                                                        min=0,
                                                        max=1,
                                                        value=0.5,
                                                        step=0.01,
                                                        marks={
                                                            str(i): str(i)
                                                            for i in [0.0, 0.5, 1.0]
                                                        },
                                                    ),
                                                    html.Button(
                                                        "Reset Threshold",
                                                        id="button-zero-threshold",
                                                    ),
                                                    html.Div(
                                                        id="download-area",
                                                        className="block",
                                                        children = [],
                                                        style={
                                                            'margin':'auto',
                                                            'paddingTop': '20px'
                                                        }
                                                    ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.columns, html.Div([
        html.P("First 5 Entries from file "+ filename),
        dash_table.DataTable(
            data=df.head().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_header={'fontWeight': 'bold'},
            style_cell={'textAlign' : 'center'},
        ),
        html.P("Dataframe shape {} x {}".format(df.shape[0], df.shape[1])),
    ])


@app.callback(Output('names', 'value'),
              [Input('upload-data', 'filename')])
def update_filename(names):
        if names is not None:
            return names
        else:
            return ""


@app.callback([Output('input-target-variable', 'options'),
               Output('input-target-variable', 'value')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_target_variables(contents, filename):
    if contents is not None:
        children = parse_contents(contents, filename)
        columns = children[0]
        options = []
        for col in columns:
            options.append({
                'label': col,
                'value': col
            })
        return options, None
    else:
        return [], None

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = parse_contents(list_of_contents, list_of_names)
        return children[1]


@app.callback(
    [Output("slider-num-feat", "marks"),
     Output("slider-num-feat", "max"),
     Output("slider-num-feat", "value")],
    [Input("upload-data", "contents")],
    [State('upload-data', 'filename')]
)
def update_slider_num_feat(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = parse_contents(list_of_contents, list_of_names)
        maximum = len(children[0]) - 2
        marks = { str(i) : str(i) for i in range(1, maximum + 1)}
        if maximum < 4:
            val = 1
        else:
            val = maximum // 2
        return [marks, maximum, val]
    else:
        return {}, 5, 1

@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
)
def reset_threshold_center(n_clicks):
    return 0.5


@app.callback(
    Output("slider-parameter-k", "disabled"),
    [Input("dropdown-classifier-selection", "value")],
)
def disable_slider_param_k(classifier):
    return classifier not in ['knn']

@app.callback(
    Output("slider-parameter-tol", "disabled"),
    [Input("dropdown-classifier-selection", "value")],
)
def disable_slider_param_tol(classifier):
    return classifier not in ['lr', 'svc', 'mlp']

@app.callback(
    Output("slider-parameter-C-coef", "disabled"),
    [Input("dropdown-classifier-selection", "value")],
)
def disable_slider_param_c_coef(classifier):
    return classifier not in ['lr', 'svc']

@app.callback(
    Output("slider-parameter-max-depth", "disabled"),
    [Input("dropdown-classifier-selection", "value")],
)
def disable_slider_param_max_depth(classifier):
    return classifier not in ['dt', 'rf']

@app.callback(
    Output("slider-parameter-n-estimators", "disabled"),
    [Input("dropdown-classifier-selection", "value")],
)
def disable_slider_param_n_estimators(classifier):
    return classifier != 'rf'


def build_download_button(uri):
    """Generates a download button for the resource"""
    button = html.Form(
        action=uri,
        method="get",
        children=[
            html.Button(
                type="submit",
                style={
                    'margin': 'auto'
                },
                children=[
                    "download model"
                ]
            )
        ]
    )
    return button


def build_dataframe(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        return df

def get_feature_selector(feat_selector, n_feat):
    if feat_selector == 'pca':
        return TruncatedSVD(n_components=n_feat)
    elif feat_selector == 'chi2':
        return SelectKBest(chi2, k=n_feat)
    elif feat_selector == 'anova':
        return SelectKBest(f_classif, k=n_feat)
    elif feat_selector == 'mi':
        return SelectKBest(mutual_info_classif, k=n_feat)

def get_preprocessor(numeric_column, non_numeric_column):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
    ])
    non_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown ='ignore')),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('non_num', non_numeric_transformer, non_numeric_column),
            ('num', numeric_transformer, numeric_column)
        ]
    )
    return preprocessor


def get_classifier(clf, tol, c, max_depth, n_estimators, k):
    RANDOM_STATE = 1
    if clf == 'lr':
        return LogisticRegression(solver='lbfgs',tol=tol, C=c, random_state=RANDOM_STATE)
    elif clf == 'nb':
        return GaussianNB()
    elif clf == 'dt':
        return DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    elif clf == 'svc':
        return SVC(tol=tol, C=c, random_state=RANDOM_STATE)
    elif clf == 'rf':
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE)
    elif clf == 'knn':
        return KNeighborsClassifier(k)
    elif clf == 'mlp':
        return MLPClassifier(random_state=RANDOM_STATE, tol=tol)

def dump_pipeline(pipeline,path='models'):
    # handle model dump
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    model_name = f"{ip}.pkl"
    uri = f"{path}/{model_name}"
    dump(pipeline, uri)
    return uri

@app.callback(
    [Output("div-graphs", "children"),
     Output('download-area', "children")],
    [
        Input("upload-data", "filename"),
        Input("upload-data", "contents"),
        Input("input-target-variable", "value"),
        Input("slider-dataset-test-size", "value"),
        Input("dropdown-feature-selection", "value"),
        Input("slider-num-feat", "value"),
        Input("dropdown-classifier-selection", "value"),
        Input("slider-parameter-tol", "value"),
        Input("slider-parameter-C-coef", "value"),
        Input("slider-parameter-max-depth", "value"),
        Input("slider-parameter-n-estimators", "value"),
        Input("slider-parameter-k", "value"),
        Input("slider-threshold", "value")
    ],
)
def update_classifier_graph(
    filename,
    contents,
    target_variable,
    test_size,
    feature_selection,
    num_feat,
    classifier,
    tol,
    C,
    max_depth,
    n_estimators,
    k,
    threshold
):
    t_start = time.time()
    # building dataframe
    df = build_dataframe(contents, filename)
    if (target_variable is not None) and (target_variable in df.keys()):
        # splitting dataset
        X, y = df.drop([target_variable], axis=1), df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=1
        )
        numeric = X._get_numeric_data().columns
        non_numeric = list(set(X.columns) - set(numeric))
        tol = 10 ** tol
        C = C / 10

        clf = get_classifier(classifier, tol, C, max_depth, n_estimators, k)
        feature_selector = get_feature_selector(feature_selection, num_feat)
        preprocessor = get_preprocessor(numeric, non_numeric)
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('feat_selector',feature_selector),
                               ('clf',clf)])
        pipe.fit(X_train, y_train)
        print(feature_selector)
        print(clf)
        roc_figure = figs.serve_roc_curve(model=pipe, X_test=X_test, y_test=y_test)
        metric, confusion_figure = figs.serve_pie_confusion_matrix(
            model=pipe, X_test=X_test, y_test=y_test, threshold=threshold
        )
        params = [
        'Accuracy', 'Precision','Recall','F1-Score','Specificity'
        ]
        t_end = time.time()
        print("Time: {}\n\n".format(t_end - t_start))
        uri = dump_pipeline(pipe)
        return [[
            html.Div(
                id="graphs-container",
                children=[
                    dcc.Loading(
                        className="graph-wrapper",
                        children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                    ),
                    dcc.Loading(
                        className="graph-wrapper",
                        children=dcc.Graph(
                            id="graph-pie-confusion-matrix", figure=confusion_figure
                        ),
                    ),
                    dcc.Loading(
                        className="graph-wrapper",
                        children=dash_table.DataTable(
                                id='table',
                                columns=(
                                    [{'id': p, 'name': p} for p in params]
                                ),
                                style_header={
                                    'fontWeight': 'bold'
                                },
                                style_cell={'textAlign' : 'center'},
                                data=metric,
                        )
                    ),
                ],
            ),
        ], build_download_button(uri)]
    else:
        return [[
            html.Div(
                id="graphs-container",
                children=[
                    dcc.Loading(
                        className="graph-wrapper",
                        children=html.H5("Please Upload file and Choose Target Variable..."),
                    )
                ],
            ),
        ], None]


@app.server.route('/models/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'models'), path
    )

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)