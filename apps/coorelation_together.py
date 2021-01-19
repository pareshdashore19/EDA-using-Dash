import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import chi2_contingency
import itertools
from dython.nominal import correlation_ratio
import dash_bootstrap_components as dbc
import base64
import io
import json
import os
import shutil

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pathlib
from app import app


PATH = pathlib.Path(__file__).parent
#DATA_PATH = PATH.joinpath("../datasets").resolve()
UPLOAD_FOLDER = PATH.joinpath("../datasets").resolve()


#UPLOAD_FOLDER = r"C:\Users\s1025221\Desktop\New folder\uploads"


categorical_col, numerical_column, df, df2, l, k, json_data_g, all_col, upload_last = None, None, None, None, None, None, None, None, None

categorical_association_table,a,correlation_ratio_table, fig1, fig2, fig3, fig4, fig5 =  None, None, None, None, None, None, None, None

def preprocess(dff):
    global l, k, df, df2, categorical_col, numerical_column,all_col
    df = dff
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)

    date_col=list()
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                date_col.append(col)
            except :
                pass

    num_cols = list(df._get_numeric_data().columns)
    categorical_col=(list(set(df.columns) - set(num_cols) - set(date_col)))
    if len(categorical_col)==0:
        df['No categorical column present'] = 'No categorical column present'
        categorical_col.append('No categorical column present')


    df2=df[categorical_col]
    df=df[num_cols]




    k=list()
    all_col=list()
    for cat_col in categorical_col:
        k.append({'label':cat_col,'value':cat_col})
        all_col.append({'label':cat_col,'value':cat_col})

    l=list()
    for num in num_cols:
        l.append({'label':num,'value':num})
        all_col.append({'label':num,'value':num})
    if len(num_cols)==0:
        print('Sorry Buddy, This App failed as number of numerical columns is zero.Before passing this dataset ,manually add a dummy numerical column')

layout = html.Div(id="distribution_corr", children=[
    html.Div(id='json_data_corr', style={'display': 'none'})
    ])


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df,df_corr):
    au_corr = df_corr.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr

def get_top_abs_correlations_categorical(df):
    au_corr = df.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr


@app.callback(
    #Output(component_id='output_container', component_property='children'),
    Output('distribution_corr', 'children'),
    [Input('upload-data', 'isCompleted')],
    [State('upload-data', 'filenames'),
     State('upload-data', 'upload_id')]
)
def initialize_graph(ic,f,uid):
    global df, df2, l, k, fig1, fig2, fig3, fig4, fig5, categorical_association_table,a,correlation_ratio_table,all_col, upload_last

    if not ic:
        return
    else:
        upload = ''
        mtime = -1
        for r, d, f  in os.walk(UPLOAD_FOLDER):
            for file in f:
                if '.csv' in f[0]:
                    path = os.path.join(r,file)
                    if os.path.getmtime(path) > mtime:
                        mtime = os.path.getmtime(path)
                        upload = path


    if upload != upload_last:
        try:
            df = pd.read_csv(upload)
        except:
            print('Hello Buddy, most probably app crashed because you didn\'s pass a csv file.Try converting this file to csv and try again' )

        preprocess(df)
        #container = "The categorical column selected by user was: {}".format(option_slctd)
        tmp=go.Figure(data=[go.Table(
        header=dict(values=['Warning'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=['Lot of features. Cant display'],
                   fill_color='lavender',
                   align='left'))
                   ])
        print('Spearman Calculation Started')
        corr=df.corr(method='spearman')
        if len(df.columns)<450:
            fig1 = px.imshow(corr,x=corr.columns,y=corr.columns,color_continuous_scale=px.colors.sequential.Oryel)
        else:
            fig1=tmp
        print('Spearman Calculation Completed')



        a = get_top_abs_correlations(df,corr)
        a=pd.DataFrame(data = a).reset_index()
        a=a.round(2)
        fig2 = go.Figure(data=[go.Table(
        header=dict(values=['Feature 1', 'Feature 2','Coorelation Index'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=a.to_numpy().T,
                   fill_color='lavender',
                   align='left',
                   font_size=8))
                   ])

    ######################################################################################################################################33
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x,y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
            return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


        cols = categorical_col

        corrM = np.zeros((len(cols),len(cols)))
        # there's probably a nice pandas way to do this
        print('Crammer V calculation starts')
        for col1, col2 in itertools.combinations(cols, 2):
            idx1, idx2 = cols.index(col1), cols.index(col2)
            corrM[idx1, idx2] = cramers_v(df2[col1], df2[col2])
            corrM[idx2, idx1] = corrM[idx1, idx2]



        x1=pd.DataFrame(corrM, index=cols, columns=cols)

        for i in x1.columns:
            x1.loc[i][i]=1 #to have coorelation between same features as 1
            
        print('Crammer V calculation Completed')
        fig3=px.imshow(x1,x=x1.columns,y=x1.columns,color_continuous_scale=px.colors.sequential.Oryel)

        categorical_association_table = get_top_abs_correlations_categorical(x1)
        categorical_association_table=pd.DataFrame(data = categorical_association_table).reset_index()

        categorical_association_table=categorical_association_table.round(2)
        fig4 = go.Figure(data=[go.Table(
        header=dict(values=['Feature 1', 'Feature 2','Coorelation Index'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=categorical_association_table.to_numpy().T,
                   fill_color='lavender',
                   align='left',
                   font_size=8))
                   ])
    ################################################################################################################################################################
        print('ETA(Correlation between Numerical and categorical features) calculation starts')
        corr_list=list()
        for i in df2.columns:
            for j in df.columns:
                corr_list.append((i,j,correlation_ratio(df2[i], df[j])))


        correlation_ratio_table=pd.DataFrame(corr_list, columns = ['Categorical_Feature' , 'Numerical_Feature', 'Correlation']).sort_values(by=['Correlation'], ascending=False)
        correlation_ratio_table=correlation_ratio_table.round(2)
        print('ETA(Correlation between Numerical and categorical features) calculation completed')
        fig5 = go.Figure(data=[go.Table(
            header=dict(values=['Categorical_Feature', 'Numerical_Feature','Coorelation Index'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=correlation_ratio_table.to_numpy().T,
                       fill_color='lavender',
                       align='left',
                       font_size=8))
                       ])

    upload_last = upload
    return html.Div([
        html.H1("Correlation Dashboards with Dash", style={'text-align': 'center'}),

        html.Div([
            html.Div([
                html.H3('Crammer V correlation matrix',style={'text-align': 'center'}),
                dcc.Graph(id='my_bee_map2',figure=fig3)
            ], className="row1"),

            html.Div([
                html.H3('Correlated categorical features',style={'text-align': 'center'}),
                dcc.Graph(id='test2',figure=fig4)
            ], className="row1"),
        ], className="row1"),

        html.Div([
            html.H3('Select the target categorical variable to look for correlation',style={'text-align': 'center'}),
            dcc.Dropdown(id="target_categorical_column",
                        style={'height': '30px', 'width': '300px','font-size': "70%"},
                         options=k,
                         multi=True,
                         value=[k[0]['value']],
                         #style={'width': "40%"}
                         ),
            html.Br(),
            dcc.Graph(id='correlation_for_target_categorical', figure={})
        ], className="row1"),

        html.Br(),


        html.Div([
                html.H3('Correlated ratio plot(ETA) Table',style={'text-align': 'center'}),
                dcc.Graph(id='coreelation_ratio_table',figure=fig5)
            ], className="row1"),

        html.Div([
                html.H3('Check correlation between numerical and categorical variabless',style={'text-align': 'center'}),
                dcc.Dropdown(id="select_col",
                            style={'height': '30px', 'width': '300px','font-size': "70%"},
                             options=all_col,
                             multi=False,
                             value=all_col[0]['value'],
                             #style={'width': "40%"}
                             ),
                html.Br(),
                dcc.Graph(id='coreelation_ratio_table_filter',figure={})
            ], className="row1"),

        html.Br(),

        html.Div([
            html.Div([
                html.H3('Spearman correlation matrix',style={'text-align': 'center'}),
                dcc.Graph(id='my_bee_map', figure=fig1)
            ], className="row1"),

            html.Div([
                html.H3('Correlated features',style={'text-align': 'center'}),
                dcc.Graph(id='test', figure=fig2)
            ], className="row1"),
        ], className="row1"),
        html.Br(),

        html.Div([
            html.Div([
                html.H3('Select the target numerical variable to look for correlation',style={'text-align': 'center'}),
                dcc.Dropdown(id="target_numerical_column",
                            style={'height': '30px', 'width': '300px','font-size': "70%"},
                             options=l,
                             multi=True,
                             value=[l[0]['value']],

                             #style={'width': "40%"}
                             ),
                html.Br(),
                dcc.Graph(id='correlation_for_target_numerical', figure={})
            ], className="row1"),



        ], className="row1"),
        html.Br(),

        html.Div([
            html.Div([
                html.H3('Regression and ScatterPlot between features',style={'text-align': 'center'}),
                html.Div([
                dcc.Dropdown(id="numerical_column1",
                            style={'height': '30px', 'width': '300px','font-size': "70%"},
                             options=l,
                             multi=False,
                             value=l[0]['value'],
                             #style={'width': "40%"}
                             )],className='row1'),
                html.Div([
                dcc.Dropdown(id="numerical_column2",
                            style={'height': '30px', 'width': '300px','font-size': "70%"},
                             options=l,
                             multi=False,
                             value=l[0]['value'],
                             #style={'width': "40%"}
                             )],className='row1'),
                html.Br(),
                #dcc.Graph(id='feature1', figure={})
            ], className="six columns"),
            dbc.Row(
            [
                dbc.Col(dcc.Loading(id = "loading-icon7",
                        children=[html.Div(dcc.Graph(id='feature1', figure={}))], type="default")),
                dbc.Col(dcc.Loading(id = "loading-icon8",
                        children=[html.Div(dcc.Graph(id='feature2', figure={}))], type="default"))
                #dbc.Col(dcc.Graph(id='feature2', figure=fig2)),

            ]
            )
            #html.Div([
                #html.H3('Correlated features',style={'text-align': 'center'}),
                #dcc.Graph(id='feature2', figure=fig2)
            #], className="six columns"),
        ], className="row1"),
    ])

@app.callback(Output("loading-output-7", "children"), Input("loading-icon7", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@app.callback(Output("loading-output-8", "children"), Input("loading-icon8", "value"))
def input_triggers_spinner2(value):
    time.sleep(1)
    return value

@app.callback(
     [Output(component_id='correlation_for_target_numerical', component_property='figure'),
     Output(component_id='correlation_for_target_categorical', component_property='figure'),
     Output(component_id='coreelation_ratio_table_filter', component_property='figure')],


     [Input(component_id='target_numerical_column', component_property='value'),
     Input(component_id='target_categorical_column', component_property='value'),
     Input(component_id='select_col', component_property='value')]
)

def updating_graph(feature1,feature2,feature3):
    global categorical_association_table,a,correlation_ratio_table

    a1=a[(a['level_1'].isin(feature1)) | (a['level_0'].isin(feature1))   ]
    a1=pd.DataFrame(data = a1).reset_index()
    a1.drop(['index'], axis=1,inplace=True)
    a1.sort_values(by=[0], ascending=False,inplace=True)
    fig_target = go.Figure(data=[go.Table(
    header=dict(values=['Feature 1', 'Feature 2','Coorelation Index'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=a1.to_numpy().T,
               fill_color='lavender',
               align='left',
               font_size=8)
               )
               ])

    categorical_association_table1=categorical_association_table[(categorical_association_table['level_1'].isin(feature2)) | (categorical_association_table['level_0'].isin(feature2))]
    categorical_association_table1=pd.DataFrame(data = categorical_association_table1).reset_index()
    categorical_association_table1.drop(['index'], axis=1,inplace=True)
    categorical_association_table1.sort_values(by=[0], ascending=False,inplace=True)
    figcategorical = go.Figure(data=[go.Table(
    header=dict(values=['Feature 1', 'Feature 2','Coorelation Index'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=categorical_association_table1.to_numpy().T,
               fill_color='lavender',
               align='left',
               font_size=8))
               ])

    correlation_ratio_table1=correlation_ratio_table[(correlation_ratio_table['Categorical_Feature']==feature3) | (correlation_ratio_table['Numerical_Feature']==feature3)]
    correlation_ratio_table1=pd.DataFrame(data = correlation_ratio_table1).reset_index()
    correlation_ratio_table1.drop(['index'], axis=1,inplace=True)

    correlation_ratio_table1.sort_values(by=['Correlation'], ascending=False,inplace=True)
    fig_all = go.Figure(data=[go.Table(
    header=dict(values=['Categorical_Feature', 'Numerical_Feature','Coorelation Index'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=correlation_ratio_table1.to_numpy().T,
               fill_color='lavender',
               align='left',
               font_size=8)
               )
               ])


    return fig_target,figcategorical,fig_all





@app.callback(
     [Output(component_id='feature1', component_property='figure'),
      Output(component_id='feature2', component_property='figure')],

     [Input(component_id='numerical_column1', component_property='value'),
     Input(component_id='numerical_column2', component_property='value'),
     ]
)

def updating_graph(feature1,feature2):
    fig = px.scatter(df, x=feature1, y=feature2, trendline="ols")
    results = px.get_trendline_results(fig)
    results_summary = results.px_fit_results.iloc[0].summary()
    results_as_html = results_summary.tables[0].as_html()
    h=pd.read_html(results_as_html)[0]
    h=h.round(2)


    vals=list()
    for col in h.columns:
        vals.append(list(h[col]))

    fig3 = go.Figure(data=[go.Table(#header=dict(values=['A Scores', 'B Scores']),
                 cells=dict(values=vals))
                     ])
    fig3.update_layout(width=700, height=900)

    return fig,fig3
