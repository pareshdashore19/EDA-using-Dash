import pandas as pd
import plotly.express as px  # (version 4.7.0)
#import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import base64
import io
import json
import os
import dash_bootstrap_components as dbc
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

df, l, k = None, None, None

def preprocess(dff):
    global df, l, k
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


    k=list()
    for cat_col in categorical_col:
        k.append({'label':cat_col,'value':cat_col})

    l=list()
    for num in num_cols:
        l.append({'label':num,'value':num})

    if len(num_cols)==0:
        print('Sorry Buddy, This App failed as number of numerical columns is zero.Before passing this dataset ,manually add a dummy numerical column')
    return



layout = html.Div(id="distribution", children=[
    html.Div(id='json_data', style={'display': 'none'})
    ])



@app.callback(
    #Output(component_id='output_container', component_property='children'),
    Output('distribution', 'children'),
    [Input('upload-data', 'isCompleted')],
    [State('upload-data', 'filenames'),
     State('upload-data', 'upload_id')]
)
def initialize_graph(ic, f, uid):
    global l,k
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

    try:
        df = pd.read_csv(upload)
    except:
        print('Hello Buddy, most probably app crashed because you didn\'s pass a csv file.Try converting this file to csv and try again')
    preprocess(df)
    return html.Div([

        html.H1("Distribution Dashboard", style={'text-align': 'center'}),
        html.Br(),
        html.H4("Distribution for Numerical Columns", style={'text-align': 'center'}),
        html.Pre(children="Select Numerical Column", style={"fontSize":"150%"}),
        dcc.Dropdown(id="numerical_column",
                     options=l,
                     multi=False,
                     value=l[0]['value'],
                    style={'height': '30px', 'width': '300px','font-size': "70%"}
                     ),

        html.Br(),
        dcc.Loading(id = "loading-icon",
                children=[html.Div(dcc.Graph(id='my_bee_map', figure={}))], type="default"),
        #dcc.Graph(id='my_bee_map', figure={}),
        html.Br(),
        html.Br(),
        html.H4("Distribution for Categorical Columns", style={'text-align': 'center'}),
        html.Pre(children="Select Categorical Column", style={"fontSize":"150%"}),

        dcc.Dropdown(id="categorical_column",
                     options=k,
                     multi=False,
                     value=k[0]['value'],
                     style={'height': '30px', 'width': '300px','font-size': "70%"}
                     ),
        html.Br(),
        dcc.Loading(id = "loading-icon2",
                children=[html.Div(dcc.Graph(id='test', figure={}))], type="default")
        #dcc.Graph(id='test', figure={}),
    ])

@app.callback(Output("loading-output-1", "children"), Input("loading-icon", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@app.callback(Output("loading-output-2", "children"), Input("loading-icon2", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@app.callback(
    #Output(component_id='output_container', component_property='children'),
    Output(component_id='my_bee_map', component_property='figure'),
    [Input(component_id='numerical_column', component_property='value')]
)
def update_graph(option_slctd):
    container = "The categorical column selected by user was: {}".format(option_slctd)

    dff = df.copy()
    fig = px.histogram(dff, x=option_slctd,
                   marginal="box", # or violin, rug
                   hover_data=df.columns,template='plotly_dark')

    return fig

@app.callback(
    #Output(component_id='output_container', component_property='children'),
     Output(component_id='test', component_property='figure'),
    [Input(component_id='categorical_column', component_property='value')]
)
def updating_graph(option_slctd):

    container = "The categorical column selected by user was: {}".format(option_slctd)

    dff = df.copy()
    #dff = dff[dff["gender"] == option_slctd]
    x=dff[option_slctd].value_counts()
    x=x.reset_index()
    #dff = dff[dff["Affected by"] == "Varroa_mites"]

    # Plotly Express
    fig = px.bar(
        data_frame=x,
        x="index",
        y=option_slctd,
        #color="gender",               # differentiate color of marks
        #opacity=0.9,                  # set opacity of markers (from 0 to 1)
        orientation="v",              # 'v','h': orientation of the marks
        barmode='relative',
        labels=dict(index=" ", option_slctd="Count") ,
                  # in 'overlay' mode, bars are top of one another.
                                      # in 'group' mode, bars are placed beside each other.
                                      # in 'relative' mode, bars are stacked above (+) or below (-) zero.
        #----------------------------------------------------------------------------------------------
        # facet_row='caste',          # assign marks to subplots in the vertical direction
        # facet_col='caste',          # assigns marks to subplots in the horizontal direction
        # facet_col_wrap=2,           # maximum number of subplot columns. Do not set facet_row!

        # color_discrete_sequence=["pink","yellow"],               # set specific marker colors. Color-colum data cannot be numeric
        # color_discrete_map={"Male": "gray" ,"Female":"red"},     # map your chosen colors
        # color_continuous_scale=px.colors.diverging.Picnic,       # set marker colors. When color colum is numeric data
        # color_continuous_midpoint=100,                           # set desired midpoint. When colors=diverging
        # range_color=[1,10000],                                   # set your own continuous color scale
        #----------------------------------------------------------------------------------------------
        # text='convicts',            # values appear in figure as text labels
        # hover_name='under_trial',   # values appear in bold in the hover tooltip
        # hover_data=['detenues'],    # values appear as extra data in the hover tooltip
        # custom_data=['others'],     # invisible values that are extra data to be used in Dash callbacks or widgets

        # log_x=True,                 # x-axis is log-scaled
        # log_y=True,                 # y-axis is log-scaled
        # error_y="err_plus",         # y-axis error bars are symmetrical or for positive direction
        # error_y_minus="err_minus",  # y-axis error bars in the negative direction

        #labels={"convicts":"Convicts in Maharashtra",
        #"gender":"Gender"},           # map the labels of the figure
        title="Frequency count for "+option_slctd, # figure title
                       # figure height in pixels
        template='plotly_dark',
                     # 'ggplot2', 'seaborn', 'simple_white', 'plotly',

                                      # 'plotly_white', 'plotly_dark', 'presentation',
                                      # 'xgridoff', 'ygridoff', 'gridon', 'none'

        # animation_frame='year',     # assign marks to animation frames
        # # animation_group=,         # use only when df has multiple rows with same object
        # # range_x=[5,50],           # set range of x-axis
        # range_y=[0,9000],           # set range of x-axis
        # category_orders={'year':    # force a specific ordering of values per column
        # [2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001]},

    )

    # barchart.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
    # barchart.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

    # barchart.update_layout(uniformtext_minsize=14, uniformtext_mode='hide',
    #                        legend={'x':0,'y':1.0}),
    # barchart.update_traces(texttemplate='%{text:.2s}', textposition='outside',
    #                        width=[.3,.3,.3,.3,.3,.3,.6,.3,.3,.3,.3,.3,.3])

    return fig
