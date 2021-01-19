import pandas as pd     #(version 1.0.0)
import plotly           #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px

import dash             #(version 1.9.1) pip install dash==1.9.1
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pathlib
from app import app

import os

#UPLOAD_FOLDER = r"C:\Users\s1025221\Desktop\New folder\uploads"



PATH = pathlib.Path(__file__).parent
#DATA_PATH = PATH.joinpath("../datasets").resolve()
UPLOAD_FOLDER = PATH.joinpath("../datasets").resolve()

df, l, k, d, all_col = None, None, None, None, None

def preprocess(dff):
    global df, l, k, d, all_col
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
    df=df.dropna()
    num_cols = list(df._get_numeric_data().columns)
    categorical_col=(list(set(df.columns) - set(num_cols) - set(date_col)))


    df.head()
    print (df[:5])
    k=list()
    all_col=list()
    for cat_col in categorical_col:
        k.append({'label':cat_col,'value':cat_col})
        all_col.append({'label':cat_col,'value':cat_col})

    l=list()
    for num in num_cols:
        l.append({'label':num,'value':num})
        all_col.append({'label':num,'value':num})

    d=list()
    for num in date_col:
        d.append({'label':num,'value':num})
        all_col.append({'label':num,'value':num})

    if len(d)==0:
        d.append({'label':'No Date column in given dataset','value':'No Date column in given dataset'})

    if len(num_cols)==0:
        print('Sorry Buddy, This App failed as number of numerical columns is zero.Before passing this dataset ,manually add a dummy numerical column')

layout = html.Div(id="distribution_time", children=[
    html.Div(id='json_data', style={'display': 'none'})
    ])


@app.callback(
    #Output(component_id='output_container', component_property='children'),
    Output('distribution_time', 'children'),
    [Input('upload-data', 'isCompleted')],
    [State('upload-data', 'filenames'),
     State('upload-data', 'upload_id')]
)
def initialize_graph(ic, f, uid):
    global d,l,k,all_col
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
        print('Hello Buddy, most probably app crashed because you didn\'s pass a csv file.Try converting this file to csv and try again' )
    preprocess(df)
    return html.Div([
        html.Div([
        html.Div([
            html.H3('Relation of Numerical Feature with Date',style={'text-align': 'center'}),
            html.Div([
            dcc.Dropdown(id="datecolumn",
                        style={'height': '30px', 'width': '300px','font-size': "70%"},
                         options=d,
                         multi=False,
                         value=d[0]['value'],
                         #style={'width': "40%"}
                         )],className='row1'),
            html.Div([
            dcc.Dropdown(id="num_date_col",
                        style={'height': '30px', 'width': '300px','font-size': "70%"},
                         options=l,
                         multi=False,
                         value=l[0]['value'],
                         #style={'width': "40%"}
                         )],className='row1'),
            dcc.Loading(id = "loading-icon3",
                    children=[html.Div(dcc.Graph(id='timelinetab', figure={}))], type="default")
            #dcc.Graph(id='timelinetab', figure={})
        ], className="row1"),


    ], className="row1"),

    html.Div([
            html.Div([
                html.H3('Relation of Numerical Feature with Date (aggregation type-average)',style={'text-align': 'center'}),
                html.Div([
                dcc.Dropdown(id="datecolumn2",
                            style={'height': '30px', 'width': '300px','font-size': "70%"},
                             options=d,
                             multi=False,
                             value=d[0]['value'],
                             #style={'width': "40%"}
                             )],className='row1'),
                html.Div([
                dcc.Dropdown(id="num_date_col2",
                            style={'height': '30px', 'width': '300px','font-size': "70%"},
                             options=l,
                             multi=False,
                             value=l[0]['value'],
                             #style={'width': "40%"}
                             )],className='row1'),
            html.Br(),
            dcc.RadioItems(id="aggregation_on",
        options=[
            {'label': 'Year', 'value': 'Year'},
            {'label': 'Month', 'value': 'Month'},
            {'label': 'Day', 'value': 'Day'}
        ],
        value='Year',
        labelStyle={'display': 'inline-block'}
    ),
    html.Br(),
                dcc.Loading(id = "loading-icon4",
                        children=[html.Div(dcc.Graph(id='timelinetab2', figure={}))], type="default")
                #dcc.Graph(id='timelinetab2', figure={})
            ], className="row1"),

            ], className='row1'),
    ])
@app.callback(Output("loading-output-3", "children"), Input("loading-icon3", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@app.callback(Output("loading-output-4", "children"), Input("loading-icon4", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@app.callback(
     Output(component_id='timelinetab', component_property='figure'),


     [Input(component_id='datecolumn', component_property='value'),
     Input(component_id='num_date_col', component_property='value')
     ]
)

def graph(feature1,feature2):
    if feature1=='No Date column in given dataset':
        tmp=go.Figure(data=[go.Table(
        header=dict(values=['Warning'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=['No date column. Cant display'],
                   fill_color='lavender',
                   align='left'))
                   ])
        return tmp
    line_chart = px.line(

            x=df[feature1],
            y=df[feature2],
            labels={
                     "x": feature1,
                     "y": feature2,

                 },
            title=feature2+' variation with '+feature1,
            #labels={'countriesAndTerritories':'Countries', 'dateRep':'date'},
            template='plotly_dark'
            )

    return line_chart

@app.callback(
     Output(component_id='timelinetab2', component_property='figure'),


     [Input(component_id='datecolumn2', component_property='value'),
     Input(component_id='num_date_col2', component_property='value'),
     Input(component_id='aggregation_on', component_property='value')]
)

def graph(feature1,feature2,agg_type):
    if feature1=='No Date column in given dataset':
        tmp=go.Figure(data=[go.Table(
        header=dict(values=['Warning'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=['No date column. Cant display'],
                   fill_color='lavender',
                   align='left'))
                   ])
        return tmp
    date_df=df[[feature1,feature2]]
    if agg_type=='Month':

        date_df['Month']=date_df[feature1].dt.month_name()
        date_df['month-int']=date_df[feature1].dt.month
        date_df=date_df.groupby(['Month','month-int']).mean().reset_index()
        date_df.sort_values('month-int',inplace=True)
    if agg_type=='Year':
        date_df['Year']=date_df[feature1].dt.year
        date_df=date_df.groupby([agg_type]).mean().reset_index()
    if agg_type=='Day':
        date_df['Day']=date_df[feature1].dt.day
        date_df=date_df.groupby([agg_type]).mean().reset_index()



    line_chart = px.line(

            x=date_df[agg_type],
            y=date_df[feature2],
            labels={
                     "x": agg_type,
                     "y": feature2,

                 },
            title=feature2+' variation with '+feature1 ,
            #labels={'countriesAndTerritories':'Countries', 'dateRep':'date'},
            template='plotly_dark'
            )

    return line_chart
