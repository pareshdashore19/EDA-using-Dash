import dash_html_components as html
import dash_bootstrap_components as dbc
import dash

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the EDA dashboard", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='EDA is a time consuming process. This app would help you in understanding your data better without writing any piece of code.'
            'Use Explore Tab in the top right corner to navigate between different pages. Try restricting your input csv file to 300MB. Tap on "Browse and select CSV file" and select the file which you are trying to explore. The apps requires the csv file to have atleast one numerical column otherwise it may crash    '
                                     )
                    , className="mb-4")
            ]),

        dbc.Row([
            dbc.Col(html.H5(children='It consists of three pages:1) Distribution Tab: Which lets you look at the distribution of Categorical and Numerical Features. '
                                     '2) Correlation Tab: Which specifies correlation between features. '
                                     '3)Timeline Tab: Provides a timeline for Numerical features. ')
                    , className="mb-5")
        ]),

        dbc.Row([




            dbc.Col(dbc.Card(children=[html.H3(children='Read the README file to understand Dashboard in Detail',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://git.syngentaaws.org/Paresh.Dashore/eda-using-dash/-/tree/master",
                                                  color="primary",
                                                  className="mt-3",
                                                  target='-blank'),

                                       ],
                             body=True, color="dark", outline=True)
                    ,  className="h-50",width={"size": 4, "order": 12,"offset": 4})
        ], className="h-50",align='center')




    ])

])

# needed only if running this as a single page app
# if __name__ == '__main__':
#     app.run_server(host='127.0.0.1', debug=True)
