import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_uploader as du
# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app
# import all pages in the app
from apps import bar_testing,coorelation_together, home, timeline
import pathlib
import os
import shutil
# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
PATH = pathlib.Path(__file__).parent
UPLOAD_FOLDER = PATH.joinpath("./datasets").resolve()
du.configure_upload(app, UPLOAD_FOLDER)

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/apps/home"),
        dbc.DropdownMenuItem("Distribution Tab", href="/apps/bar_testing"),
        dbc.DropdownMenuItem("Correlation Tab", href="/apps/coorelation_together"),
        dbc.DropdownMenuItem("Timeline Tab", href="/apps/timeline")
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/virus.png", height="40px")),
                        dbc.Col(dbc.NavbarBrand("EDA with DASH", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([

    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div([
        du.Upload(
            id='upload-data',
            filetypes=['csv'],
            max_file_size=1800,
            text = 'Browse and select CSV file',
            text_completed = 'Completed :',
            pause_button = False,
            cancel_button = True

        ),], className="row"),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/bar_testing':
        return bar_testing.layout
    if pathname == '/apps/coorelation_together':
        return coorelation_together.layout
    if pathname == '/apps/timeline':
        return timeline.layout
    else:
        for root, dirs, files in os.walk(str(UPLOAD_FOLDER)):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        return home.layout

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
