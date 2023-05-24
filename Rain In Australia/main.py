import pandas as pd
import numpy as np
from dash import Dash, html, Output, Input, dcc
import dash_bootstrap_components as dbc
from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
from visualizer import Visualizer
from preprocessor import Preprocessor

# Data and Logic
# =======================================

visualizer = Visualizer()

# =======================================

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# App Layout and Callbacks
# ===========================================================================
sidebar = html.Div([html.H4('Rain In Australia', className='display-5'),
                    html.Hr(),
                    dbc.Nav([dbc.NavLink("EDA", href="/", active="exact"),
                            dbc.NavLink("Prediction", href="/prediction", active="exact"),
                            dbc.NavLink("Model Explanation", href="/model-explanation", active="exact")],
                            vertical=True,
                            pills=True)],
                    style={
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "bottom": 0,
                        "width": "250px",
                        "padding": "30px 15px",
                        "background-color": "#f8f9fa",
                    })

content = html.Div(
    id='page',
    style={
       "margin-left": "288px",
        "margin-right": "33px",
        "padding": "30px 15px",
    }
)
app.layout = dbc.Container([
    dcc.Location(id='url'),
    sidebar,
    content
])

@app.callback(
    Output('page', 'children'),
    [Input('url', 'pathname')]
)
def change_page(pathname):
    eda = dbc.Container([
                        html.H1('Exploratory Data Analysis', style={'text-align': 'center'}),
                        dbc.Tabs([dbc.Tab(label='General', tab_id='general'),
                                dbc.Tab(label='Winds', tab_id='winds'),
                                dbc.Tab(label='Scatterplots', tab_id='scatter'),],
                                id='tabs',
                                active_tab='general'),
                                html.Div(id='tab-content', children=[])
                         ])

    if pathname == '/':
        return eda
    elif pathname == '/prediction':
        return 'No page yet'
    elif pathname == "/model-explanation":
        return 'No page yet'
    else:
        return '404. Page does not exist :('

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")]
)
def show_tabs(active_tab):
    if active_tab:
        if active_tab == "general":
            return active_tab
        elif active_tab == "winds":
            return dcc.Graph(id='location-map', figure=visualizer.winds_location('Sydney'))
        elif active_tab == 'scatter':
            return active_tab
    return "No tab selected"


app.run_server(debug=True, port=7648)















# app.layout = dbc.Container([
#
#     html.H1('Australia Rain Prediction', style={'text-align': 'center'}),
#     dbc.Tabs([
#         dcc.Store(id='store'),
#         dbc.Tab(label='EDA', tab_id='eda'),
#         dbc.Tab(label='Prediction', tab_id='prediction'),
#         dbc.Tab(label='Model Interpretation', tab_id='model'),
#         ],
#         id='tabs',
#         active_tab='eda'),
#         html.Div(id='tab-content')
# ])
# @app.callback(
#     Output("tab-content", "children"),
#     [Input("tabs", "active_tab")]
# )
# def show_tabs(active_tab):
#     if active_tab:
#         if active_tab == "eda":
#             return dbc.Container([
#                 html.Br(),
#                 html.P('Text'), #Introduction
#                 dbc.Row([
#                     dbc.Col()
#                 ])
#             ])
#         elif active_tab == "prediction":
#             return 'p'
#         elif active_tab == 'model':
#             return 'm'
#     return "No tab selected"

