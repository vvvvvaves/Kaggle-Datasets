import pandas as pd
import numpy as np
from dash import Dash, html, Output, Input, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
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
       "margin-left": "150px",
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
    eda = html.Div([
                        html.H1('Exploratory Data Analysis', style={'text-align': 'center'}),
                        dbc.Tabs([dbc.Tab(label='General', tab_id='general'),
                                dbc.Tab(label='Winds', tab_id='winds'),
                                dbc.Tab(label='Violinplots', tab_id='violin'),],
                                id='tabs',
                                active_tab='general'),
                                html.Div(id='tab-content', children=[])
                         ],
                    style={
                        'width': '1200px',
                        'height': '500px'
                    })

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
            bar, fig = visualizer.winds_location('Sydney')
            return html.Div([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H6('Rain'),
                        dcc.Dropdown(['Yes', 'No'], 'Yes', id='rain-dropdown')],
                         style={'width': '25%', 'padding': "25px 0px 0px 50px"}), width=5),
                    dbc.Col(html.Div([
                        html.H6('Location'),
                        dcc.Dropdown(visualizer.preprocessor.locations['Location_reduced'], 'Sydney',
                                     id='winds-dropdown')],
                        style={'width': '35%', 'padding': "25px"})),
                ], className="g-0", style={'width': '1250px', 'height':'100px'}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='winddir-bar', figure=bar, style={'width': '500px', 'height': '550px'})),
                    dbc.Col(dcc.Graph(id='location-map', figure=fig))
                ], id='winds-plots', className="g-0")
            ])
        elif active_tab == 'violin':
            return html.Div([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H6('Numerical Feature'),
                        dcc.Dropdown(['Humidity3pm', 'Pressure3pm', 'WindGustSpeed', 'Pressure9am', 'Sunshine'], 'Humidity3pm',
                                     id='numerical-dropdown')
                    ], style={'width': '100%', 'padding': "0px 0px 0px 50px"}),  width=3),
                    dbc.Col(html.Div([
                        daq.BooleanSwitch(id='overlay-switch', on=False, label='overlay',  labelPosition='top')
                    ], style={'padding': '5px'}), width=1),
                    dbc.Col(html.Div([
                        daq.BooleanSwitch(id='box-switch', on=True, label='box',  labelPosition='top')
                    ], style={'padding': '5px'}), width=1),
                    dbc.Col(html.Div([
                        daq.BooleanSwitch(id='points-switch', on=True, label='points',  labelPosition='top')
                    ], style={'padding': '5px'}), width=1)
                ], style={'padding': "20px", 'height': '85px', 'wight': '200px'}),
                dcc.Graph(id='violin-plot',
                          figure=visualizer.violinplot('Humidity3pm', overlay=False,
                                                       box=True, points=True))
            ])
    return "No tab selected"


@app.callback(
    Output("winds-plots", "children"),
    [Input("winds-dropdown", "value"), Input("rain-dropdown", "value")]
)
def winds_choose_location(location, _rain):
    rain = 1 if _rain == 'Yes' else 0
    bar, fig = visualizer.winds_location(location, rain)
    return [
                    dbc.Col(dcc.Graph(id='winddir-bar', figure=bar, style={'width': '500px', 'height': '550px'})),
                    dbc.Col(dcc.Graph(id='location-map', figure=fig))
                ]

@app.callback(
    Output('violin-plot', 'figure'),
    [Input('numerical-dropdown', 'value'),
     Input('overlay-switch', 'on'),
     Input('box-switch', 'on'),
     Input('points-switch', 'on')]
)
def violin_plot(feature, overlay, box, points):
    return visualizer.violinplot(column=feature,
                                 overlay=overlay,
                                 box=box, points=points)


app.run_server(debug=True, port=7648)







