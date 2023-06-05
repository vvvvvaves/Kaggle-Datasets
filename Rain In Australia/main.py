import pandas as pd
import numpy as np
from dash import Dash, html, Output, Input, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
from visualizer import Visualizer
from predictor import Predictor
from preprocessor import Preprocessor

# Data and Logic
# =======================================

visualizer = Visualizer()

# Model
# =======================================

predictor = Predictor()

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# App Layout and Callbacks
# ===========================================================================
sidebar = html.Div([html.H4('Rain In Australia', className='display-5'),
                    html.Hr(),
                    dbc.Nav([dbc.NavLink("Main Page", href="/", active="exact"),
                            dbc.NavLink("EDA", href="/eda", active="exact"),
                            dbc.NavLink("Model Presentation", href="/model-presentation", active="exact"),
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
                        dbc.Tabs([dbc.Tab(label='Histograms', tab_id='histograms'),
                                dbc.Tab(label='Winds', tab_id='winds'),
                                dbc.Tab(label='Violinplots', tab_id='violin'),],
                                id='tabs',
                                active_tab='histograms'),
                                html.Div(id='tab-content', children=[])
                         ],
                    style={
                        'width': '1200px',
                        'height': '500px'
                    })

    auc_score, conf_m, accuracy, recall, precision, f1_score = predictor.model_presentation(threshold=0.5)

    presentation = html.Div([
                        html.H1('Model Presentation', style={'text-align': 'center'}),
                        html.Br(),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                    dcc.Graph(id='metrics-bar', figure=visualizer.metrics_barplot(0.5), style={"width": "550px"})
                            ], style={'width': '550px', 'height': '550px'}),
                            dbc.Col([
                                html.Div([
                                    dcc.Graph(id='conf_heatmap', figure=visualizer.confusion_heatmap(conf_m))
                                ], style={'width': '500px', 'height': '500px'}),
                                html.Div([dcc.Slider(0, 1, 0.1,
                                                     value=0.5,
                                                     id='threshold-slider'
                                                     )], style={'width': '500px'})
                            ])
                        ])
                         ],
                    style={
                        'width': '1200px',
                        'height': '500px'
                    })

    explanation = html.Div([
                        html.H1('Model Explanation', style={'text-align': 'center'}),
                        html.Br(),
                        html.Hr(),
                        dcc.Graph(figure=visualizer.shap_barplot(), style={"height": "600px"})
                            ],
                        style={
                        'width': '1200px',
                        'height': '500px'
                        })

    main_page = html.Div([
                        html.H1('Rain In Australia', style={'text-align': 'center'}),
                        html.Plaintext('Project made by Viktoriia Shkurenko', style={'text-align': 'center'}),
                        html.Hr(),
                        dcc.Graph(id='Australia-map', figure=visualizer.australia_map(), style={'position': 'absolute',
                                                                                                        'left': '300px'}),
                        html.Plaintext("\n"*17+"This map represents amounts of rain at  \t\t\t\nAustralian weather stations in the period  \t\t\t\nbetween 2007 and 2015.  \t\t\t\t\t\t\n\nData was taken from kaggle.com  \t\t\t\t\t", style={'text-align': 'right',
                                                                       'font-family': 'italic',
                                                                       'font-size': '17px'})
                    ],
                    style={
                        'width': '1200px',
                        'height': '500px'
                    })
    if pathname == "/":
        return main_page
    elif pathname == '/eda':
        return eda
    elif pathname == '/model-presentation':
        return presentation
    elif pathname == "/model-explanation":
        return explanation
    else:
        return '404. Page does not exist :('

@app.callback(
    [Output('metrics-bar', 'figure'), Output('conf_heatmap', 'figure')],
    Input('threshold-slider', 'value')
)
def move_threshold(threshold):
    auc_score, conf_m, accuracy, recall, precision, f1_score = visualizer.predictor.model_presentation(threshold)
    return visualizer.metrics_barplot(threshold=threshold), visualizer.confusion_heatmap(conf_m)

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")]
)
def show_tabs(active_tab):
    if active_tab:
        if active_tab == "histograms":
            months_bar_ = visualizer.rain_months('All')
            rain_week_roll = visualizer.raintoday_roll('All')
            return html.Div([
                html.Div([
                    html.H6('Location'),
                    dcc.Dropdown(['All']+list(visualizer.preprocessor.locations['Location_reduced']), 'All',
                             id='months-dropdown')
                    ], style={'width': '25%', 'padding': "25px 0px 0px 65px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='months-bar', figure=months_bar_, style={'width': '500px', 'height': '550px'})),
                    dbc.Col(dcc.Graph(id='weekroll-bar', figure=rain_week_roll, style={'width': '630px', 'height': '550px'}))
                ], id='months-plots', className="g-0"),
                html.Plaintext("The distributions of amounts of rain over the year differ with respect to location. On the left figure numbers reflect \ntotal amount of rain during a particular month between 2007 and 2015."),
                html.Plaintext("As the right histogram shows, the probability of rain rises with the amount of rainy days during the past week.")
            ])
        elif active_tab == "winds":
            bar, fig = visualizer.winds_location('Sydney')
            line = visualizer.windspeed('Sydney')
            winds_note = "Although winds come from all sorts of directions, it seems as though RainTomorrow is True only when winds blow from plains or the ocean."
            winds_note += "\nDue to that, it made sense for me to apply target encoding on wind directions with respect to locations."
            winds_note2 = "As you can also notice, during every month median wind speed is higher when there is rain tomorrow."
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
                ], id='winds-plots', className="g-0"),
                html.Plaintext(winds_note),
                html.Div([
                    dcc.Graph(id='windspeed-map', figure=line)
                ]),
                html.Plaintext(winds_note2),
            ])
        elif active_tab == 'violin':
            return html.Div([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H6('Numerical Feature'),
                        dcc.Dropdown(['Humidity3pm', 'Pressure3pm', 'WindGustSpeed',
                                      'Pressure9am', 'Sunshine', 'Rainfall'],
                                     'Humidity3pm',
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
                                                       box=True, points=True)),
                html.Plaintext("The difference in distributions of numerical features presented above gives us more insight into the differences between rainy \nand non-rainy days.")
            ])
    return "No tab selected"


@app.callback(
    [Output("months-bar", "figure"), Output('weekroll-bar', 'figure')],
    [Input("months-dropdown", "value")]
)
def months_bar(location):
    return visualizer.rain_months(location=location), visualizer.raintoday_roll(location=location)


@app.callback(
    Output("windspeed-map", "figure"),
    Input("winds-dropdown", "value")
)
def winds_linechart(location):
    return visualizer.windspeed(location)


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







