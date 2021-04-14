# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

"""
   Copyright © Investing.com
   Licensed under Private License.
   See LICENSE file for more information.
"""

#################################################

# Project: Affiliate Performance Optimisation
# A/B Test - Affiliation Campaigns
# Based on Chi-Square statistical test, alpha=.05

#################################################


# Libraries

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from src import (
    experiment,
    greedy,
    epsilon_greedy,
    ucb1,
    optimistic_initial_values,
    thompson_sampling)

# define constants

ALGORITHMS = ['Greedy', 'Epsilon Greedy', 'Optimistic Initial Values', 'UCB1', 'Thompson Sampling']

EPS = [0.01, 0.05, 0.1]

OPTIMISTIC_INITIAL_VALUES = [1, 5, 10]

METRICS = ['', 'win_rate', 'performance', 'slotA', 'slotB', 'slotC']

METRICS_TITLE = ['', 'Win Rate', 'Performance', 'Slot A', 'Slot B', 'Slot C']


# Helper functions


# instantiate dash app

app = dash.Dash()

# create app layout

app.layout = html.Div([

    # header div
    html.Div([

        # titlediv
        html.H1('Experiment With Multi-Armed Bandit Algorithms',
                style={'textAlign': 'center',
                       'color': 'white',
                       'fontSize': 40,
                       'fontWeight': 'bold',
                       'marginTop': 0,
                       'marginBottom': 0,
                       'background-color': 'rgb(165, 42, 42',
                       'borderTop': 'solid black'}),

        # filters div
        html.Div([

            html.Div([

                html.Div([

                    html.H3('Select Algorithm'),

                    dcc.Dropdown(id="algorithms", value='Greedy',
                                 options=[{'label': a, 'value': a} for a in ALGORITHMS]),

                ], style={'margin': 'auto', 'width': '20%'}),

                html.Div([

                    html.H3('Number of Trials'),

                    dcc.Input(id="num_trials", type="number", value=1000, style={'text-align': 'center'}),

                ], style={'margin': 'auto', 'width': '20%'}),

                # background image Div
                html.Div([

                ], style={'height': 200,
                          'background-image': 'url("assets/mab.png")',
                          'background-repeat': 'no-repeat',
                          'background-size': '650px 200px',
                          'background-position': 'center top',
                          'margin': '10px'}),

                html.Div([

                    html.H3('Slot A'),

                    dcc.Input(id="A", type="number", value=0.25, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '10%'}),

                html.Div([

                    html.H3('Slot B'),

                    dcc.Input(id="B", type="number", value=0.50, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '10%'}),

                html.Div([

                    html.H3('Slot C'),

                    dcc.Input(id="C", type="number", value=0.75, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '10%'}),

                html.Div([

                    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

                ], style={'margin': '10px'}),

            ], style={'textAlign': 'center',
                      'borderBottom': 'solid black',
                      'borderTop': 'solid black',
                      'background-color': 'rgb(96, 189, 104'}),

        ]),

    ]),

    html.Div([

        html.Div([

            html.Div([

                html.H3('Cumulative Performance')

            ], style={'width': '100%', 'text-align': 'center', 'borderBottom': 'solid rgb(165, 42, 42) 0.5px'}),

            # performance graph div
            html.Div([

                dcc.Graph(id='performance_graph',
                          figure={})

            ]),

        ], style={'display': 'inline-block', 'width': '50%', 'height': '500px', 'border-right': 'solid rgb(165, 42, 42) 0.5px'}),

        # blocks
        html.Div([

            html.Div([

                html.H3('Algorithms Comparison Over 100 Repetitions')

            ], style={'width': '100%', 'text-align': 'center', 'borderBottom': 'solid rgb(165, 42, 42) 0.5px'}),

            # metrics table div
            html.Div([

                dt.DataTable(id='metrics_table',
                             columns=[{'name': mn, 'id': mi} for mn, mi in zip(METRICS_TITLE, METRICS)],
                             data=[],
                             sort_action='native',
                             filter_action='native',
                             style_header={'fontWeight': 'bold', 'fontSize': 20},
                             style_table={'height': '300px'})

            ])

        ], style={'display': 'inline-block', 'width': '49.5%', 'height': '500px', 'verticalAlign': 'top'})

    ], style={'horizontalAlign': 'Top'}),

])


@app.callback(Output('performance_graph', 'figure'),
              [Input('submit-button-state', 'n_clicks'),
               State('algorithms', 'value'),
               State('num_trials', 'value'),
               State('A', 'value'),
               State('B', 'value'),
               State('C', 'value')])
def update_graph(n_clicks, algorithm, num_trials, a, b, c):

    bandit_probabilities = [a, b, c]

    # add relevant traces for the performance graph
    data = list()

    # add reference line for maximum performance (probability equals that of best bandit)

    data.append(go.Scatter(
        y=np.ones(num_trials),
        legendgroup='Expected Performance',
        name='Expected Performance',
    ))

    output = experiment.run(algorithm, num_trials, bandit_probabilities)

    data.append(go.Scatter(
        y=output['performances'],
        legendgroup=algorithm,
        name=algorithm,
    ))

    figure = {'data': data,
              'layout': {'legend': {'x': 1, 'y': 0.8},
                         'margin': {'t': 30},
                         'height': 385}}

    return figure


@app.callback(Output('metrics_table', 'data'),
              [Input('submit-button-state', 'n_clicks'),
               State('num_trials', 'value'),
               State('A', 'value'),
               State('B', 'value'),
               State('C', 'value')])
def update_metrics(n_clicks, num_trials, a, b, c):

    bandit_probabilities = [a, b, c]

    final_outputs = []

    num_repetitions = 100

    for algorithm in ALGORITHMS:

        final_output = {k: (0 if k != '' else algorithm) for k in METRICS}

        for r in range(1, num_repetitions + 1):

            output = experiment.run(algorithm, num_trials, bandit_probabilities)

            final_output['win_rate'] = (final_output['win_rate'] * (r - 1) + output['win_rates'][-1]) / r

            final_output['performance'] = (final_output['performance'] * (r - 1) + output['performances'][-1]) / r

            final_output['slotA'] = (final_output['slotA'] * (r - 1) + output['bandits_counter'][0]) / r

            final_output['slotB'] = (final_output['slotB'] * (r - 1) + output['bandits_counter'][1]) / r

            final_output['slotC'] = (final_output['slotC'] * (r - 1) + output['bandits_counter'][2]) / r

        final_output = {

            k: (v if isinstance(v, str) else round(v) if v > 1 else round(v*100)) for k, v in final_output.items()

        }

        final_outputs.append(final_output)

    return final_outputs


# Execute application on server

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
