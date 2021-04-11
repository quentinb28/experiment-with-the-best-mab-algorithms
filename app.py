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
from src import greedy, epsilon_greedy, epsilon_greedy_decay, ucb1, optimistic_initial_values, thompson_sampling

# define constants

ALGORITHMS = ['Greedy', 'Epsilon Greedy', 'Epsilon Greedy Decay', 'Optimistic Initial Values', 'UCB1', 'Thompson Sampling']

EPS = [0.01, 0.05, 0.1]

OPTIMISTIC_INITIAL_VALUES = [1, 5, 10]

METRICS = ['', 'win_rate', 'performance', 'num_times_explored', 'num_times_exploited', 'num_optimal']

METRICS_TITLE = ['', 'Win Rate', 'Performance', 'Explored', 'Exploited', 'Optimal']


# Helper functions


# instantiate dash app

app = dash.Dash()

# create app layout

app.layout = html.Div([

    # header div
    html.Div([

        # titlediv
        html.H1('Exploration Exploitation Dilemma',
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

                    html.H3('Probability A'),

                    dcc.Input(id="A", type="number", value=0.25, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '10%'}),

                html.Div([

                    html.H3('Probability B'),

                    dcc.Input(id="B", type="number", value=0.50, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '10%'}),

                html.Div([

                    html.H3('Probability C'),

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

        ], style={'display': 'inline-block', 'width': '49.5%', 'height': '500px', 'border-right': 'solid rgb(165, 42, 42) 0.5px'}),

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

    if algorithm == 'Greedy':

        output_greedy = greedy.run_experiment(num_trials, bandit_probabilities)

        data.append(go.Scatter(
            y=output_greedy['performances'],
            legendgroup='greedy',
            name='greedy',
        ))

    elif algorithm == 'Epsilon Greedy':

        for eps in EPS:

            output_eps = epsilon_greedy.run_experiment(num_trials, bandit_probabilities, eps)

            data.append(go.Scatter(
                y=output_eps['performances'],
                legendgroup=f'eps_{eps}',
                name=f'eps_{eps}',
            ))

    elif algorithm == 'Epsilon Greedy Decay':

        eps = 1

        output_eps_decay = epsilon_greedy_decay.run_experiment(num_trials, bandit_probabilities, eps)

        data.append(go.Scatter(
            y=output_eps_decay['performances'],
            legendgroup=f'eps_{eps}_decay',
            name=f'eps_{eps}_decay',
        ))

    elif algorithm == 'Optimistic Initial Values':

        for oiv in OPTIMISTIC_INITIAL_VALUES:

            performances_oiv = optimistic_initial_values.run_experiment(num_trials, bandit_probabilities, oiv)

            data.append(go.Scatter(
                y=performances_oiv['performances'],
                legendgroup=f'oiv_{oiv}',
                name=f'oiv_{oiv}',
            ))

    elif algorithm == 'UCB1':

        performances_ucb1 = ucb1.run_experiment(num_trials, bandit_probabilities)

        data.append(go.Scatter(
            y=performances_ucb1['performances'],
            legendgroup='ucb1',
            name='ucb1',
        ))

    else:

        performances_ts = thompson_sampling.run_experiment(num_trials, bandit_probabilities)

        data.append(go.Scatter(
            y=performances_ts['performances'],
            legendgroup='ts',
            name='ts',
        ))

    # add reference line for maximum performance (probability equals that of best bandit)

    data.append(go.Scatter(
        y=np.ones(num_trials),
        legendgroup='max_perf',
        name='max_perf',
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

    outputs = []

    num_repetitions = 100

    # greedy
    output = {k: (0 if k != '' else 'greedy') for k in METRICS}

    for t in range(1, num_repetitions + 1):

        output_greedy = greedy.run_experiment(num_trials, bandit_probabilities)

        output['win_rate'] = (output['win_rate'] * (t - 1) +
                              output_greedy['win_rates'][-1]) / t

        output['performance'] = (output['performance'] * (t - 1) +
                                 output_greedy['performances'][-1]) / t

        output['num_times_explored'] = (output['num_times_explored'] * (t - 1) +
                                        output_greedy['num_times_explored']) / t

        output['num_times_exploited'] = (output['num_times_exploited'] * (t - 1) +
                                         output_greedy['num_times_exploited']) / t

        output['num_optimal'] = (output['num_optimal'] * (t - 1) +
                                 output_greedy['num_optimal']) / t

    # round values
    output['win_rate'] = round(output['win_rate'] * 100)
    output['performance'] = round(output['performance'] * 100)
    output['num_times_explored'] = round(output['num_times_explored'])
    output['num_times_exploited'] = round(output['num_times_exploited'])
    output['num_optimal'] = round(output['num_optimal'])

    outputs.append(output)

    # eps greedy
    for eps in EPS:

        output = {k: (0 if k != '' else f'eps_{eps}') for k in METRICS}

        for t in range(1, num_repetitions + 1):

            output_eps = epsilon_greedy.run_experiment(num_trials, bandit_probabilities, eps)

            output['win_rate'] = (output['win_rate'] * (t-1) +
                                  output_eps['win_rates'][-1]) / t

            output['performance'] = (output['performance'] * (t - 1) +
                                     output_eps['performances'][-1]) / t

            output['num_times_explored'] = (output['num_times_explored'] * (t - 1) +
                                            output_eps['num_times_explored']) / t

            output['num_times_exploited'] = (output['num_times_exploited'] * (t - 1) +
                                             output_eps['num_times_exploited']) / t

            output['num_optimal'] = (output['num_optimal'] * (t - 1) +
                                     output_eps['num_optimal']) / t

        # round values
        output['win_rate'] = round(output['win_rate']*100)
        output['performance'] = round(output['performance'] * 100)
        output['num_times_explored'] = round(output['num_times_explored'])
        output['num_times_exploited'] = round(output['num_times_exploited'])
        output['num_optimal'] = round(output['num_optimal'])

        outputs.append(output)

    # eps greedy decay
    eps = 1

    output = {k: (0 if k != '' else f'eps_{eps}_decay') for k in METRICS}

    for t in range(1, num_repetitions + 1):

        output_eps_decay = epsilon_greedy_decay.run_experiment(num_trials, bandit_probabilities, eps)

        output['win_rate'] = (output['win_rate'] * (t - 1) +
                              output_eps_decay['win_rates'][-1]) / t

        output['performance'] = (output['performance'] * (t - 1) +
                                 output_eps_decay['performances'][-1]) / t

        output['num_times_explored'] = (output['num_times_explored'] * (t - 1) +
                                        output_eps_decay['num_times_explored']) / t

        output['num_times_exploited'] = (output['num_times_exploited'] * (t - 1) +
                                         output_eps_decay['num_times_exploited']) / t

        output['num_optimal'] = (output['num_optimal'] * (t - 1) +
                                 output_eps_decay['num_optimal']) / t

    # round values
    output['win_rate'] = round(output['win_rate'] * 100)
    output['performance'] = round(output['performance'] * 100)
    output['num_times_explored'] = round(output['num_times_explored'])
    output['num_times_exploited'] = round(output['num_times_exploited'])
    output['num_optimal'] = round(output['num_optimal'])

    outputs.append(output)

    # oiv
    for oiv in OPTIMISTIC_INITIAL_VALUES:

        output = {k: (0 if k != '' else f'oiv_{oiv}') for k in METRICS}

        for t in range(1, num_repetitions + 1):

            output_oiv = optimistic_initial_values.run_experiment(num_trials, bandit_probabilities, oiv)

            output['win_rate'] = (output['win_rate'] * (t - 1) +
                                  output_oiv['win_rates'][-1]) / t

            output['performance'] = (output['performance'] * (t - 1) +
                                     output_oiv['performances'][-1]) / t

            output['num_times_explored'] = (output['num_times_explored'] * (t - 1) +
                                            output_oiv['num_times_explored']) / t

            output['num_times_exploited'] = (output['num_times_exploited'] * (t - 1) +
                                             output_oiv['num_times_exploited']) / t

            output['num_optimal'] = (output['num_optimal'] * (t - 1) +
                                     output_oiv['num_optimal']) / t

        # round values
        output['win_rate'] = round(output['win_rate'] * 100)
        output['performance'] = round(output['performance'] * 100)
        output['num_times_explored'] = round(output['num_times_explored'])
        output['num_times_exploited'] = round(output['num_times_exploited'])
        output['num_optimal'] = round(output['num_optimal'])

        outputs.append(output)

    # ucb1
    output = {k: (0 if k != '' else 'ucb1') for k in METRICS}

    for t in range(1, num_repetitions + 1):

        output_ucb1 = ucb1.run_experiment(num_trials, bandit_probabilities)

        output['win_rate'] = (output['win_rate'] * (t - 1) +
                              output_ucb1['win_rates'][-1]) / t

        output['performance'] = (output['performance'] * (t - 1) +
                                 output_ucb1['performances'][-1]) / t

        output['num_times_explored'] = (output['num_times_explored'] * (t - 1) +
                                        output_ucb1['num_times_explored']) / t

        output['num_times_exploited'] = (output['num_times_exploited'] * (t - 1) +
                                         output_ucb1['num_times_exploited']) / t

        output['num_optimal'] = (output['num_optimal'] * (t - 1) +
                                 output_ucb1['num_optimal']) / t

    # round values
    output['win_rate'] = round(output['win_rate'] * 100)
    output['performance'] = round(output['performance'] * 100)
    output['num_times_explored'] = round(output['num_times_explored'])
    output['num_times_exploited'] = round(output['num_times_exploited'])
    output['num_optimal'] = round(output['num_optimal'])

    outputs.append(output)

    # ts
    output = {k: (0 if k != '' else 'ts') for k in METRICS}

    for t in range(1, num_repetitions + 1):

        output_ts = thompson_sampling.run_experiment(num_trials, bandit_probabilities)

        output['win_rate'] = (output['win_rate'] * (t - 1) +
                              output_ts['win_rates'][-1]) / t

        output['performance'] = (output['performance'] * (t - 1) +
                                 output_ts['performances'][-1]) / t

        output['num_times_explored'] = (output['num_times_explored'] * (t - 1) +
                                        output_ts['num_times_explored']) / t

        output['num_times_exploited'] = (output['num_times_exploited'] * (t - 1) +
                                         output_ts['num_times_exploited']) / t

        output['num_optimal'] = (output['num_optimal'] * (t - 1) +
                                 output_ts['num_optimal']) / t

    # round values
    output['win_rate'] = round(output['win_rate'] * 100)
    output['performance'] = round(output['performance'] * 100)
    output['num_times_explored'] = round(output['num_times_explored'])
    output['num_times_exploited'] = round(output['num_times_exploited'])
    output['num_optimal'] = round(output['num_optimal'])

    outputs.append(output)

    return outputs


# Execute application on server

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
