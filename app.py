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
from src import epsilon_greedy, ucb1, optimistic_initial_values, thompson_sampling

# Define constants

ALGORITHMS = ['Epsilon Greedy', 'Optimistic Initial Values', 'UCB1', 'Thompson Sampling']

EPS = [0.01, 0.05, 0.1]

OPTIMISTIC_INITIAL_VALUES = [1, 5, 10]


# Helper functions


# Instantiate dash app

app = dash.Dash()

# Create app layout

app.layout = html.Div([

    # Header Div
    html.Div([

        # Title Div
        html.H1('How Much To Lose To Win?',
                style={'textAlign': 'center',
                       'fontSize': 40,
                       'fontWeight': 'bold',
                       'marginTop': 0,
                       'marginBottom': 0,
                       'border-bottom': 'solid black'}),

        # Background image Div
        html.Div([

        ], style={'height': 300,
                  'background-image': 'url("assets/mab.png")',
                  'background-repeat': 'no-repeat',
                  'background-size': '900px 300px',
                  'background-position': 'center top'}),

        # Filters Div
        html.Div([

            html.Div([

                html.Div([

                    html.H3('Select Algorithm'),

                    dcc.Dropdown(id="algorithms", value='Epsilon Greedy',
                                 options=[{'label': a, 'value': a} for a in ALGORITHMS]),

                ], style={'margin': 'auto', 'width': '20%'}),

                html.Div([

                    html.H3('Number of Trials'),

                    dcc.Input(id="num_trials", type="number", value=1000, max=1000),

                ], style={'margin': 'auto', 'width': '20%'}),

                html.Div([

                    html.H3('Probability A'),

                    dcc.Input(id="A", type="number", value=0.25),

                ], style={'display': 'inline-block', 'width': '20%'}),

                html.Div([

                    html.H3('Probability B'),

                    dcc.Input(id="B", type="number", value=0.50),

                ], style={'display': 'inline-block', 'width': '20%'}),

                html.Div([

                    html.H3('Probability C'),

                    dcc.Input(id="C", type="number", value=0.75),

                ], style={'display': 'inline-block', 'width': '20%'}),

                html.Div([

                    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

                ], style={'margin': '10px'}),

            ], style={'textAlign': 'center', 'borderBottom': 'solid black'}),

        ]),

    ]),

    # performance graph div
    html.Div([

        dcc.Graph(id='performance_graph',
                  figure={})

    ]),

    # metrics div
    html.Div([

        html.Div(id='metrics')

    ], style={'height': '200px', 'border': 'solid black'})

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

    if algorithm == 'Epsilon Greedy':

        for eps in EPS:

            output_eps = epsilon_greedy.run_experiment(num_trials, bandit_probabilities, eps)

            data.append(go.Scatter(
                y=output_eps['performances'],
                legendgroup=f'eps_{eps}',
                name=f'eps_{eps}',
            ))

    elif algorithm == 'Optimistic Initial Values':

        for oiv in OPTIMISTIC_INITIAL_VALUES:

            performances_oiv = optimistic_initial_values.run_experiment(num_trials, bandit_probabilities, oiv)

            data.append(go.Scatter(
                y=performances_oiv,
                legendgroup=f'oiv_{oiv}',
                name=f'oiv_{oiv}',
            ))

    elif algorithm == 'UCB1':

        performances_ucb1 = ucb1.run_experiment(num_trials, bandit_probabilities)

        data.append(go.Scatter(
            y=performances_ucb1,
            legendgroup='ucb1',
            name='ucb1',
        ))

    else:

        performances_ts = thompson_sampling.run_experiment(num_trials, bandit_probabilities)

        data.append(go.Scatter(
            y=performances_ts,
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
              'layout': {'yaxis': {'title': 'Cumulative Performance',
                                   'tickvals': np.linspace(0, 1, 11)},
                         'legend': {'orientation': 'h', 'x': 0.8, 'y': 1.4},
                         'height': 400}}

    return figure


@app.callback(Output('metrics', 'children'),
              [Input('submit-button-state', 'n_clicks'),
               State('algorithms', 'value'),
               State('num_trials', 'value'),
               State('A', 'value'),
               State('B', 'value'),
               State('C', 'value')])
def update_metrics(n_clicks, investment, algorithm, num_trials, a, b, c):

    bandit_probabilities = [a, b, c]

    if algorithm == 'Epsilon Greedy':

        outputs = dict()

        for eps in EPS:

            output_eps = epsilon_greedy.run_experiment(num_trials, bandit_probabilities, eps)

            outputs[eps] = dict()

            outputs[eps]['win_rate'] = output_eps['win_rates'][-1]

            outputs[eps]['performance'] = output_eps['performances'][-1]

            outputs[eps]['num_times_explored'] = output_eps['num_times_explored']

            outputs[eps]['num_times_exploited'] = output_eps['num_times_exploited']

            outputs[eps]['num_optimal'] = output_eps['num_optimal']

        return outputs

    elif algorithm == 'Optimistic Initial Values':

        pass

    elif algorithm == 'UCB1':

        pass

    else:

        pass


# Execute application on server

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
