# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

#################################################

# Project: Experiment With Multi-Armed Bandit Algorithms

#################################################


# Libraries

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
import plotly
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from src import experiment


# Define constants

ALGORITHMS = ['Greedy', 'Epsilon Greedy', 'Optimistic Initial Values', 'UCB1', 'Thompson Sampling']

# EPS = [0.01, 0.05, 0.1]

# OPTIMISTIC_INITIAL_VALUES = [1, 5, 10]

METRICS = ['', 'win_rate', 'performance', 'slotA', 'slotB', 'slotC']

METRICS_TITLE = ['', 'Win Rate', 'Performance', 'Slot A', 'Slot B', 'Slot C']

NUM_TRIALS = 100

NUM_REPETITIONS = 100

COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


# Instantiate dash app

app = dash.Dash()


# Create app layout

app.layout = html.Div([

    # header div
    html.Div([

        # title div
        html.H1('Experiment With The Best MAB Algorithms',
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

                    html.H3('How much do you want to bet? ($)'),

                    dcc.Input(id="investment", type="number", value=1000, style={'text-align': 'center'}),

                ], style={'margin': 'auto', 'width': '20%'}),

                # background image div
                html.Div([

                ], style={'height': 100,
                          'background-image': 'url("assets/mab.png")',
                          'background-repeat': 'no-repeat',
                          'background-size': '300px 100px',
                          'background-position': 'center top',
                          'margin': '10px'}),

                html.Div([

                    html.H3('Slot A'),

                    dcc.Input(id="A", type="number", value=0.25, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '6.5%'}),

                html.Div([

                    html.H3('Slot B'),

                    dcc.Input(id="B", type="number", value=0.50, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '6.5%'}),

                html.Div([

                    html.H3('Slot C'),

                    dcc.Input(id="C", type="number", value=0.75, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '6.5%'}),

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

        # LEFT SIDE: PERFORMANCE

        html.Div([

            html.Div([

                html.H3('Cumulative Performance'),

                dcc.Markdown(id='markdown_performance')

            ], style={'width': '100%', 'text-align': 'center', 'borderBottom': 'solid rgb(165, 42, 42) 0.5px'}),

            html.Div([

                dcc.Graph(id='performance_graph',
                          figure={})

            ],  style={'height': '300px', 'margin': '20px'}),

            html.Div([

                dcc.Graph(id='violin_plot',
                          figure={})

            ], style={'height': '300px', 'margin': '20px'}),

        ], style={'display': 'inline-block', 'width': '49.5%'}),

        # RIGHT SIDE: GAINS

        html.Div([

            html.Div([

                html.H3('Total Gains'),

                dcc.Markdown(id='markdown_gains')

            ], style={'width': '100%', 'text-align': 'center', 'borderBottom': 'solid rgb(165, 42, 42) 0.5px'}),

            html.Div([

                dcc.Graph(id='bar_chart',
                          figure={})

            ], style={'height': '300px', 'margin': '20px'}),

            html.Div([

                dcc.Graph(id='pie_chart',
                          figure={})

            ], style={'height': '300px', 'margin': '20px'}),

        ], style={'display': 'inline-block', 'width': '49.5%'}),

        #'border-right': 'solid rgb(165, 42, 42) 0.5px'

    ])

])


@app.callback([Output('performance_graph', 'figure'),
               Output('violin_plot', 'figure'),
               Output('markdown_performance', 'children')],
              [Input('submit-button-state', 'n_clicks'),
               State('investment', 'value'),
               State('A', 'value'),
               State('B', 'value'),
               State('C', 'value')])
def update_graph(n_clicks, investment, a, b, c):

    # store bandit probabilities entered by client
    bandit_probabilities = [a, b, c]

    # data to store list of relevant traces
    data = list()

    violin_data = {'algo': [], 'perf': []}

    for algorithm in ALGORITHMS:

        performances_avg = 0

        for repetition in range(NUM_REPETITIONS):

            # run experiment and add trace with algorithm performances
            output = experiment.run(investment, algorithm, NUM_TRIALS, bandit_probabilities)

            if np.sum(performances_avg) == 0:

                performances_avg = output['performances']

            else:

                performances_avg = np.mean([performances_avg, output['performances']], axis=0)

            violin_data['algo'].append(algorithm)
            violin_data['perf'].append(output['performances'][-1])

        data.append(go.Scatter(
            y=performances_avg,
            legendgroup=algorithm,
            name=algorithm,
        ))

    # add reference line for maximum performance (probability equals that of best bandit)
    data.append(go.Scatter(
        y=np.ones(NUM_TRIALS),
        legendgroup='Expected Performance',
        name='Expected Performance',
    ))

    figure_graph = go.Figure({'data': data,
                              'layout': {'legend': {'orientation':'h'},
                                         'margin': {'t': 30},
                                         'height': 300,
                                         'xaxis': dict(showticklabels=False)}})

    figure_graph.update_yaxes(visible=False, showticklabels=False)

    """figure_violin = go.Figure(data=[go.Violin(y=violin_data['perf'], x=violin_data['algo'], meanline_visible=True)],
                       layout={'legend': {'x': 1, 'y': 0.8},
                   'margin': {'t': 30},
                   'height': 300})"""

    figure_violin = px.violin(violin_data, x='algo', y='perf', color='algo', box=True, height=300)

    figure_violin.update_layout(showlegend=False, xaxis=dict(title=''), yaxis=dict(title=''))

    return figure_graph, figure_violin, f'''_Performances are averaged over 100 repetitions for {investment} trials._'''


@app.callback([Output('bar_chart', 'figure'),
               Output('pie_chart', 'figure'),
               Output('markdown_gains', 'children')],
              [Input('submit-button-state', 'n_clicks'),
               State('investment', 'value'),
               State('A', 'value'),
               State('B', 'value'),
               State('C', 'value')])
def update_bar_chart(n_clicks, investment, a, b, c):

    # store bandit probabilities entered by client
    bandit_probabilities = [a, b, c]

    # data to store list of relevant traces
    # data = list()

    data = {'algo': [], 'investment': []}

    for algorithm in ALGORITHMS:

        investment_avg = 0

        for repetition in range(NUM_REPETITIONS):

            # run experiment and add trace with algorithm performances
            output = experiment.run(investment, algorithm, investment, bandit_probabilities)

            if investment_avg == 0:

                investment_avg = output['investment']

            else:

                investment_avg = np.mean([investment_avg, output['investment']], axis=0)

        data['algo'].append(algorithm)
        data['investment'].append(investment_avg-investment)

    data['investment'] = [0 if i < 0 else i for i in data['investment']]

    colors = ['lightslategray', ] * len(data['investment'])

    most_profitable_index = np.argmax(data['investment'])

    colors[most_profitable_index] = 'crimson'

    figure_bar_chart = go.Figure(data=[go.Bar(
        x=data['algo'],
        y=data['investment'],
        text=[f'{int(i)}$' for i in data['investment']],
        textposition='auto',
        marker_color=colors)],
        layout={'legend': {'x': 1, 'y': 0.8},
                   'margin': {'t': 30},
                   'height': 300})

    figure_bar_chart.update_yaxes(visible=False, showticklabels=False)

    pull = np.zeros(len(data['investment']))

    pull[most_profitable_index] = 0.1

    figure_pie_chart = go.Figure(data=[go.Pie(
        labels=data['algo'],
        values=data['investment'],
        pull=pull)],
        layout={'legend': {'x': 1, 'y': 0.8},
                'margin': {'t': 30},
                'height': 300})

    figure_pie_chart.update_traces(marker=dict(colors=colors))

    figure_pie_chart.update_layout(margin=dict(t=0, b=0, l=0, r=0))

    return figure_bar_chart, figure_pie_chart, f'''_Gains are averaged over 100 repetitions for {investment} trials (t$ invested at trial t)._'''


# Execute application on server

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8081, debug=True)