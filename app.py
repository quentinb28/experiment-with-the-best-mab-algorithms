# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

#################################################

# Project: Experiment With Multi-Armed Bandit Algorithms

#################################################


# Libraries

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from src import experiment


# Define constants

ALGORITHMS = ['Greedy', 'Epsilon Greedy', 'Optimistic Initial Values', 'UCB1', 'Thompson Sampling']

METRICS = ['', 'win_rate', 'performance', 'slotA', 'slotB', 'slotC']

METRICS_TITLE = ['', 'Win Rate', 'Performance', 'Slot A', 'Slot B', 'Slot C']

NUM_REPETITIONS = 100


# Helper functions

def getGraphFigure(outputs):

    data = list()

    for output in outputs:

        data.append(go.Scatter(
            y=output['cumulative_performance_avg'],
            legendgroup=output['algorithm'],
            name=output['algorithm']
        ))

    # reference line = max performance
    data.append(go.Scatter(
        y=np.ones(len(output['cumulative_performance_avg'])),
        legendgroup='Expected Performance',
        name='Expected Performance'
    ))

    figure = go.Figure({'data': data,
                        'layout': {'legend': {'orientation':'h'},
                                   'margin': {'t': 30},
                                   'height': 300,
                                   'xaxis': dict(showticklabels=False)}})

    figure.update_yaxes(visible=False, showticklabels=False)

    return figure


def getViolinFigure(outputs):

    data = {'algorithms': [], 'final_performances': []}

    for output in outputs:

        data['algorithms'] += [output['algorithm']]*len(output['final_performances'])

        data['final_performances'] += output['final_performances']

    figure = px.violin(data, x='algorithms', y='final_performances', color='algorithms', box=True, height=300)

    figure.update_layout(showlegend=False, xaxis=dict(title=''), yaxis=dict(title=''))

    return figure


def getBarFigure(outputs):

    data = {'algorithms': [], 'investments': []}

    for output in outputs:

        data['algorithms'].append(output['algorithm'])

        data['investments'].append(output['investment'])

    # set negative gains to 0$
    data['investments'] = [0 if i < 0 else i for i in data['investments']]

    most_profitable_index = np.argmax(data['investments'])

    # highlight most profitable (color)
    colors = ['lightslategray', ] * len(data['investments'])

    colors[most_profitable_index] = 'crimson'

    figure = go.Figure(data=[go.Bar(
        x=data['algorithms'],
        y=data['investments'],
        text=[f'{int(i)}$' for i in data['investments']],
        textposition='auto',
        marker_color=colors)],
        layout={'legend': {'x': 1, 'y': 0.8},
                'margin': {'t': 30},
                'height': 300})

    return figure


def getPieFigure(outputs):

    data = {'algorithms': [], 'investments': []}

    for output in outputs:

        data['algorithms'].append(output['algorithm'])

        data['investments'].append(output['investment'])

    most_profitable_index = np.argmax(data['investments'])

    # highlight most profitable (color)
    colors = ['lightslategray', ] * len(data['investments'])

    colors[most_profitable_index] = 'crimson'

    # highlight most profitable (size)
    pull = np.zeros(len(data['investments']))

    pull[most_profitable_index] = 0.1

    figure = go.Figure(data=[go.Pie(
        labels=data['algorithms'],
        values=data['investments'],
        pull=pull)],
        layout={'legend': {'x': 1, 'y': 0.8},
                'margin': {'t': 30},
                'height': 300})

    figure.update_traces(marker=dict(colors=colors))

    figure.update_layout(margin=dict(t=0, b=0, l=0, r=0))

    return figure


# Instantiate dash app

app = dash.Dash()


# Create app layout

app.layout = html.Div([

    # HEADER

    html.Div([

        # title
        html.H1('Experiment With The Best MAB Algorithms',
                style={'textAlign': 'center',
                       'color': 'white',
                       'fontSize': 40,
                       'fontWeight': 'bold',
                       'marginTop': 0,
                       'marginBottom': 0,
                       'background-color': 'rgb(165, 42, 42',
                       'borderTop': 'solid black'}),

        # filters
        html.Div([

            html.Div([

                # initial investment
                html.Div([

                    html.H3('How much do you want to invest? ($)'),

                    dcc.Input(id="investment", type="number", value=1000, style={'text-align': 'center'}),

                ], style={'margin': 'auto', 'width': '20%'}),

                # background image
                html.Div([

                ], style={'height': 100,
                          'background-image': 'url("assets/mab.png")',
                          'background-repeat': 'no-repeat',
                          'background-size': '300px 100px',
                          'background-position': 'center top',
                          'margin': '10px'}),

                # win rate Slot A
                html.Div([

                    html.H3('Slot A'),

                    dcc.Input(id="A", type="number", value=0.25, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '6.5%'}),

                # win rate Slot B
                html.Div([

                    html.H3('Slot B'),

                    dcc.Input(id="B", type="number", value=0.50, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '6.5%'}),

                # win rate Slot C
                html.Div([

                    html.H3('Slot C'),

                    dcc.Input(id="C", type="number", value=0.75, style={'text-align': 'center'}),

                ], style={'display': 'inline-block', 'width': '6.5%'}),

                # submit button
                html.Div([

                    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

                ], style={'margin': '10px'}),

            ], style={'textAlign': 'center',
                      'borderBottom': 'solid black',
                      'borderTop': 'solid black',
                      'background-color': 'rgb(96, 189, 104'}),

        ]),

    ]),

    # BODY

    html.Div([

        # LEFT SIDE: PERFORMANCE

        html.Div([

            html.Div([

                html.H3('Performance'),

                dcc.Markdown(id='markdown_performance', children='''_Performance is averaged over 100 repetitions._''')

            ], style={'width': '100%', 'text-align': 'center', 'borderBottom': 'solid rgb(165, 42, 42) 0.5px'}),

            # performance graph
            html.Div([

                dcc.Graph(id='performance_graph',
                          figure={})

            ],  style={'height': '300px', 'margin': '20px'}),

            # performance violin plot
            html.Div([

                dcc.Graph(id='performance_violin',
                          figure={})

            ], style={'height': '300px', 'margin': '20px'}),

        ], style={'display': 'inline-block', 'width': '49.5%'}),

        # RIGHT SIDE: GAINS

        html.Div([

            html.Div([

                html.H3('Gains'),

                dcc.Markdown(id='markdown_gains', children='''_Gains are averaged over 100 repetitions._''')

            ], style={'width': '100%', 'text-align': 'center', 'borderBottom': 'solid rgb(165, 42, 42) 0.5px'}),

            # gains bar chart
            html.Div([

                dcc.Graph(id='investment_bar',
                          figure={})

            ], style={'height': '300px', 'margin': '20px'}),

            # gains pie chart
            html.Div([

                dcc.Graph(id='investment_pie',
                          figure={})

            ], style={'height': '300px', 'margin': '20px'}),

        ], style={'display': 'inline-block', 'width': '49.5%'}),

    ])

])


@app.callback([Output('performance_graph', 'figure'),
               Output('performance_violin', 'figure'),
               Output('investment_bar', 'figure'),
               Output('investment_pie', 'figure')],
              [Input('submit-button-state', 'n_clicks'),
               State('investment', 'value'),
               State('A', 'value'),
               State('B', 'value'),
               State('C', 'value')])
def update_figures(n_clicks, investment, a, b, c):

    bandit_probabilities = [a, b, c]

    outputs = []

    for algorithm in ALGORITHMS:

        output = experiment.run_repetitions(algorithm, investment, bandit_probabilities)

        outputs.append(output)

    graph_figure = getGraphFigure(outputs)

    violin_figure = getViolinFigure(outputs)

    bar_figure = getBarFigure(outputs)

    pie_figure = getPieFigure(outputs)

    return graph_figure, violin_figure, bar_figure, pie_figure


# Execute application on server

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8081, debug=True)