from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table  # pip install dash
import plotly.graph_objects as go
import plotly.express as px
import dash_ag_grid as dag                       # pip install dash-ag-grid
import dash_bootstrap_components as dbc          # pip install dash-bootstrap-components
import pandas as pd                             # pip install pandas

import matplotlib                                # pip install matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from scipy.optimize import curve_fit
from scipy.fft import fft, ifft, fftfreq
import scipy.stats as stats

import numpy as np
import random

#Blinding
f_ref = 0.2291 #MHz
dR = (100*random.random() - 50)

#Fit <parameters
fit_parameter_columns = [
    {'name': 'Parameter Name', 'id': 'Parameter Name'},
    {'name': 'Value', 'id': 'Value', 'type': 'numeric', 'format': {'specifier': '.6g'}},
    {'name': 'Uncertainty', 'id': 'Uncertainty', 'type': 'numeric', 'format': {'specifier': '.6g'}},
]

#The input data
df = pd.read_csv("Run3B.csv")

#convert time to us
df['time']/=1000.0

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

divFlex = {"display": "flex", "align-content": "stretch"}
labelFlex = {"width": "50%"}
slideFlex = {"width": "50%"}
slideFlexFull = {"width": "100%"}

app.layout = dbc.Container([
    html.H1("Interactive Wiggle Fit", className='mb-2', style={'textAlign':'center'}),

    dbc.Row([
        dbc.Col([
            html.H2('Histogram Options'),
            dcc.Checklist(
                id='hist-options',
                options=[
                    {'label': ' Set y-axis to log scale', 'value' : 'log-y'}
                ],
                value=[],
                labelStyle={'display': 'block'}
            )
        ], width=3),
        dbc.Col([
            dcc.Graph(id='wiggle-plot', figure={})
        ], width=9)
    ]),

    dbc.Row([
        dbc.Col([
            html.H2('Fit Options'),

            html.Div(children=[html.Label('Fit start time [us]:', style=labelFlex),html.Div(
                dcc.Slider(
                    id='fit-start-time',
                    min=30,
                    max=230,
                    step=1,
                    value=30,
                    marks={i: f'{i}' for i in range(30,250,100)}
                )
            ,style=slideFlex)],style=divFlex),

            html.Div(children=[html.Label('Fit end time [us]:', style=labelFlex),html.Div(
                dcc.Slider(
                    id='fit-end-time',
                    min=300,
                    max=500,
                    step=1,
                    value=500,
                    marks={i: f'{i}' for i in range(300,501,100)}
                )
            ,style=slideFlex)],style=divFlex),
            html.Div(children=[html.Label('N0:', style=labelFlex)]),
            html.Div(children=[html.Div(
                dcc.Slider(
                    id='N0',
                    min=4e6,
                    max=8e6,
                    step=1e4,
                    value=5.825e6,
                    marks={4000000:'4M',6000000:'6M',8000000:'8M'}
                )
            ,style=slideFlexFull)],style=divFlex),

            html.Div(children=[html.Label('tau [us]:', style=labelFlex)]),
            html.Div(children=[html.Div(
                dcc.Slider(
                    id='tau',
                    min=0,
                    max=100,
                    step=1,
                    value=64.44,
                    marks={i: f'{i:0.0f}' for i in range(0,101,20)}
                )
            ,style=slideFlexFull)],style=divFlex),

            html.Div(children=[html.Label('A:', style=labelFlex)]),
            html.Div(children=[html.Div(
                dcc.Slider(
                    id='A',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.374581,
                    marks={0:"0",0.25:"0.25",0.5:"0.50",0.75:"0.75",1:"1.0"}
                )
            ,style=slideFlexFull)],style=divFlex),

            html.Div(children=[html.Label('R [ppm]:', style=labelFlex)]),
            html.Div(children=[html.Div(
                dcc.Slider(
                    id='R',
                    min=-200,
                    max=200,
                    step=1,
                    value=0,
                    marks={i: f'{i:0.0f}' for i in range(-200,201,100)}
                )
            ,style=slideFlexFull)],style=divFlex),

            html.Div(children=[html.Label('phi [rad]:', style=labelFlex)]),
            html.Div(children=[html.Div(
                dcc.Slider(
                    id='phi',
                    min=0,
                    max=6.28,
                    step=0.01,
                    value=2.16682,
                    marks={i: f'{i}' for i in range(0,7,1)}
                )
            ,style=slideFlexFull)],style=divFlex),

            html.Button('Do the Fit!', id='do-fit-button'),

            html.H2('Fit Results',style={'margin-top': '20px'}),
            dash_table.DataTable(
                id='fit-results',
                columns=fit_parameter_columns,
                #data=[],
                #editable=True,
                #filter_action='native',  # Allows filtering of data
                #sort_action='native',    # Allows sorting of data
                #sort_mode='multi',       # Allows multiple column sorting
                #row_selectable='single', # Allows selection of a single row
                #selected_rows=[],        # Initialize selected_rows to be empty
                page_size=10,             # Number of rows per page
                style_table={'overflowX': 'auto'},  # Enable horizontal scrolling if content overflows
            )

        ], width=3),
        dbc.Col([
            dcc.Graph(id='residuals', figure={}),
            dcc.Graph(id='fft', figure={})
        ], width=9)
    ]),

])

def has_trace_with_name(figure, name):
    for trace in figure.data:
        if trace.name == name:
            return True
    return False

# Plot the data
@app.callback(
    Output('wiggle-plot', 'figure'),
    Output('residuals', 'figure'),
    Output('fft', 'figure'),
    Output('fit-results', 'data'),
    Output('N0', 'value'),
    Output('tau', 'value'),
    Output('A', 'value'),
    Output('R', 'value'),
    Output('phi', 'value'),
    [Input('wiggle-plot', 'figure'),
     Input('residuals', 'figure'),
     Input('fft', 'figure'),
     Input('fit-results', 'data'),
     Input('hist-options', 'value'),
     Input('fit-start-time', 'value'),
     Input('fit-end-time', 'value'),
     Input('N0', 'value'),
     Input('tau', 'value'),
     Input('A', 'value'),
     Input('R', 'value'),
     Input('phi', 'value'),
     Input('do-fit-button', 'n_clicks')],
    [State('wiggle-plot', 'relayoutData')]
)
def plot_data(figure_input
    ,residuals_input
    ,fft_input
    ,fit_results
    ,hist_options
    ,fit_start_time
    ,fit_end_time
    ,N0
    ,tau
    ,A
    ,R
    ,phi
    ,do_fit_button
    ,figure_layout
    ):

    #Convert input figure to something useful
    figure = go.Figure(figure_input)
    if figure_layout:
        if ("xaxis.range[0]" in figure_layout) and ("xaxis.range[0]" in figure_layout):
            figure.update_layout(xaxis_range=[figure_layout["xaxis.range[0]"],figure_layout["xaxis.range[1]"]])
        if ("yaxis.range[0]" in figure_layout) and ("yaxis.range[0]" in figure_layout):
            figure.update_layout(yaxis_range=[figure_layout["yaxis.range[0]"],figure_layout["yaxis.range[1]"]])

    # #Residuals plot
    # residuals_fig = go.Figure(residuals_input)
    # if residuals_layout:
    #     if ("xaxis.range[0]" in residuals_layout) and ("xaxis.range[0]" in residuals_layout):
    #         residuals_fig.update_layout(xaxis_range=[residuals_layout["xaxis.range[0]"],residuals_layout["xaxis.range[1]"]])
    #     if ("yaxis.range[0]" in residuals_layout) and ("yaxis.range[0]" in residuals_layout):
    #         residuals_fig.update_layout(yaxis_range=[residuals_layout["yaxis.range[0]"],residuals_layout["yaxis.range[1]"]])

    # #FFT plot
    # fft_fig = go.Figure(fft_input)
    # if fft_layout:
    #     if ("xaxis.range[0]" in fft_layout) and ("xaxis.range[0]" in fft_layout):
    #         fft_fig.update_layout(xaxis_range=[fft_layout["xaxis.range[0]"],fft_layout["xaxis.range[1]"]])
    #     if ("yaxis.range[0]" in fft_layout) and ("yaxis.range[0]" in fft_layout):
    #         fft_fig.update_layout(yaxis_range=[fft_layout["yaxis.range[0]"],fft_layout["yaxis.range[1]"]])

    #Fit results
    fit_results_df = pd.DataFrame(fit_results)

    #Check if figure has "Data" trace
    has_wiggle = has_trace_with_name(figure,"Data")

    if not has_wiggle:
    
        # Build the Plotly figure    
        figure.add_trace(go.Scatter(x=df["time"].values,
                                            y=df["bin_content"].values,
                                            error_y=dict(type='data', array=df["bin_error"], visible=True),
                                            mode='lines',
                                            line_shape='hvh',
                                            name='Data',
                                            line=dict(color='blue')))
        
        figure.update_layout(
            title='Wiggle plot',
            xaxis=dict(
                title='time [us]'
            ),
            yaxis=dict(
                title='Counts / 149.2 ns'
            ),
            showlegend=False  # Show legend
        )


    # Check histogram options
    if 'log-y' in hist_options:
        # figure.update_layout(yaxis_range=[2,6.778])
        figure.update_yaxes(type='log')
    else:
        figure.update_yaxes(type='linear')
        # figure.update_layout(yaxis_range=[0,6e6])

    figure.update_layout(width=1000, height=400)

    # figure.update_layout(xaxis_range=[-10,700])
    figure.update_xaxes(domain=(0.05, 1.0))

    #Check fit options
    
    # Remove old lines at fit start and end times
    figure.layout.shapes = [shape for shape in figure.layout.shapes if shape['type'] != "line"]

    #Add new lines
    figure.add_vline(x=fit_start_time,line_color='red',line_dash='dash',name="FitStartTime")
    figure.add_vline(x=fit_end_time,line_color='red',line_dash='dash',name="FitEndTime")

    if ((ctx.triggered_id == 'fit-start-time') | (ctx.triggered_id == 'fit-end-time')):
        #Remove old fit, residuals, and fft
        new_data = []
        for trace in figure.data:
            if (trace.name != "Fit"):
                new_data.append(trace)
        figure = go.Figure(data=new_data, layout=figure.layout)

        residuals_fig = go.Figure()
        fft_fig = go.Figure()

        #reset fit results
        fit_results_df = pd.DataFrame()

    # Check if we are doing a fit

    
    residuals_fig = go.Figure()
    fft_fig = go.Figure()

    #Remove old fit
    new_data = []
    for trace in figure.data:
        if (trace.name != "Fit") :
            new_data.append(trace)
    figure = go.Figure(data=new_data, layout=figure.layout)
    
    df_to_fit = df[((df['time'] >= fit_start_time) & (df['time'] <= fit_end_time))]

    starting_parameters=[N0,tau,A,R,phi]
    params=[]
    covariance=[]
    if ctx.triggered_id == 'do-fit-button':
        params, covariance = curve_fit(five_parameter_fit, df_to_fit['time'].values, df_to_fit['bin_content'].values,sigma=df_to_fit['bin_error'].values, absolute_sigma=True,p0=starting_parameters)
        if params[2] < 0:
            starting_parameters[2]*=-1;
            starting_parameters[4]-=3.14;
        params, covariance = curve_fit(five_parameter_fit, df_to_fit['time'].values, df_to_fit['bin_content'].values,sigma=df_to_fit['bin_error'].values, absolute_sigma=True,p0=starting_parameters)

        N0=params[0]
        tau=params[1]
        A=params[2]
        R=params[3]
        phi=params[4]


    else:
        params=starting_parameters
        covariance=np.zeros((np.size(params),np.size(params)))

    #compute the fit values
    fit_values = five_parameter_fit(df_to_fit['time'].values,*params)

    #compute residuals and fft of residuals
    residuals = df_to_fit['bin_content'].values-fit_values
    fft_values = fft(residuals)
    n_samples = len(residuals)
    bin_width = df_to_fit['time'].values[1]-df_to_fit['time'].values[0]
    fft_freqs = fftfreq(n_samples, bin_width)[:n_samples//2]
    fft_amps = 2.0/n_samples * np.abs(fft_values[0:n_samples//2])

    #compute chi2, etc.
    chi2 = np.sum(pow(residuals/df_to_fit['bin_error'].values,2))
    ndf = len(residuals) - 5
    p_value = 1 - stats.chi2.cdf(chi2, ndf)

    fit_data = {
        'Parameter Name': ["chi2","ndf","p-value","N","tau","A","R","phi"],
        'Value':[chi2,ndf,p_value] + list(params),
        'Uncertainty': [0,0,0] + list(np.sqrt(np.diag(covariance)))
    }
    fit_results_df = pd.DataFrame(fit_data)

    figure.add_trace(go.Scatter(x=df_to_fit['time'].values,
                                y=fit_values,
                                mode='lines',
                                name='Fit',
                                line_color='red'))

    # Make the fit residuals
    residuals_fig.add_trace(go.Scatter(x=df_to_fit['time'].values,
                                   y=residuals,
                                   mode='lines',
                                   line_shape='hvh',
                                   name='Residuals',
                                   line=dict(color='blue')))
    
    # Make the residuals FFT
    fft_fig.add_trace(go.Scatter(x=fft_freqs,
                            y=fft_amps,
                            mode='lines',
                            line_shape='hvh',
                            name='FFT',
                            line=dict(color='blue')))
        

    residuals_fig.update_layout(width=1000, height=400)
    residuals_fig.update_xaxes(domain=(0.05, 1.0))
    residuals_fig.update_layout(
        title='Residuals = Data - Fit',
        xaxis=dict(
            title='time [us]'
        ),
        yaxis=dict(
            title='Residuals / 149.2 ns'
        ),
        showlegend=False  # Show legend
    )

    # residuals_fig.update_layout(xaxis_range=[-10,700])
    # residuals_fig.update_layout(yaxis_range=[-2e4,2e4])

    fft_fig.update_layout(width=1000, height=400)
    fft_fig.update_xaxes(domain=(0.05, 1.0))
    fft_fig.update_layout(
        title='Fast Fourier Transform (FFT) of the Residuals',
        xaxis=dict(
            title='Frequency [MHz]'
        ),
        yaxis=dict(
            title='FFT amplitude / 149.2 ns'
        ),
        showlegend=False  # Show legend
    )
    # fft_fig.update_layout(xaxis_range=[0,3.4])
    # fft_fig.update_layout(yaxis_range=[-100,1200])

    return figure, residuals_fig, fft_fig, fit_results_df.to_dict("records"), N0, tau, A, R, phi


# Define the model function to fit (e.g., linear function)
def five_parameter_fit(t, *p):
    return p[0]*np.exp(-t/p[1])*(1+p[2]*np.cos(2*np.pi*f_ref*(1+(p[3]+dR)*1e-6)*t+p[4]))

if __name__ == '__main__':
    app.run(debug=False, port=8002)
    
