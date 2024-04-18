import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import base64
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_table
import numpy as np
import dash_table
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from dash.exceptions import PreventUpdate
from pathlib import Path

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

# Styling for the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "transition": "margin-left 0.5s",
}
# Padding for the page content
CONTENT_STYLE = {
    "margin-right": "1rem",
    "padding": "1rem 1rem",
    "transition": "margin-left 0.5s",
}

custom_classes = {
    "sidebar": "bg-dark",  # Example: Apply a dark background to the sidebar
    "content": "bg-light",  # Example: Apply a light background to the content area
    "custom-div": "my-custom-class"  # Example: Apply a custom class to a specific div
}
# Define the button
button = html.Button('Menu', id='button',
                     n_clicks=0, style={'position': 'fixed',
                                        'top': '1px',
                                        "left": "20px",
                                        'transition': 'left 0.5s',
                                        'background-color': '#2c3e50',
                                        'color': 'white',
                                        'display': 'none',
                                        'border': 'none',
                                        'padding': '5px 15px',
                                        'text-align': 'center',
                                        'text-decoration': 'none',
                                        'display': 'inline-block',
                                        'font-size': '16px',
                                        'cursor': 'pointer',
                                        'border-radius': '10px',
                                       })


sidebar = html.Div(
    [
        html.H4("Sleep Apnea", className="display-7"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", id="home-link", active="exact", style={"color": "black"}),
                dbc.NavLink("Insights", href="/page-1",id="insights-link", active="exact", style={"color": "black"}),
                dbc.NavLink("Prediction", href="/page-2",id="prediction-link", active="exact", style={"color": "black"}),
                dbc.NavLink("Logout", href="/logout", id="logout-link", active="exact", style={"color": "black"}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
    className=custom_classes["sidebar"]
)

# Load the image from your filesystem
image_filename = 'Sleep Apnea.png'  # replace with your image file
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# Define the image component
image = html.Div(
    html.Img(src=f"data:image/jpeg;base64,{encoded_image.decode()}", style={"width": "100%", "height": "100%"}),
    style={"padding": "20px"}
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# Layout of login page
login_layout = html.Div(style={'textAlign': 'center'}, children=[
    html.Div(id='login-feedback'),
    html.Div(style={'background-color': '#7b8a8b', 'padding': '20px', 'border-radius': '1px',
                    'width': '250px', 'margin':'auto'}, children=[
        html.H2("Login"),
        dcc.Input(id='username-input', type='text', placeholder='Enter your username'),
        html.Br(),
        html.Br(),
        dcc.Input(id='password-input', type='password', placeholder='Enter your password'),
        html.Br(),
        html.Br(),
        html.Button('Login', id='login-button', n_clicks=0,
                    style={
                    "background-color": "#2c3e50",
                    "color": "white",
                    "border": "none",
                    "padding": "5px 15px",
                    "width": "150px", 
                    "text-decoration": 'none'# Fixed width for the button
                }),  # Login button
        html.Div(id='login-feedback', style={'align-self': 'flex-end', 'color': 'red'})
    ])
])


app.layout = html.Div([dcc.Location(id="url"), html.Div(id='auth-state', children=False, style={'display': 'none'}),sidebar, content, button,
                       html.Meta(name="viewport", content="width=device-width, initial-scale=1")])

@app.callback(
    [
        Output("home-link", "style"),
        Output("insights-link", "style"),
        Output("prediction-link", "style"),
        Output("logout-link", "style"), 
    ],
    [Input("url", "pathname")]
)
def update_link_styles(pathname):
    # Initialize styles for each link
    home_style = {"color": "black"}
    insights_style = {"color": "black"}
    prediction_style = {"color": "black"}
    logout_style = {"color": "black"}
    # Determine which link is active based on the current URL pathname
    if pathname == "/":
        home_style = {"color": "white"}  # Set "Home" link style to white
    elif pathname == "/page-1":
        insights_style = {"color": "white"}  # Set "Insights" link style to white
    elif pathname == "/page-2":
        prediction_style = {"color": "white"}  # Set "Prediction" link style to white
    elif pathname == "/logout":
        logout_style = {"color": "white"}  # Set "Logout" link style to white

    return home_style, insights_style, prediction_style, logout_style


@app.callback(
    [
        Output("sidebar", "style"),
        Output("button", "style"),
        Output("page-content", "style"),
    ],
    [
        Input("button", "n_clicks"),
        Input('auth-state', 'children')
    ],
    [
        State("sidebar", "style"),
        State("button", "style"),
        State("page-content", "style"),
    ],
)
def update_styles(n_clicks, authenticated, sidebar_style, button_style, content_style):
    # Toggle sidebar and button styles based on button click
    if n_clicks % 2 == 1:
        sidebar_style["margin-left"] = "0"
        button_style["left"] = "calc(15rem + 20px)"
        content_style["margin-left"] = "15rem"
    else:
        sidebar_style["margin-left"] = "-15rem"
        button_style["left"] = "20px"
        content_style["margin-left"] = "0"

    # Show or hide button based on authentication state
    button_style["display"] = "block" if authenticated else "none"

    return sidebar_style, button_style, content_style



@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"),
    Input('auth-state', 'children')]
)
def render_page_content(pathname, authenticated):
    if not authenticated:
        return login_layout
    elif pathname == "/":
            return [
                image
        ]
    elif pathname == "/page-1":
        return [
            html.H1('Sleep Apnea Statistics', style={'textAlign': 'center'}),
            dbc.Tabs(
                [
                    dbc.Tab(label="Demographics", tab_id="tab-1"),
                    dbc.Tab(label="Vital Signs", tab_id="tab-2"),
                    dbc.Tab(label="ESS Score", tab_id="tab-3"),
                ],
                id="tabs",
                active_tab="tab-1",
            ),
            html.Div(id="tabs-content"),
        ]
    elif pathname == "/page-2":
        return [
            html.H1('Sleep Apnea Prediction', style={'textAlign': 'center'}),
            html.Div(id='output-container-button',
                         children='Enter your details and press submit -'),
            html.Br(),
            html.Div([
                html.Div([
                html.Label('1.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Age :', style={'display': 'inline-block', 'margin-right': '10px'}),
                dcc.Input(id='input-age', type='number', placeholder='Age', min=0,
                          style={'width': '50px', 'margin-right': '10px'}),
                html.Div(style={'margin-top': '10px'}),  # Add space between elements
                html.Label('2.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Gender :', style={'margin-right': '10px'}),
                dcc.RadioItems(
                    id='input-gender',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                html.Br(),
                # html.Div(style={'margin-top': '10px'}),  # Add space between elements
                html.Label('3.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('BMI :',
                           style={'display': 'inline-block', 'margin-top': '10px', 'margin-right': '10px'}),
                dcc.Input(id='input-bmi', type='number', placeholder='BMI', min=0,
                          style={'display': 'inline-block', 'width': '51px', 'margin-right': '10px'}),
                html.Div(style={'margin-top': '10px'}),
                html.Label('4.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Neck Size(cms) :', style={'margin-right': '10px'}),
                dcc.Input(id='input-neck', type='number', placeholder='NS', min=0,
                          style={'width': '50px', 'margin-right': '10px'}),
                html.Div(style={'margin-top': '10px'}),
                html.Label('5.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('ESS Score :', style={'margin-right': '10px'}),
                dcc.Input(id='input-ess', type='number', placeholder='ESS', min=0,max=25,
                          style={'width': '50px', 'margin-right': '5px'}),
                    html.Br(),
                    html.Label('6.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Active smoker? ', style={'margin-right': '10px', 'margin-top': '5px'}),
                dcc.RadioItems(
                    id='input-smoker',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                    html.Br(),
        ], style={'width': '39%','border': '2px solid #2c3e50', 'padding': '5px','display': 'inline-block','float':'left','border-radius': '10px'}) ]),
                html.Div([
                    html.Div([
                html.Label('7.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Do you often feel TIRED, fatigued, or sleepy during daytime? ',
                           style={'margin-right': '10px'}),
                dcc.RadioItems(
                    id='input-tired',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                html.Br(),
                html.Label('8.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Has anyone OBSERVED you stop breathing during your sleep? ',
                           style={'margin-right': '10px', 'margin-top': '10px'}),
                dcc.RadioItems(
                    id='input-stop-breathing',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                html.Br(),
                html.Label('9.', style={'display': 'inline-block', 'width': '20px'}),
                html.Label('Do you have complaints of Memory loss? ',
                           style={'margin-right': '10px', 'margin-top': '10px'}),
                dcc.RadioItems(
                    id='input-memory-loss',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                html.Br(),
                html.Label('10. ', style={'display': 'inline-block', 'width': '30px'}),
                html.Label('Do you have morning headaches? ', style={'margin-right': '10px', 'margin-top': '10px'}),
                dcc.RadioItems(
                    id='input-morning-headaches',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                html.Br(),
                html.Label('11.', style={'display': 'inline-block', 'width': '30px'}),
                html.Label('Do you have Hypertension? ', style={'margin-right': '10px', 'margin-top': '10px'}),
                dcc.RadioItems(
                    id='input-hypertension',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                html.Br(),
                html.Label('12.', style={'display': 'inline-block', 'width': '30px'}),
                html.Label('Do you SNORE loudly? ', style={'margin-right': '10px', 'margin-top': '10px'}),
                dcc.RadioItems(
                    id='input-snore',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block'}
                ),
                    ],style={'width': '59%','border': '2px solid #2c3e50', 'padding': '12px','display': 'inline-block','float':'right','border-radius': '10px'})
                ]),
                html.Br(),
            html.Button(
                    'Submit',
                    id='submit-button',
                    n_clicks=0,
                    style={
                        'background-color': '#2c3e50',
                        'border': 'none',
                        'color': 'white',
                        'padding': '5px 15px',
                        'text-decoration': 'none',
                        'display': 'block',
                        #'flex-direction':'row',
                        'font-size': '16px',
                        'margin-top': '210px',
                        'margin-left': '1px',
                        #'cursor': 'pointer',
                        'border-radius': '30px',
                        #'float':'left'
                    }
                ),
                html.Br(),
                html.Div(id='data-table-container')
        ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# Callback to update the tab content
@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        # Read the Sleep Apnea dataset
        Sleep_Apnea = pd.read_excel("Sleep_Apnea_Data_Merged.xlsx")
        Sleep_Apnea.rename(columns=lambda x: x.replace('.', ' '), inplace=True)
        Sleep_Apnea['TotalAHI'] = Sleep_Apnea['TotalAHI'].apply(lambda x: 1 if x > 5 else 0)
        filtered_data = Sleep_Apnea[Sleep_Apnea['TotalAHI'] == 1]

        grouped = Sleep_Apnea.groupby('Gender').agg({'TotalAHI': ['count', lambda x: (x == 1).sum()]})
        # Renaming columns

        grouped.columns = ['Count', 'Y=1 Count']

        # Calculating percentage
        grouped['Percentage Y=1'] = round((grouped['Y=1 Count'] / grouped['Count']) * 100, 2)

        # Define colors for TotalAHI categories
        colors = {1: '#13967d'}

        # Create traces for each TotalAHI category
        traces = []
        for total_ahi, color in colors.items():
            trace = go.Bar(
                x=grouped.index,
                y=grouped['Percentage Y=1'],  # Using correct column for percentages
                name='TotalAHI ' + str(total_ahi),
                marker=dict(color=color),
            )
            traces.append(trace)

            # Create layout for the chart
            layout = go.Layout(
                barmode='group',
                xaxis=dict(title='Gender', showgrid=False),
                yaxis=dict(title='Presence of sleep apnea(%)', showgrid=False),
                height=350,
                margin=dict(l=10, r=50, t=20, b=20))
            fig_gender = go.Figure(data=traces, layout=layout)

        # Define bins and labels for age categories
        bins1 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
        labels1 = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90']

        # Categorize ages into bins for filtered data
        Sleep_Apnea['age_categories'] = pd.cut(Sleep_Apnea['Age'], bins=bins1, labels=labels1, right=False)
        Age_grp = Sleep_Apnea.groupby('age_categories').agg({'TotalAHI': ['count', lambda x: (x == 1).sum()]})

        # Renaming columns
        Age_grp.columns = ['Count', 'Y=1 Count']

        # Calculating percentage
        Age_grp['Percentage Y=1'] = round((Age_grp['Y=1 Count'] / Age_grp['Count']) * 100, 2)

        # Define colors for TotalAHI categories
        colorscale = {1: '#13967d'}

        # Create traces for each TotalAHI category
        traces = []
        for total_ahi, color in colors.items():
            trace = go.Bar(
                x=Age_grp.index,
                y=Age_grp['Percentage Y=1'],  # Using correct column for percentages
                name='TotalAHI ' + str(total_ahi),
                marker=dict(color=color),
            )
            traces.append(trace)

            # Create layout for the chart
            layout = go.Layout(
                barmode='group',
                xaxis=dict(title='Age Group', showgrid=False),
                yaxis=dict(title='Presence of sleep apnea(%)', showgrid=False),
                height=350,
                margin=dict(l=10, r=50, t=20, b=20))

            # Create figure for age distribution histogram
            fig_age = go.Figure(data=traces, layout=layout)

        # heat map
        Age_Gender_grp = Sleep_Apnea.groupby(['Gender', 'age_categories']).agg(
            {'TotalAHI': ['count', lambda x: (x == 1).sum()]})

        # Renaming columns Gender
        Age_Gender_grp.columns = ['Count', 'Y=1 Count']

        # Calculating percentage
        Age_Gender_grp['Percentage Y=1'] = round((Age_Gender_grp['Y=1 Count'] / Age_Gender_grp['Count']) * 100, 2)

        heatmap_data = Age_Gender_grp['Percentage Y=1'].unstack()

        # Replace NaN values with 0
        heatmap_data = heatmap_data.fillna(0)

        # Create heatmap
        fig_age_gender = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='blues'))

        # Add labels
        fig_age_gender.update_layout(
            xaxis=dict(title='Age Group', showgrid=False),
            yaxis=dict(title='Gender', showgrid=False),
            coloraxis_colorbar=dict(title='Percentage'),
            barmode='stack',
            height=350,  # Set the height of the chart
            margin=dict(l=10, r=20, t=20, b=20),
        )
        return [
            html.H2('', style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apne(%) by Gender",
                                    id="collapse-button-gender",
                                    color="primary",
                                    className="mb-0"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="genderplot-spinner",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="genderplot",
                                            figure=fig_gender
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-gender",
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width=6),  # Adjust the column width
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apne(%) by Age Group",
                                    id="collapse-button-age",
                                    color="primary",
                                    className="mb-0"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="ageplot-spinner",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="ageplot",
                                            figure=fig_age
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-age",
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width=6)  # Adjust the column width
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apne(%) by Gender and Age",
                                    id="collapse-button-age-gender",
                                    color="primary",
                                    className="mb-0"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="age-gender-plot-spinner",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="age-heatmap",
                                            figure=fig_age_gender
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-age-gender",
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width={'size': 8, 'offset': 3}, )  # Adjust the column width
            ]),
        ]


    elif active_tab == "tab-2":
        # Read the Sleep Apnea dataset
        Sleep_Apnea = pd.read_excel("Sleep_Apnea_Data_Merged.xlsx")

        Sleep_Apnea['TotalAHI'] = Sleep_Apnea['TotalAHI'].apply(lambda x: 1 if x > 5 else 0)
        Sleep_Apnea.rename(columns=lambda x: x.replace('.', ' '), inplace=True)
        filtered_data = Sleep_Apnea[Sleep_Apnea['TotalAHI'] == 1]
        Sleep_Apnea = Sleep_Apnea.assign(
            BMICategories=pd.cut(Sleep_Apnea['BMI'], bins=[0, 30, float('inf')], right=False,
                                 labels=["Others", "Obesity"]))
        # Group by BMI Categories and TotalAHI, then count occurrences
        BMI_grp = Sleep_Apnea.groupby('BMICategories').agg({'TotalAHI': ['count', lambda x: (x == 1).sum()]})
        # Renaming columns
        BMI_grp.columns = ['Count', 'Y=1 Count']
        # Calculating percentage
        BMI_grp['Percentage Y=1'] = round((BMI_grp['Y=1 Count'] / BMI_grp['Count']) * 100, 2)
        # Define colors for TotalAHI categories
        colors = {1: '#13967d'}

        # Create traces for each TotalAHI category
        traces = []
        for category, color in colors.items():
            trace = go.Bar(
                x=BMI_grp.index,  # Using index for BMI categories
                y=BMI_grp['Percentage Y=1'],  # Using 'Percentage Y=1' column for percentages
                name='TotalAHI ' + str(category),
                marker=dict(color=color),
            )
            traces.append(trace)

            # Create layout for the chart
            layout = go.Layout(
                barmode='group',
                xaxis=dict(title='BMI Categories', showgrid=False),
                yaxis=dict(title='Presence of sleep apnea(%)', showgrid=False),
                height=350,  # Set the height of the chart
                margin=dict(l=10, r=10, t=10, b=20),
            )

        # Create figure
        fig_bmi = go.Figure(data=traces, layout=layout)

        HTN_grp = Sleep_Apnea.groupby('Hypertension').agg({'TotalAHI': ['count', lambda x: (x == 1).sum()]})
        # Renaming columns
        HTN_grp.columns = ['Count', 'Y=1 Count']

        # Calculating percentage
        HTN_grp['Percentage Y=1'] = round((HTN_grp['Y=1 Count'] / HTN_grp['Count']) * 100, 2)

        # Define colors for TotalAHI categories
        colors = {1: '#1f77b4'}

        traces = []
        for total_ahi, color in colors.items():
            trace = go.Bar(
                y=HTN_grp.index,  # Use index as x-axis
                x=HTN_grp['Percentage Y=1'],  # Use correct column for percentages
                name='TotalAHI ' + str(total_ahi),
                marker=dict(color=color),
                orientation='h'
            )
            traces.append(trace)

            # Create layout for the chart
            layout = go.Layout(
                barmode='group',
                xaxis=dict(title='Presence of sleep apnea(%)', showgrid=False),
                yaxis=dict(title='Hypertension Categories', showgrid=False),
                height=350,  # Set the height of the chart
                margin=dict(l=10, r=10, t=10, b=20),
            )
        fig_htn = go.Figure(data=traces, layout=layout)

        # pox plot for neck size
        Sleep_Apnea['NeckSize'] = np.where(Sleep_Apnea['Neck Size(cms)'] <= 37, '<= 37',
                                           np.where((Sleep_Apnea['Neck Size(cms)'] > 37) & (
                                                       Sleep_Apnea['Neck Size(cms)'] <= 42), '>37 to <=42',
                                                    '> 42'))
        NS_grp = Sleep_Apnea.groupby('NeckSize').agg({'TotalAHI': ['count', lambda x: (x == 1).sum()]})
        # Renaming columns
        NS_grp.columns = ['Count', 'Y=1 Count']

        # Calculating percentage
        NS_grp['Percentage Y=1'] = round((NS_grp['Y=1 Count'] / NS_grp['Count']) * 100, 2)
        NS_grp = NS_grp.reindex(['<= 37', '>37 to <=42', '> 42'])
        # Define colors for TotalAHI categories
        colors = {1: '#13967d'}
        # Create traces for each TotalAHI category
        traces = []
        for total_ahi, color in colors.items():
            trace = go.Bar(
                x=NS_grp.index,
                y=NS_grp['Percentage Y=1'],  # Use 'Percentage Y=1' column for percentages
                name='TotalAHI ' + str(total_ahi),
                marker=dict(color=color),
            )
            traces.append(trace)
            # Create layout for the chart
            layout = go.Layout(
                barmode='group',
                xaxis=dict(title='Neck Size Categories', showgrid=False),
                yaxis=dict(title='Presence of sleep apnea(%)', showgrid=False),
                height=350,  # Set the height of the chart
                margin=dict(l=10, r=10, t=10, b=10),
            )
        fig_neck_size = go.Figure(data=traces, layout=layout)

        # Percentage of ActiveSmoker with TotalAHI
        # Group by Active smoker Categories and TotalAHI, then count occurrences
        AS_grp = Sleep_Apnea.groupby('Active smoker').agg({'TotalAHI': ['count', lambda x: (x == 1).sum()]})
        # Renaming columns
        AS_grp.columns = ['Count', 'Y=1 Count']

        # Calculating percentage
        AS_grp['Percentage Y=1'] = round((AS_grp['Y=1 Count'] / AS_grp['Count']) * 100, 2)
        # Define colors for TotalAHI categories
        colors = {1: '#1f77b4'}

        # Create traces for each TotalAHI category
        traces = []
        for total_ahi, color in colors.items():
            trace = go.Bar(
                y=AS_grp.index,  # Use index as x-axis
                x=AS_grp['Percentage Y=1'],  # Use correct column for percentages
                name='TotalAHI ' + str(total_ahi),
                marker=dict(color=color),
                orientation='h'
            )
            traces.append(trace)

            # Create layout for the chart
            layout = go.Layout(
                barmode='group',
                xaxis=dict(title='Presence of sleep apnea(%)', showgrid=False),
                yaxis=dict(title='Active Smoking Categories', showgrid=False),
                height=350,  # Set the height of the chart
                margin=dict(l=10, r=10, t=10, b=20),
            )
        fig = go.Figure(data=traces, layout=layout)
        return [
            html.H2('', style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apnea(%) by BMI",
                                    id="collapse-button-bmi-2",  # Update the id here
                                    color="primary",
                                    className="mb-0"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="bmiplot-spinner",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="bmiplot",
                                            figure=fig_bmi
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-bmi",
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width=6),  # Adjust the column width
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apnea(%) by Hypertension",
                                    id="collapse-button-htn",
                                    color="primary",
                                    className="mb-0"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="htnplot-spinner",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="htnplot",
                                            figure=fig_htn
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-htn",
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width=6)]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apnea(%) by Neck Size",
                                    id="collapse-button-neck-size",
                                    color="primary",
                                    className="mb-3"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="neck-size-plot-spinner",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="neck-size-plot",
                                            figure=fig_neck_size
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-neck-size",
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Sleep Apnea(%) by Active Smoker",  # Update button label
                                    id="collapse-button-as",  # Update button id
                                    color="primary",
                                    className="mb-3"
                                )
                            )
                        ),
                        dbc.Collapse(
                            dbc.CardBody(
                                dcc.Loading(
                                    id="asplot-spinner",  # Update loading id
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="asplot",  # Update graph id
                                            figure=fig  # Use the fig for Active Smoker Distribution
                                        )
                                    ]
                                )
                            ),
                            id="collapse-body-as",  # Update collapse id
                            is_open=True  # Set to False to initially collapse the card
                        ),
                    ], className="mb-3"),
                ], width=6)
            ]),
        ]
    elif active_tab == "tab-3":
        Sleep_Apnea = pd.read_excel("Sleep_Apnea_Data_Merged.xlsx")
        Sleep_Apnea.rename(columns=lambda x: x.replace('.', ' '), inplace=True)
        Sleep_Apnea['TotalAHI'] = Sleep_Apnea['TotalAHI'].apply(lambda x: 'Yes' if x > 5 else 'No')
        # Filter data where TotalAHI equals 1
        y = Sleep_Apnea[Sleep_Apnea['TotalAHI'] == 1]
        fig = px.box(Sleep_Apnea, x='TotalAHI', y='Total ScoreESS')
        fig.update_layout(
            xaxis=dict(title='Presence of sleep apnea', showgrid=False),
            yaxis=dict(title='ESS Score', showgrid=False),
            height=350,  # Set the height of the chart
            margin=dict(l=10, r=20, t=10, b=10),
        )

    return [
        html.H2('', style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H2(
                            dbc.Button(
                                "ESS Score for Sleep Apnea",
                                id="collapse-button-osa-2",  # Update the id here
                                color="primary",
                                className="mb-0"
                            )
                        )
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            dcc.Loading(
                                id="essplot-spinner",
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="essplot",
                                        figure=fig  # Update this with your figure
                                    )
                                ]
                            )
                        ),
                        id="collapse-body-osa",
                        is_open=True  # Set to False to initially collapse the card
                    ),
                ], className="mb-3"),
            ], width={'size': 8, 'offset': 2}, )
        ])]


html.Br(),

dbc.Button(
    "Toggle Collapse",
    id="toggle-collapse",
    color="primary",
    className="mb-3"
),
html.Br()


# Toggle collapse
@app.callback(
    [Output("collapse-body-gender", "is_open"),
     Output("collapse-body-age", "is_open"),
     Output("collapse-body-age-gender", "is_open")],
    [Input("collapse-button-gender", "n_clicks"),
     Input("collapse-button-age", "n_clicks"),
     Input("collapse-button-age-gender", "n_clicks")],
    [State("collapse-body-gender", "is_open"),
     State("collapse-body-age", "is_open"),
     State("collapse-body-age-gender", "is_open")],
)
def toggle_collapse(n_gender, n_age, n_age_gender, is_open_gender, is_open_age, is_open_age_gender):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, True, True
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'collapse-button-gender':
            return not is_open_gender, is_open_age, is_open_age_gender
        elif button_id == 'collapse-button-age':
            return is_open_gender, not is_open_age, is_open_age_gender
        elif button_id == 'collapse-button-age-gender':
            return is_open_gender, is_open_age, not is_open_age_gender
        else:
            return True, True, True


@app.callback(
    [Output("collapse-body-bmi", "is_open"),
     Output("collapse-body-htn", "is_open"),
     Output("collapse-body-neck-size", "is_open"),
     Output("collapse-body-as", "is_open")],  # Add output for neck size card
    [Input("collapse-button-bmi-2", "n_clicks"),
     Input("collapse-button-htn", "n_clicks"),
     Input("collapse-button-neck-size", "n_clicks"),
     Input("collapse-button-as", "n_clicks")],  # Add input for neck size button
    [State("collapse-body-bmi", "is_open"),
     State("collapse-body-htn", "is_open"),
     State("collapse-body-neck-size", "is_open"),
     State("collapse-body-as", "is_open")]
)
def toggle_collapse(n_bmi, n_htn, n_neck_size, n_as, is_open_bmi, is_open_htn, is_open_neck_size, is_open_as):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open_bmi, is_open_htn, is_open_neck_size, is_open_as
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'collapse-button-bmi-2':
            return not is_open_bmi, is_open_htn, is_open_neck_size, is_open_as
        elif button_id == 'collapse-button-htn':
            return is_open_bmi, not is_open_htn, is_open_neck_size, is_open_as
        elif button_id == 'collapse-button-neck-size':
            return is_open_bmi, is_open_htn, not is_open_neck_size, is_open_as
        elif button_id == 'collapse-button-as':
            return is_open_bmi, is_open_htn, is_open_neck_size, not is_open_as
        else:
            return is_open_bmi, is_open_htn, is_open_neck_size, is_open_as


# Toggle collapse for ESS Score Distribution card
@app.callback(
    Output("collapse-body-osa", "is_open"),
    [Input("collapse-button-osa-2", "n_clicks")],
    [State("collapse-body-osa", "is_open")]
)
def toggle_collapse_ess_score(n_clicks, is_open):
    if n_clicks is not None:
        return not is_open  # Toggle the state only when button is clicked
    return True


@app.callback(
    Output('data-table-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-age', 'value'),
     State('input-gender', 'value'),
     State('input-bmi', 'value'),
     State('input-neck', 'value'),
     State('input-ess', 'value'),
     State('input-tired', 'value'),
     State('input-stop-breathing', 'value'),
     State('input-memory-loss', 'value'),
     State('input-morning-headaches', 'value'),
     State('input-hypertension', 'value'),
     State('input-smoker', 'value'),
     State('input-snore', 'value')])
def update_table(n_clicks, Age, Gender, WeightStatus, Neck_Size_cms, Total_ScoreESS,
                 Do_you_often_feel_TIRED_fatigued_or_sleepy_during_daytime,
                 Has_anyone_OBSERVED_you_stop_breathing_during_your_sleep, Do_you_have_complaints_of_Memory_loss,
                 Do_you_have_morning_headaches,
                 Hypertension, Active_smoker, Do_you_SNORE_loudly):
    if n_clicks > 0:
        if None in (Age, Gender, WeightStatus, Neck_Size_cms, Total_ScoreESS,
                    Do_you_often_feel_TIRED_fatigued_or_sleepy_during_daytime,
                    Has_anyone_OBSERVED_you_stop_breathing_during_your_sleep, Do_you_have_complaints_of_Memory_loss,
                    Do_you_have_morning_headaches, Hypertension, Active_smoker, Do_you_SNORE_loudly):
            return 'Please fill out all fields.'
        elif not all(isinstance(v, (int, float)) for v in (Age, WeightStatus, Neck_Size_cms, Total_ScoreESS)):
            return 'Age, BMI, Neck size, and ESS Score must be numerical values.'
        else:
            # Create a pandas DataFrame with the entered data
            data = {
                'Age': [Age],
                'Gender': [Gender],
                'BMI': [WeightStatus],
                'Neck Size': [Neck_Size_cms],
                'ESS Score': [Total_ScoreESS],
                'Tiredness': [Do_you_often_feel_TIRED_fatigued_or_sleepy_during_daytime],
                'Stop Breathing': [Has_anyone_OBSERVED_you_stop_breathing_during_your_sleep],
                'Memory Loss': [Do_you_have_complaints_of_Memory_loss],
                'Morning Headaches': [Do_you_have_morning_headaches],
                'Hypertension': [Hypertension],
                'Active Smoker': [Active_smoker],
                'Loud Snoring': [Do_you_SNORE_loudly]
            }
            df = pd.DataFrame(data)
            df2 = pd.DataFrame(data)
            df2.rename(columns={'Neck Size': 'Neck_Size(cms)'}, inplace=True)
            df2.rename(columns={'ESS Score': 'Total_ScoreESS'}, inplace=True)
            df2.rename(columns={'Tiredness': 'Do_you_often_feel_TIRED_fatigued_or_sleepy_during_daytime'}, inplace=True)
            df2.rename(columns={'Stop Breathing': 'Has_anyone_OBSERVED_you_stop_breathing_during_your_sleep'},
                       inplace=True)
            df2.rename(columns={'Memory Loss': 'Do_you_have_complaints_of_Memory_loss'}, inplace=True)
            df2.rename(columns={'Morning Headaches': 'Do_you_have_morning_headaches'}, inplace=True)
            df2.rename(columns={'Hypertension': 'Hypertension'}, inplace=True)
            df2.rename(columns={'Active Smoker': 'Active_smoker'}, inplace=True)
            df2.rename(columns={'Loud Snoring': 'Do_you_SNORE_loudly'}, inplace=True)
            # print(df2.info())
            df2.to_csv('test_data.csv', index=False)
            current_directory = Path.cwd()
            print("Current Directory:", current_directory)

            # Read the Excel file
            df1 = pd.read_excel("Sleep_Apnea_Data_Merged.xlsx")
            df1.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
            df1.rename(columns=lambda x: x.replace(',', ''), inplace=True)
            df1.rename(columns=lambda x: x.replace('?', ''), inplace=True)
            df1.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

            # Create binary target variable based on cutoff
            df1['Y'] = df1['TotalAHI'].apply(lambda x: 1 if x > 5 else 0)
            df1.head()
            # Create categorical variables for WeightStatus and NeckSizeCategories
            df1['WeightStatus'] = pd.cut(df1['BMI'], bins=[-float('inf'), 30, float('inf')],
                                         labels=['Others', 'Obesity']).astype('category')
            df1['WeightStatus'] = df1['WeightStatus'].cat.reorder_categories(['Obesity', 'Others'], ordered=True)
            df1['NeckSizeCategories'] = pd.cut(df1['Neck_Size(cms)'], bins=[-float('inf'), 37, 42, float('inf')],
                                               labels=['<=37', '>37 to <=42', '>42']).astype('category')

            # Define the formula string for the model
            formula = 'Y ~ Age + Gender + NeckSizeCategories + Total_ScoreESS + Do_you_often_feel_TIRED_fatigued_or_sleepy_during_daytime + Has_anyone_OBSERVED_you_stop_breathing_during_your_sleep + Do_you_have_complaints_of_Memory_loss + Do_you_have_morning_headaches + Hypertension + Active_smoker + WeightStatus + Do_you_SNORE_loudly'

            # Fit the logistic regression model
            Binary_model = sm.Logit.from_formula(formula, data=df1).fit()

            predicted_probabilities = Binary_model.predict()
            # print(predicted_probabilities)

            fpr, tpr, thresholds = roc_curve(df1['Y'], predicted_probabilities)
            roc_auc_train = auc(fpr, tpr)
            # print("ROC-AUC on entered data:", roc_auc_train)
            # Create categorical variables for WeightStatus and NeckSizeCategories
            df2['WeightStatus'] = pd.cut(df2['BMI'], bins=[-float('inf'), 30, float('inf')],
                                         labels=['Others', 'Obesity']).astype('category')
            df2['WeightStatus'] = df2['WeightStatus'].cat.reorder_categories(['Obesity', 'Others'], ordered=True)
            bins = [-float('inf'), 37, 42, float('inf')]
            labels = ['<=37', '>37 to <=42', '>42']
            df2['NeckSizeCategories'] = pd.cut(df2['Neck_Size(cms)'], bins=bins, labels=labels).astype('category')

            predicted_probabilities_test = Binary_model.predict(df2)
            # print("Predicted Probabilities on Test Data:")
            # print(predicted_probabilities_test)
            threshold = 0.92
            predicted_classes_test = (predicted_probabilities_test > threshold).astype(int)

            # Display the DataFrame in a dash_table.DataTable
            table = dash_table.DataTable(
                id='data-table',
                columns=[
                    {'name': col, 'id': col} for col in df.columns
                ],
                data=df.to_dict('records'),
                style_table={'overflowX': 'auto', 'width': '90%', 'margin': 'auto', 'margin-bottom': '20px',
                             'margin-left': '15px'}  # Optional: Styling the table
            )
            # Create a div to display the predicted probabilities_test
            if predicted_probabilities_test.values[0] < 0.50:
                prediction_result = f'Patient does not have sleep apnea (Estimated Prob={predicted_probabilities_test.values[0]:.2f}).'
                prediction_color = 'green'
            elif predicted_probabilities_test.values[0] >= 0.50 and predicted_probabilities_test.values[0] < 0.92:
                prediction_result = f'Patient likely has sleep apnea (Estimated Prob={predicted_probabilities_test.values[0]:.2f}).'
                prediction_color = 'orange'
            else:
                prediction_result = f'Patient has sleep apnea (Estimated Prob={predicted_probabilities_test.values[0]:.2f}).'
                prediction_color = 'red'

                # Create separate components for table and prediction message
            table_component = html.Div([html.H4('Data Table:'), table])
            prediction_component = html.Div([html.H4('Prediction Result:'), html.P(html.Strong(prediction_result), style={'color': prediction_color}
                                                                                  )])

            return [table_component, prediction_component]
    else:
        return None

@app.callback(
    Output('auth-state', 'children'),
    [Input('login-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value')],
    prevent_initial_call=True
)
def authenticate_user(n_clicks, username, password):
    if not n_clicks:
        raise PreventUpdate  # No action performed yet

    if username == 'u' and password == 'p':
        # Set authentication state to True
        return True
    else:
        # Set authentication state to False
        return False

@app.callback(
    Output('url', 'pathname'),  # Redirect to login layout
    [Input('logout-link', 'n_clicks')],
    prevent_initial_call=True
)
def logout_user(n_clicks):
    if n_clicks:
        return "/" 
@app.callback(
    Output('login-feedback', 'children'),
    [Input('auth-state', 'children')],
    prevent_initial_call=True
)
def display_login_feedback(authenticated):
    if not authenticated:
        return "Invalid username or password"
    
if __name__ == '__main__':
    app.run_server(debug=True, port=3051)
