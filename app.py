# Import libraries
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import numpy as np
import base64
from olympics_predictions import predict_athlete_performance, predict_top_countries
import pickle

# Load Olympic Games dataset
df = pd.read_csv("csv_data/olympics.csv")

# Load athlete performance prediction model
athlete_model_path = 'athlete_model.pkl'
with open(athlete_model_path, 'rb') as f:
    athlete_model = pickle.load(f)

# Load top countries prediction model
top_countries_model_path = 'top_countries_model.pkl'
with open(top_countries_model_path, 'rb') as f:
    top_countries_model = pickle.load(f)

# Define app
app = dash.Dash(external_stylesheets=[dbc.themes.SIMPLEX], suppress_callback_exceptions=True)

# Define app layout
app.layout = html.Div([
    # Header
    html.Div(
        [
            html.Img(src="assets/olympics.png", height="60px", style={"margin-left": "70px"}),
            html.H1("Olympics Dashboard", className="ml-2 align-self-center", style={"font-size": "20px", "text-decoration": "none", "color": "#2D3C6B", "text-decoration": "none", "font-weight": "bold"})
        ],
        className="row",
        style={"height": "80px", "width": "100%", "background-color": "#f8f9fa", "padding": "10px 20px"}
    ),

    # Navigation bar
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink(html.Img(src="assets/home.png", height="40px"), href="/", style={"padding-left": "20px"})),
            dbc.NavItem(dbc.NavLink(html.Img(src="assets/pred.png", height="40px"), href="/pred", style={"padding-left": "20px"})),
        ],
        className="mr-auto",
        navbar=True,
        style={"padding-left": "950px", "margin-top": "10px"}
    ),

    # Body
    html.Div(
        [
            # Athlete performance prediction section
            html.Div(
                [
                    html.H2("Athlete Performance Prediction", className="mb-4"),
                    html.P("Enter athlete information to predict their performance."),

                    # Athlete information form
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label("Name", html_for="athlete-name"),
                                    dbc.Input(id="athlete-name", placeholder="Enter athlete name"),
                                ],
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Age", html_for="athlete-age"),
                                    dbc.Input(id="athlete-age", placeholder="Enter athlete age", type="number"),
                                ],
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Height (cm)", html_for="athlete-height"),
                                    dbc.Input(id="athlete-height", placeholder="Enter athlete height", type="number"),
                                ],
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Weight (kg)", html_for="athlete-weight"),
                                    dbc.Input(id="athlete-weight", placeholder="Enter athlete weight", type="number"),
                                ],
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Sport", html_for="athlete-sport"),
                                    dbc.Select(
                                        id="athlete-sport",
                                        options=[{'label': sport, 'value': sport} for sport in df['Sport'].unique()],
                                        placeholder="Select athlete sport",
                                    ),
                                ],
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Sex", html_for="athlete-sex"),
                                    dbc.RadioItems(
                                        id="athlete-sex",
                                        options=[{'label': 'Male', 'value': 'M'}, {'label': 'Female', 'value': 'F'}],
                                        value='M',
                                    ),
                                ],
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Year", html_for="athlete-year"),
                                    dbc.Input(id="athlete-year", placeholder="Enter year", type="number"),
                                ],
                            ),
                            dbc.Button("Predict", id="predict-athlete-button", color="primary"),
                        ],
                    ),

                    # Prediction results
                    html.Div(id="athlete-prediction-results", style={"margin-top": "20px"}),
                ],
                className="col-md-6",
            ),

            # Top countries prediction section
            html.Div(
                [
                    html.H2("Top Countries Prediction", className="mb-4"),
                    html.P("Predict the top countries in terms of medal count for a given year."),

                    # Year selection dropdown
                    dcc.Dropdown(
                        id="top-countries-year-dropdown",
                        options=[{'label': str(year), 'value': year} for year in df['Year'].unique()],
                        value=2024,
                        clearable=False,
                    ),

                    # Prediction results
                    html.Div(id="top-countries-prediction-results", style={"margin-top": "20px"}),
                ],
                className="col-md-6",
            ),
        ],
        className="row",
        style={"margin-top": "20px"}
    ),
])

# Define callbacks
@app.callback(
    [Output("athlete-prediction-results", "children"),
     Output("top-countries-prediction-results", "children")],
    [Input("predict-athlete-button", "n_clicks"),
     Input("top-countries-year-dropdown", "value")]
)
def update_predictions(n_clicks, year):
    """
    Updates the prediction results based on user input.
    """

    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] == 'predict-athlete-button.n_clicks':
        # Athlete performance prediction
        athlete_name = dash.callback_context.inputs['athlete-name']
        athlete_age = dash.callback_context.inputs['athlete-age']
        athlete_height = dash.callback_context.inputs['athlete-height']
        athlete_weight = dash.callback_context.inputs['athlete-weight']
        athlete_sport = dash.callback_context.inputs['athlete-sport']
        athlete_sex = dash.callback_context.inputs['athlete-sex']
        athlete_year = dash.callback_context.inputs['athlete-year']

        # Predict athlete performance
        athlete_data = {
            'Name': athlete_name,
            'Age': int(athlete_age),
            'Height': int(athlete_height),
            'Weight': int(athlete_weight),
            'Sport': athlete_sport,
            'Sex': athlete_sex,
            'Year': int(athlete_year),
        }
        predictions = predict_athlete_performance(athlete_data, athlete_model_path)

        # Display prediction results
        results = f"""
        ## Athlete Performance Prediction Results

        **Name:** {athlete_name}
        **Predicted Medal:** {predictions['Predicted Medal']}
        **Predicted Rank:** {predictions['Predicted Rank']}
        """
        return results, None

    elif ctx.triggered[0]['prop_id'] == 'top-countries-year-dropdown.value':
        # Top countries prediction
        top_countries = predict_top_countries(df, year, top_countries_model_path)

        # Display prediction results
        results = f"""
        ## Top Countries Prediction Results for {year}

        * {top_countries[0]}
        * {top_countries[1]}
        * {top_countries[2]}
        * {top_countries[3]}
        * {top_countries[4]}
        * {top_countries[5]}
        * {top_countries[6]}
        * {top_countries[7]}
        * {top_countries[8]}
        * {top_countries[9]}
        """
        return None, results

    else:
        return None, None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
