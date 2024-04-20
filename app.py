import base64
import pickle
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from keras.models import load_model


# Load data and models
with open('one_hot_encoder_sec.pkl', 'rb') as f:
    one_hot_encoder_sec_func = pickle.load(f)

with open('scaler_sec.pkl', 'rb') as f:
    scaler_sec_func = pickle.load(f)

with open('ordinal_sec.pkl', 'rb') as f:
    ordinal_sec_func = pickle.load(f)

with open('one_hot_encoder_main.pkl', 'rb') as f:
    one_hot_encoder_main_func = pickle.load(f)

with open('scaler_main.pkl', 'rb') as f:
    scaler_main_func = pickle.load(f)

with open('features_main.pkl', 'rb') as f:
    features_main_func = pickle.load(f)

# Load Keras models
model_sec = load_model('pred_sport_from_type.keras')
model_main = load_model('pred_sport_from_all1.keras')

# Load Sports Category Data
gender_sort = pd.read_csv("Sports_Cat.csv")
# Load necessary files and models
with open('one_hot_encoder_sec.pkl', 'rb') as f:
    one_hot_encoder_sec_func = pickle.load(f)

with open('scaler_sec.pkl', 'rb') as f:
    scaler_sec_func = pickle.load(f)

with open('ordinal_sec.pkl', 'rb') as f:
    ordinal_sec_func = pickle.load(f)

df = pd.read_csv("olympics_cleaned.csv")

model_sec = load_model('pred_sport_from_type.keras')

# Define options for dropdown menus
age_options = [{'label': str(i), 'value': i} for i in range(15, 61)]
weight_options = [{'label': str(i), 'value': i} for i in range(25, 201)]
height_options = [{'label': str(i), 'value': i} for i in range(150, 251)]
gender_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}]
country_options = [{'label': country, 'value': country} for country in [
    "China", "Denmark", "Netherlands", "USA", "Finland", "Norway", "Romania", "Estonia", "France", "Morocco",
    "Spain", "Egypt", "Iran", "Bulgaria", "Italy", "Azerbaijan", "Sudan", "Russia", "Argentina", "Cuba", "Belarus",
    "Greece", "Cameroon", "Turkey", "Chile", "Mexico", "Nicaragua", "Hungary", "Nigeria", "Chad", "Algeria", "Kuwait",
    "Bahrain", "Pakistan", "Iraq", "Syria", "Lebanon", "Qatar", "Malaysia", "Germany", "Canada", "Ireland", "Australia",
    "South Africa", "Eritrea", "Tanzania", "Jordan", "Tunisia", "Libya", "Belgium", "Djibouti", "Palestine", "Comoros",
    "Kazakhstan", "Brunei", "India", "Saudi Arabia", "Maldives", "Ethiopia", "United Arab Emirates", "Yemen", "Indonesia",
    "Philippines", "Uzbekistan", "Kyrgyzstan", "Tajikistan", "Japan", "Switzerland", "Brazil", "Monaco", "Israel", "Sweden",
    "Virgin Islands, US", "Sri Lanka", "Armenia", "Ivory Coast", "Kenya", "Benin", "Ukraine", "UK", "Ghana", "Somalia",
    "Latvia", "Niger", "Mali", "Poland", "Costa Rica", "Panama", "Georgia", "Slovenia", "Croatia", "Guyana", "New Zealand",
    "Portugal", "Paraguay", "Angola", "Venezuela", "Colombia", "Bangladesh", "Peru", "Uruguay", "Puerto Rico", "Uganda",
    "Honduras", "Ecuador", "El Salvador", "Turkmenistan", "Mauritius", "Seychelles", "Czech Republic", "Luxembourg",
    "Mauritania", "Saint Kitts", "Trinidad", "Dominican Republic", "Saint Vincent", "Jamaica", "Liberia", "Suriname",
    "Nepal", "Mongolia", "Austria", "Palau", "Lithuania", "Togo", "Namibia", "Curacao", "Iceland", "American Samoa",
    "Samoa", "Rwanda", "Dominica", "Haiti", "Malta", "Cyprus", "Guinea", "Belize", "South Korea", "Bermuda", "Serbia",
    "Sierra Leone", "Papua New Guinea", "Afghanistan", "Individual Olympic Athletes", "Oman", "Fiji", "Vanuatu", "Moldova",
    "Bahamas", "Guatemala", "Virgin Islands, British", "Mozambique", "Central African Republic", "Madagascar",
    "Bosnia and Herzegovina", "Guam", "Cayman Islands", "Slovakia", "Barbados", "Guinea-Bissau", "Thailand", "Timor-Leste",
    "Democratic Republic of the Congo", "Gabon", "San Marino", "Laos", "Botswana", "North Korea", "Senegal", "Cape Verde",
    "Equatorial Guinea", "Boliva", "Andorra", "Antigua", "Zimbabwe", "Grenada", "Saint Lucia", "Micronesia", "Myanmar",
    "Malawi", "Zambia", "Taiwan", "Sao Tome and Principe", "Republic of Congo", "Macedonia", "Tonga", "Liechtenstein",
    "Montenegro", "Gambia", "Solomon Islands", "Cook Islands", "Albania", "Swaziland", "Burkina Faso", "Burundi", "Aruba",
    "Nauru", "Vietnam", "Cambodia", "Bhutan", "Marshall Islands", "Kiribati", "Kosovo", "South Sudan", "Lesotho"
]]
sport_type_options = [{'label': sport_type, 'value': sport_type} for sport_type in [
    "TeamSports", "CombatSports", "WinterSports", "Athletics", "Aquatics", "RacquetSports", "WaterSports",
    "IndividualSports", "Weightlifting", "Equestrianism", "Shooting", "Cycling", "ModernPentathlon", "Archery", "Triathlon"
]]

def image_source(img):
    image_filename = f'assets/{img}'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    src='data:image/png;base64,{}'.format(encoded_image.decode())
    return src



def create_layout():
    # تصميم شريط التنقل
    navbar = dbc.Navbar(
        [
            dbc.NavbarBrand(
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(src=image_source("olympics.png"), height="60px", style={"margin-left": "70px"}),
                                style={"margin-right": "5px"},
                            ),
                            dbc.Col(html.H1("Predictions", className="ml-2 align-self-center", style={"font-size": "20px", "text-decoration": "none", "color": "#2D3C6B", "text-decoration": "none", "font-weight": "bold"})),
                        ],
                        align="center",
                    ),
                    href="/",
                    style={"text-decoration": "none"}
                )
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink(html.Img(src=image_source("home.png"), height="40px"), href="/", style={"padding-left": "20px"})),
                    dbc.NavItem(dbc.NavLink(html.Img(src=image_source("pred.png"), height="40px"), href="/", style={"padding-left": "20px"})),
                ],
                className="mr-auto",
                navbar=True,
                style = {
                    "padding-left": "950px"
                }
            ),
        ],
        color="light",
        dark=False,
        className="shadow-sm mb-5 bg-white",
        sticky="top",
        style={
            "height": "80px",
            "width": "100%", 
        }
    )

    # تصميم محتوى التطبيق
    layout = dbc.Card(
        html.Div([
            html.Div([
                navbar
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Please Enter Your Data", className='display-5', style={
                            'margin-bottom': '40px',
                            'font_weight': 'bold',
                            'color': 'black',
                            'margin-top': '10px',
                            'text-align': 'left',
                            'font-size': '20px'
                        }),
                        # إضافة حقول اختيار البيانات هنا
                    ], style={'width': '35%', 'margin-left': '32%', 'margin-right': 'auto', 'margin-top': '50px'}),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predicted Sport", className='display-5', style={
                            'margin-bottom': '40px',
                            'font_weight': 'bold',
                            'color': 'black',
                            'margin-top': '10px',
                            'text-align': 'left',
                            'font-size': '20px'
                        }),
                        html.Div(id='prediction-output', style={'font-size': '20px', 'margin-top': '20px'}),
                        html.Div(id='sport-image', style={'text-align': 'center', 'margin-top': '20px'}),
                        html.Div(id='sport-description', style={'font-size': '16px', 'margin-top': '20px'}),
                    ], style={'width': '35%', 'margin-left': 'auto', 'margin-right': '32%', 'margin-top': '50px'}),
                ]),
            ]),
        ], style={'margin-top': '20px'})
    )

    return layout
















    

# Define callback for prediction
@app.callback(
    [Output('prediction-output', 'children'),
     Output('sport-image', 'children'),
     Output('sport-description', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('age-dropdown', 'value'),
     State('weight-dropdown', 'value'),
     State('height-dropdown', 'value'),
     State('gender-dropdown', 'value'),
     State('country-dropdown', 'value'),
     State('sport-type-dropdown', 'value')]
)
def predict_sport(n_clicks, age, weight, height, gender, country, sport_type):
    if n_clicks is None:
        return '', '', ''
    else:
        # Preprocess data
        data = {'Age': age, 'Weight': weight, 'Height': height, 'Gender': gender, 'Country': country, 'Sport_Type': sport_type}
        df_pred = pd.DataFrame(data, index=[0])
        df_pred['Gender'] = ordinal_sec_func.transform(df_pred['Gender'].values.reshape(-1, 1))
        df_pred['Country'] = one_hot_encoder_sec_func.transform(df_pred['Country'].values.reshape(-1, 1)).toarray()
        df_pred['Sport_Type'] = one_hot_encoder_main_func.transform(df_pred['Sport_Type'].values.reshape(-1, 1)).toarray()
        df_pred = df_pred[features_main_func]
        df_pred = scaler_main_func.transform(df_pred)

        # Predict sport using secondary and main models
        sport_sec = model_sec.predict(df_pred)[0]
        sport_main = model_main.predict(df_pred)[0]

        # Get sport name and description
        sport_name = gender_sort.loc[gender_sort['Code'] == sport_main, 'Sport'].values[0]
        sport_description = gender_sort.loc[gender_sort['Code'] == sport_main, 'Description'].values[0]

        # Display prediction results
        prediction_output = f"Predicted Sport: **{sport_name}**"
        sport_image = html.Img(src=image_source(f"{sport_name.lower()}.png"), height="200px")
        sport_description = f"Description: {sport_description}"

        return prediction_output, sport_image, sport_description

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)




