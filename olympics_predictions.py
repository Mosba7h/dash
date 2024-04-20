import pickle
import numpy as np
import pandas as pd

def predict_athlete_performance(athlete_data, model_path='athlete_model.pkl'):
    """
    Predicts the performance of an athlete based on given data.

    Args:
        athlete_data: A dictionary containing athlete information.
        model_path: The path to the trained athlete performance prediction model.

    Returns:
        A dictionary containing predicted performance metrics.
    """

    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Extract features from athlete data
    features = np.array([
        athlete_data['Age'],
        athlete_data['Height'],
        athlete_data['Weight'],
        athlete_data['Sport'],
        athlete_data['Sex'],
        athlete_data['Year'],
    ])

    # Predict performance
    predictions = model.predict(features.reshape(1, -1))

    # Return predictions as a dictionary
    return {
        'Predicted Medal': predictions[0][0],
        'Predicted Rank': predictions[0][1],
    }

def predict_top_countries(df, year, model_path='top_countries_model.pkl'):
    """
    Predicts the top countries in terms of medal count for a given year.

    Args:
        df: The Olympic Games dataset.
        year: The year for which to make predictions.
        model_path: The path to the trained top countries prediction model.

    Returns:
        A list of the top countries in terms of predicted medal count.
    """

    # Filter data for the given year
    df_year = df[df['Year'] == year]

    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predict medal counts for each country
    predictions = model.predict(df_year[['NOC']])

    # Sort countries by predicted medal count
    df_year['Predicted Medal Count'] = predictions
    top_countries = df_year.sort_values(by='Predicted Medal Count', ascending=False).head(10)['NOC'].tolist()

    return top_countries
