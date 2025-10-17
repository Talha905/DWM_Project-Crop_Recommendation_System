# app.py - Enhanced Flask application for Crop Recommendation System with External APIs and Plotly Graphs

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, flash, redirect, url_for
import requests
import os
import sys  # Added for error printing
from google import generativeai as genai # For Gemini API
from dotenv import load_dotenv
load_dotenv() 
import re  # For JSON extraction

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # For flash messages

# Retrieve Gemini API Key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables. Please set it using 'export GEMINI_API_KEY=your_key'.", file=sys.stderr)
    sys.exit(1)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Load the dataset with error handling
DATASET_PATH = 'Crop_recommendation.csv'
try:
    print("Loading dataset...")  # Debug print
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")  # Debug print

    # Define the feature columns explicitly
    feature_columns = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']

    # Clean the data: Keep only feature columns and label
    df = df[feature_columns + ['label']]
    if '' in df.columns:
        df = df.drop(columns=[''])
except FileNotFoundError as e:
    print(f"Error: CSV file not found at {DATASET_PATH}. Please place it in the same directory as app.py.", file=sys.stderr)
    sys.exit(1)  # Exit with error
except Exception as e:
    print(f"Unexpected error loading dataset: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Function to fetch weather data from Open-Meteo (no API key needed)
def fetch_weather(city):
    try:
        # Get coordinates from geocode
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geocode_url).json()
        if 'results' not in geo_response or not geo_response['results']:
            return None
        lat = geo_response['results'][0]['latitude']
        lon = geo_response['results'][0]['longitude']

        # Fetch current weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain&daily=rain_sum&timezone=auto"
        weather_response = requests.get(weather_url).json()
        current = weather_response.get('current', {})
        daily = weather_response.get('daily', {})
        
        return {
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'rainfall': daily.get('rain_sum', [0])[0] if daily else 0  # Daily rain sum
        }
    except Exception:
        return None

# Function to fetch crop info using Gemini API
def fetch_crop_info(crop_name):
    try:
        print(f"Attempting to fetch info for crop: {crop_name}")  # Debug log
        prompt = f"""
        Provide detailed information about the crop "{crop_name}" in the following JSON format only:
        {{
            "common_name": "Common name of the crop",
            "scientific_name": "Scientific name of the crop",
            "description": "A brief description of the crop, its uses(Atleast 2), and growing conditions",
            "watering": {{
                "value": "Frequency",
                "unit": "days or weeks"
            }},
            "sunlight": ["List of sunlight requirements"]
        }}
        Do not add any extra text or explanations outside the JSON.
        """
        
        print(f"Sending prompt to Gemini: {prompt}")  # Debug log
        response = model.generate_content(prompt)
        print(f"Raw response from Gemini: {response.text}")  # Debug log
        
        content = response.text.strip()
        print(f"Stripped content: {content}")  # Debug log
        
        # Extract JSON if extra text is present
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        # Parse JSON from response
        import json
        parsed = json.loads(content)
        print(f"Parsed JSON: {parsed}")  # Debug log
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)} - Content was: {content}")
        return {
            'common_name': f'{crop_name.capitalize()} (Generic)',
            'scientific_name': 'Unknown',
            'description': f'Information about {crop_name} could not be fetched due to JSON parsing failure. It is a crop requiring typical agricultural conditions.',
            'watering': {'value': 'Moderate', 'unit': 'days'},
            'sunlight': ['Full Sun', 'Partial Shade']
        }
    except Exception as e:
        print(f"Error fetching crop info from Gemini: {str(e)}")
        return {
            'common_name': f'{crop_name.capitalize()} (Generic)',
            'scientific_name': 'Unknown',
            'description': f'Information about {crop_name} could not be fetched. It is a crop requiring typical agricultural conditions.',
            'watering': {'value': 'Moderate', 'unit': 'days'},
            'sunlight': ['Full Sun', 'Partial Shade']
        }

# Home route - Redirect to analysis
@app.route('/')
def home():
    return redirect(url_for('analysis'))

# Dashboard - Analysis
@app.route('/analysis')
def analysis():
    # Dataset Overview
    overview = df.head().to_html(classes='table table-striped', index=False)
    shape = f"{df.shape[0]} rows, {df.shape[1]} columns"

    # Summary Statistics
    summary = df.describe().to_html(classes='table table-striped')

    # Crop Distribution - Plotly
    fig_crop_dist = px.bar(df['label'].value_counts().reset_index(), x='label', y='count',
                           title="Crop Distribution", color='label', color_discrete_sequence=px.colors.qualitative.Set2)
    crop_dist_div = fig_crop_dist.to_html(full_html=False, include_plotlyjs='cdn')

    # Correlation Heatmap - Plotly
    corr = df[feature_columns].corr()
    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', texttemplate="%{z:.2f}"))
    fig_corr.update_layout(title="Feature Correlations")
    corr_div = fig_corr.to_html(full_html=False, include_plotlyjs='cdn')

    # Pairplot - Use Plotly Scatter Matrix (sampled)
    sample_df = df.sample(frac=0.1)
    fig_pair = px.scatter_matrix(sample_df, dimensions=feature_columns, color='label', title="Pairplot of Features")
    pair_div = fig_pair.to_html(full_html=False, include_plotlyjs='cdn')

    # Boxplots - Generate list of Plotly divs
    boxplot_divs = []
    for feature in feature_columns:
        fig_box = px.box(df, x='label', y=feature, color='label', title=f"{feature} by Crop")
        boxplot_divs.append(fig_box.to_html(full_html=False, include_plotlyjs='cdn'))

    return render_template('analysis.html', overview=overview, shape=shape, summary=summary,
                           crop_dist_div=crop_dist_div, corr_div=corr_div, pair_div=pair_div,
                           boxplot_divs=boxplot_divs)

# Dashboard - Classification
@app.route('/classification')
def classification():
    # Prepare data
    X = df[feature_columns]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = f"{accuracy_score(y_test, y_pred):.2f}"
    class_report = classification_report(y_test, y_pred)

    # Confusion Matrix - Plotly
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=model.classes_, y=model.classes_, colorscale='Blues', texttemplate="%{z}", textfont={"size":12}))
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    cm_div = fig_cm.to_html(full_html=False, include_plotlyjs='cdn')

    # Feature Importance - Plotly
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': importances}).sort_values('Importance', ascending=False)
    fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', title="Feature Importance",
                     color='Importance', color_continuous_scale='Magma')
    imp_div = fig_imp.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('classification.html', accuracy=accuracy, class_report=class_report,
                           cm_div=cm_div, imp_div=imp_div)

# Dashboard - Clustering
@app.route('/clustering')
def clustering():
    # Prepare data (standardize)
    X = df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method - Plotly
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    fig_elbow = px.line(x=range(1, 11), y=inertia, markers=True, title='Elbow Method', labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    fig_elbow.update_traces(line_color='teal', marker=dict(color='teal'))
    elbow_div = fig_elbow.to_html(full_html=False, include_plotlyjs='cdn')

    # KMeans with optimal k=4
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clustered['PC1'] = X_pca[:, 0]
    df_clustered['PC2'] = X_pca[:, 1]

    # Cluster plot - Plotly
    fig_cluster = px.scatter(df_clustered, x='PC1', y='PC2', color='Cluster', title="Clusters in PCA Space",
                             color_continuous_scale='Viridis', size_max=100)
    cluster_div = fig_cluster.to_html(full_html=False, include_plotlyjs='cdn')

    # Clusters vs Actual Crops - Plotly
    fig_crops = px.scatter(df_clustered, x='PC1', y='PC2', color='label', title="Actual Crops in PCA Space",
                           color_discrete_sequence=px.colors.qualitative.Set1, size_max=100)
    crops_div = fig_crops.to_html(full_html=False, include_plotlyjs='cdn')

    # Cluster centers
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_columns)
    centers_html = centers.to_html(classes='table table-striped')

    return render_template('clustering.html', elbow_div=elbow_div, cluster_div=cluster_div,
                           crops_div=crops_div, centers_html=centers_html)

# Crop Recommendation
@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    weather_data = None
    city = ''
    prediction = None
    prob_div = None
    contrib_div = None
    explanation = None
    crop_info = None
    nitrogen = request.form.get('nitrogen', 50.0)
    phosphorus = request.form.get('phosphorus', 50.0)
    potassium = request.form.get('potassium', 50.0)
    temperature = request.form.get('temperature', 25.0)
    humidity = request.form.get('humidity', 50.0)
    ph = request.form.get('ph', 6.5)
    rainfall = request.form.get('rainfall', 100.0)

    if request.method == 'POST':
        city = request.form.get('city', '')
        action = request.form.get('action')  # Distinguish between buttons

        if action == 'fetch_weather' and city:
            weather_data = fetch_weather(city)
            if weather_data:
                flash('Weather data fetched successfully!', 'success')
                temperature = weather_data['temperature']
                humidity = weather_data['humidity']
                rainfall = weather_data['rainfall']
            else:
                flash('Could not fetch weather for the city.', 'danger')

        elif action == 'recommend_crop':
            try:
                # Get user inputs (override with form values)
                nitrogen = float(request.form['nitrogen'])
                phosphorus = float(request.form['phosphorus'])
                potassium = float(request.form['potassium'])
                temperature = float(request.form['temperature'])
                humidity = float(request.form['humidity'])
                ph = float(request.form['ph'])
                rainfall = float(request.form['rainfall'])

                # Train the model
                X = df[feature_columns]
                y = df['label']
                model = RandomForestClassifier(random_state=42)
                model.fit(X, y)

                # Predict
                input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
                prediction = model.predict(input_data)[0]

                # Fetch crop info using Gemini API
                crop_info = fetch_crop_info(prediction)

                # Probabilities - Plotly Table
                probs = model.predict_proba(input_data)[0]
                prob_df = pd.DataFrame({'Crop': model.classes_, 'Probability': probs}).sort_values('Probability', ascending=False)
                fig_prob = go.Figure(data=[go.Table(header=dict(values=list(prob_df.columns)),
                                                   cells=dict(values=[prob_df.Crop, prob_df.Probability]))])
                fig_prob.update_layout(title="Prediction Probabilities")
                prob_div = fig_prob.to_html(full_html=False, include_plotlyjs='cdn')

                # Feature contributions - Plotly Bar
                importances = model.feature_importances_
                input_values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
                feature_contrib = pd.DataFrame({
                    'Feature': feature_columns,
                    'Input Value': input_values,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                fig_contrib = px.bar(feature_contrib, x='Importance', y='Feature', orientation='h',
                                     title=f"Feature Importance for {prediction} Recommendation",
                                     color='Importance', color_continuous_scale='Greens')
                contrib_div = fig_contrib.to_html(full_html=False, include_plotlyjs='cdn')

                # Top features explanation
                top_features = feature_contrib.head(3)
                explanation = f"The model chose <strong>{prediction}</strong> because of the following key factors:<br>"
                for index, row in top_features.iterrows():
                    explanation += f"- <strong>{row['Feature']}</strong>: Your input value of {row['Input Value']:.2f} was significant, with an importance score of {row['Importance']:.3f}.<br>"
                explanation += "These values align with conditions favorable for the recommended crop based on the model's training data."

            except ValueError:
                flash('Invalid input. Please enter numeric values.', 'danger')

    return render_template('recommendation.html', prediction=prediction, prob_div=prob_div,
                           contrib_div=contrib_div, explanation=explanation, crop_info=crop_info,
                           weather_data=weather_data, city=city,
                           nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
                           temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall)

if __name__ == '__main__':
    print("Starting Flask server...")  # Debug print
    app.run(debug=True)