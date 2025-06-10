import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import joblib
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create and preprocess synthetic dataset (40 samples, balanced)
data = {
    'fever': ['high', 'low', 'none', 'high', 'low', 'none', 'high', 'none', 'low', 'high',
              'none', 'high', 'low', 'none', 'high', 'none', 'none', 'high', 'low', 'none',
              'high', 'low', 'none', 'high', 'none', 'low', 'high', 'none', 'low', 'high',
              'none', 'high', 'low', 'none', 'high', 'none', 'none', 'high', 'low', 'none'],
    'cough': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes',
              'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes',
              'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes',
              'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
    'fatigue': ['yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes',
                'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes',
                'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes',
                'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
    'headache': ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes',
                 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes',
                 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no',
                 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
    'temperature': [30, 20, 25, 35, 22, 18, 32, 24, 21, 33,
                    28, 26, 23, 19, 34, 20, 27, 31, 22, 29,
                    36, 21, 24, 33, 19, 28, 30, 25, 22, 34,
                    29, 23, 26, 32, 20, 27, 31, 24, 21, 35],
    'measure': ['rest_hydrate_doctor', 'rest_hydrate', 'monitor_hydrate', 'rest_doctor',
                'monitor', 'rest_hydrate', 'doctor_hydrate', 'monitor_hydrate',
                'rest', 'doctor_hydrate',
                'monitor', 'rest_doctor', 'rest_hydrate', 'monitor_hydrate',
                'doctor_hydrate', 'rest', 'monitor_hydrate', 'rest_doctor',
                'rest_hydrate', 'monitor',
                'rest_doctor', 'monitor_hydrate', 'rest', 'doctor_hydrate',
                'monitor', 'rest_hydrate', 'doctor_hydrate', 'monitor_hydrate',
                'rest', 'rest_doctor',
                'monitor_hydrate', 'rest_hydrate', 'doctor_hydrate', 'monitor',
                'rest_doctor', 'rest', 'monitor_hydrate', 'rest_hydrate_doctor',
                'monitor', 'doctor_hydrate']
}
df = pd.DataFrame(data)

# Ensure 'temperature' is numeric
df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce').fillna(18.0).astype(float)

# Encode categorical variables
le_fever = LabelEncoder()
le_cough = LabelEncoder()
le_fatigue = LabelEncoder()
le_headache = LabelEncoder()
le_measure = LabelEncoder()

df['fever'] = le_fever.fit_transform(df['fever'])
df['cough'] = le_cough.fit_transform(df['cough'])
df['fatigue'] = le_fatigue.fit_transform(df['fatigue'])
df['headache'] = le_headache.fit_transform(df['headache'])
df['measure'] = le_measure.fit_transform(df['measure'])

# Features and target
X = df[['fever', 'cough', 'fatigue', 'headache', 'temperature']]
y = df['measure']

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Debug: Print class distribution before SMOTE
print("Pre-SMOTE y_train Class Counts:", dict(zip(le_measure.classes_, np.bincount(y_train, minlength=len(le_measure.classes_)))))

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42, k_neighbors=1)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Step 2: Compare multiple ML algorithms
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50),
    'SVM': SVC(random_state=42, kernel='linear', C=1.0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'accuracy': accuracy, 'f1_score': f1}
    print(f"\n{name} - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [le_measure.classes_[i] for i in unique_classes]
    print(f"{name} Unique Classes: {unique_classes} ({target_names})")
    print(f"{name} y_test Class Counts: {dict(zip(le_measure.classes_, np.bincount(y_test, minlength=len(le_measure.classes_))))}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))
    print(f"{name} Predictions: {y_pred}")

# Debug: Print results
print("\nResults dictionary:", results)

# Select best model (Random Forest for deployment)
best_model = RandomForestClassifier(random_state=42, n_estimators=50)
best_model.fit(X_train, y_train)

# Save best model and encoders
joblib.dump(best_model, 'symptom_measure_model.pkl')
joblib.dump(le_fever, 'le_fever.pkl')
joblib.dump(le_cough, 'le_cough.pkl')
joblib.dump(le_fatigue, 'le_fatigue.pkl')
joblib.dump(le_headache, 'le_headache.pkl')
joblib.dump(le_measure, 'le_measure.pkl')

# Step 3: Fetch real-time weather data
def get_weather_data(city, api_key="your weather API key", fallback_city="Nairobi"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    # Set up retry strategy
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(url, timeout=10)
        print(f"API Response Status Code: {response.status_code}")
        print(f"API Response Content: {response.text}")
        if response.status_code == 200:
            data = response.json()
            if 'main' in data and 'temp' in data['main']:
                return data['main']['temp']
            else:
                st.error(f"Invalid response format for city '{city}'. Trying fallback city '{fallback_city}'.")
        else:
            error_message = data.get('message', 'Unknown error') if 'data' in locals() else 'No error message'
            st.error(f"API error (Status {response.status_code}): {error_message}. Trying fallback city '{fallback_city}'.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to weather API for '{city}': {str(e)}. Trying fallback city '{fallback_city}'.")
        print(f"Request Exception: {str(e)}")
    
    # Try fallback city
    if city.lower() != fallback_city.lower():
        fallback_url = f"http://api.openweathermap.org/data/2.5/weather?q={fallback_city}&appid={api_key}&units=metric"
        try:
            response = session.get(fallback_url, timeout=10)
            print(f"Fallback API Response Status Code: {response.status_code}")
            print(f"Fallback API Response Content: {response.text}")
            if response.status_code == 200:
                data = response.json()
                if 'main' in data and 'temp' in data['main']:
                    st.warning(f"Using temperature from fallback city '{fallback_city}'.")
                    return data['main']['temp']
                else:
                    st.error(f"Invalid response format for fallback city '{fallback_city}'. Using default temperature (18°C).")
            else:
                error_message = data.get('message', 'Unknown error') if 'data' in locals() else 'No error message'
                st.error(f"API error for fallback city (Status {response.status_code}): {error_message}. Using default temperature (18°C).")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to weather API for fallback city '{fallback_city}': {str(e)}. Using default temperature (18°C).")
            print(f"Fallback Request Exception: {str(e)}")
    
    return 18.0

# Step 4: Streamlit Web App
st.title("Symptom-Based Health Measure Recommender")
st.write("Enter your symptoms and city to get personalized health recommendations.")

# User inputs
fever = st.selectbox("Fever", ['none', 'low', 'high'])
cough = st.selectbox("Cough", ['no', 'yes'])
fatigue = st.selectbox("Fatigue", ['no', 'yes'])
headache = st.selectbox("Headache", ['no', 'yes'])
city = st.text_input("Enter your city", "Nairobi")

# Load model and encoders
model = joblib.load('symptom_measure_model.pkl')
le_fever = joblib.load('le_fever.pkl')
le_cough = joblib.load('le_cough.pkl')
le_fatigue = joblib.load('le_fatigue.pkl')
le_headache = joblib.load('le_headache.pkl')
le_measure = joblib.load('le_measure.pkl')

# Predict when user submits
if st.button("Get Recommendation"):
    # Fetch weather data
    temperature = get_weather_data(city)
    st.write(f"Current temperature in {city}: {temperature}°C")

    # Prepare input
    input_data = pd.DataFrame({
        'fever': [fever],
        'cough': [cough],
        'fatigue': [fatigue],
        'headache': [headache],
        'temperature': [temperature]
    })
    input_data['fever'] = le_fever.transform(input_data['fever'])
    input_data['cough'] = le_cough.transform(input_data['cough'])
    input_data['fatigue'] = le_fatigue.transform(input_data['fatigue'])
    input_data['headache'] = le_headache.transform(input_data['headache'])
    input_data['temperature'] = input_data['temperature'].astype(float)

    # Predict
    prediction = model.predict(input_data)
    measure = le_measure.inverse_transform(prediction)[0]
    st.success(f"Recommended measure: {measure.replace('_', ' and ')}")

# Step 5: Retrain Models Button
if st.button("Retrain Models"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(1000), stratify=y)
    print("New Pre-SMOTE y_train Class Counts:", dict(zip(le_measure.classes_, np.bincount(y_train, minlength=len(le_measure.classes_)))))
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'accuracy': accuracy, 'f1_score': f1}
        print(f"\n{name} - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
    st.session_state['results'] = results  # Store results for plotting

# Use stored results if available
results = st.session_state.get('results', results)

# Step 6: Visualize model comparison
# Main Model Performance Comparison
st.subheader("Model Performance Comparison")
if results and all(np.isfinite([results[name]['accuracy'] for name in results])) and \
      all(np.isfinite([results[name]['f1_score'] for name in results])):
    # Scale metrics for visibility
    accuracies = [max(results[name]['accuracy'], 0.1) for name in results]
    f1_scores = [max(results[name]['f1_score'], 0.1) for name in results]

    # Warn if all metrics are at minimum
    if all(a == 0.1 for a in accuracies) and all(f == 0.1 for f in f1_scores):
        st.warning("All metrics are very low or zero, using minimum value for visualization.")

    fig, ax = plt.subplots(figsize=(8, 6))
    model_names = list(results.keys())

    # Debug: Print values to be plotted
    print("Accuracies to plot:", accuracies)
    print("F1 Scores to plot:", f1_scores)

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4')
    ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)  # Fix y-axis
    ax.set_title('Model Performance Comparison')
    ax.legend()
    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)

    # Save plot
    plt.savefig('model_comparison.png')
    plt.close(fig)
else:
    st.error("No valid model performance data available. Displaying dummy chart.")
    fig, ax = plt.subplots(figsize=(8, 6))
    model_names = ['Decision Tree', 'Random Forest', 'SVM']
    dummy_accuracies = [0.5, 0.6, 0.55]
    dummy_f1_scores = [0.45, 0.58, 0.50]
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, dummy_accuracies, width, label='Accuracy', color='#1f77b4')
    ax.bar(x + width/2, dummy_f1_scores, width, label='F1 Score', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title('Dummy Model Performance Comparison')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.savefig('dummy_model_comparison.png')
    plt.close(fig)