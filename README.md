# Symptom-Based Health Measure Recommender

Welcome to a dope Python app that dishes out health tips based on your symptoms and local weather, all while vibing with **Sustainable Development Goal 3 (SDG 3)**: Good Health and Well-Being. Built for beginners like me, this Streamlit web app uses machine learning (Decision Tree, Random Forest, SVM) to recommend actions like "rest and hydrate and doctor" from symptoms (fever, cough, fatigue, headache) and temperature. It’s got slick blue/orange bars to compare model performance and pulls real-time temperature from the OpenWeatherMap API. Let’s get it running locally, no fuss! 😄

## Features
- **Health Recommendations**: Drop your symptoms (fever, cough, fatigue, headache) and city to get personalized health measures.
- **Weather Integration**: Grabs local temperature using the OpenWeatherMap API to make predictions smarter.
- **Model Comparison**: Shows off accuracy and F1 scores (~0.5–0.8) for three ML models with blue (accuracy) and orange (F1 score) bars.
- **Retrain Models Button**: Hit “Retrain Models” to shuffle the data and update the bar chart with new metrics.
- **Beginner Vibes**: Runs on a 40-sample synthetic dataset, balanced with SMOTE for solid predictions.
- **SDG 3 Love**: Makes health recommendations transparent and accessible for all.

## Tech Stack
- **Python**: The main language, keeping it simple.
- **Streamlit**: Builds the interactive web app.
- **Scikit-learn**: Powers the ML models and metrics.
- **Imbalanced-learn**: Balances data with SMOTE.
- **Pandas & NumPy**: Handles data like a pro.
- **Matplotlib**: Draws those sweet bar charts.
- **Requests**: Talks to the OpenWeatherMap API.
- **Joblib**: Saves models and encoders.

## Prerequisites
- **Python 3.8+**: Make sure it’s installed (check with `python --version` in your terminal).
- **OpenWeatherMap API Key**: Free tier, grab one at [openweathermap.org](https://openweathermap.org).
- **Internet Connection**: For the weather API and installing packages.

## Installation
1. **Download or Clone the Project**:
   - If you’re using GitHub, clone it:
     ```bash
     git clone https://github.com/your-username/symptom-measure-recommender.git
     cd symptom-measure-recommender
     ```
   - Or just download the project folder to your computer (e.g., `C:\Users\YourName\Desktop\medic`).

2. **Install Python Packages**:
   - Open your terminal (Command Prompt on Windows) and navigate to the project folder:
     ```bash
     cd C:\Users\YourName\Desktop\medic
     ```
   - Install the required packages:
     ```bash
     pip install numpy==1.23.5 scikit-learn==1.2.2 imbalanced-learn==0.10.1 streamlit pandas matplotlib requests joblib
     ```

3. **Get an OpenWeatherMap API Key**:
   - Head to [openweathermap.org](https://openweathermap.org) and sign up for a free account.
   - Go to “API Keys” in your profile, copy your key (it might take 10–30 minutes to activate).
   - Open `symptom_measure_app_visualized.py` in a text editor (e.g., Notepad or VS Code).
   - Find the `get_weather_data` function and replace `api_key="YOUR_API_KEY_HERE"` with your key:
     ```python
     def get_weather_data(city, api_key="YOUR_NEW_KEY", fallback_city="Nairobi"):
     ```

## Running the App
1. **Launch the App**:
   - In your terminal, navigate to the project folder (e.g., `C:\Users\YourName\Desktop\medic`):
     ```bash
     cd C:\Users\YourName\Desktop\medic
     ```
   - Run the Streamlit app:
     ```bash
     streamlit run symptom_measure_app_visualized.py
     ```

2. **Check It Out**:
   - Open your browser and go to:
     [http://localhost:8501/#symptom-based-health-measure-recommender](http://localhost:8501/#symptom-based-health-measure-recommender)
   - Enter symptoms (e.g., fever="high", cough="yes") and a city (e.g., "Mombasa").
   - Click “Get Recommendation” to see your health tip.
   - Hit “Retrain Models” to update the blue/orange bars with new model scores.

## How It Works
- **Data**: Uses a 40-sample dataset with symptoms (fever, cough, fatigue, headache, temperature) and 7 health measures (e.g., "doctor_hydrate").
- **Prep**: Encodes categories, balances data with SMOTE, and splits 80/20 for training/testing.
- **Models**: Trains Decision Tree, Random Forest (50 trees), and SVM (linear kernel).
- **Weather**: Pulls temperature for your city, falling back to Nairobi or 18°C if the API hiccups.
- **Bars**: Shows model accuracy and F1 scores (~0.5–0.8) in a bar chart (blue = accuracy, orange = F1 score).
- **Retrain**: Shuffles the data split to give new model scores when you click “Retrain Models”.

## Project Files

medic/├── symptom_measure_app_visualized.py  # The main app code├── symptom_measure_model.pkl          # Saved Random Forest model├── le_*.pkl                           # Saved encoders├── model_comparison.png               # Bar chart output├── README.md                          # This guide

## Troubleshooting
- **API Issues**:
  - **Status 401**: Get a new API key from [openweathermap.org](https://openweathermap.org).
  - **Timeout**: Check your internet or try a different city (e.g., “London”).
  - Test the API in your browser:
    ```
    http://api.openweathermap.org/data/2.5/weather?q=Mombasa&appid=YOUR_KEY&units=metric
    ```
- **Bars Missing**:
  - Check the terminal for `Accuracies to plot` and `F1 Scores to plot`.
  - Look at `model_comparison.png` in the project folder.
- **SMOTE Error**:
  - Ensure `k_neighbors=1` in `SMOTE(random_state=42, k_neighbors=1)` in the code.
- **Python Errors**:
  - Run `pip list` to confirm package versions match (`numpy==1.23.5`, etc.).
  - Reinstall packages if needed:
    ```bash
    pip uninstall numpy scikit-learn imbalanced-learn streamlit pandas matplotlib requests joblib -y
    pip install numpy==1.23.5 scikit-learn==1.2.2 imbalanced-learn==0.10.1 streamlit pandas matplotlib requests joblib
    ```

## Next Steps
- Add user symptoms to the dataset to make the models learn over time.
- Try other weather APIs for backup.
- Host the app online with Streamlit Cloud.
- Grow the dataset with real health data (with permission, of course!).

## Wanna Help?
Got ideas? Fork the project, tweak it, and send a pull request. Drop issues or suggestions on GitHub (if hosted).

## License
MIT License. Check [LICENSE](LICENSE) for the deets.

## Shoutouts
- Built with mad passion by a Python newbie chasing SDG 3 vibes.
- Big thanks to OpenWeatherMap for the free API and Streamlit for making web apps easy.
- Powered by “here we go” energy and a love for health tech! 😄

