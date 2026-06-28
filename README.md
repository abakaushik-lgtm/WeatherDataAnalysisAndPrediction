# Weather Data Analysis & Temperature Prediction Web App

A professional, interactive Web Dashboard built with Python, Streamlit, and Machine Learning. The application provides exploratory data analysis (EDA), data cleaning tools, and comparative evaluations of three different regression algorithms to predict temperatures based on atmospheric parameters.

This project is structured, commented, and presented to meet the requirements of an **AI/ML Internship Project**.

---

## 🌟 Key Features

1. **Modern Dashboard Interface**: Adaptive design styling with global typography (Outfit font) and responsive layout adapting to dark and light modes.
2. **Interactive EDA (Exploratory Data Analysis)**:
   - Line graphs tracking Temperature and Humidity trends over time.
   - Histograms visualizing Rainfall distributions and Box plots measuring monthly variances.
   - Dynamic scatter plots with trendlines.
   - Pearson correlation heatmaps visualizing feature co-dependencies.
3. **Data Cleaning Pipeline**: Automated duplicate removal and missing value imputation using median values for numerical inputs.
4. **Machine Learning Suite**:
   - Split controls (e.g. 80/20 train/test split).
   - Training and comparisons of **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**.
   - Side-by-side performance table showing MAE, MSE, RMSE, and $R^2$ Score.
   - Automatic best-model selection.
   - Feature importance mapping (Random Forest).
5. **Robust Temperature Forecasts**:
   - Interactive prediction form for Humidity, Wind Speed, Pressure, and Rainfall.
   - Temperature estimation and gauge dial visualizations.
   - Scientific **Confidence Score** based on model accuracy and the statistical distance of input data from training distributions (Z-scores).
   - CSV result export containing timestamp, inputs, predicted outputs, and model parameters.

---

## 📂 Project Directory Structure

```text
Weather-Prediction/
│── app.py                   # Main Streamlit dashboard application
│── utils.py                 # Core analytical pipeline functions
│── requirements.txt         # Package dependencies file
│── README.md                # Project documentation
│── dataset/                 # Raw/Uploaded CSVs
│   └── sample_weather.csv   # Automatically generated realistic weather dataset (1000 days)
└── models/                  # Serialized pipelines and model binaries
    ├── best_model.pkl       # Serialized best-performing estimator
    ├── scaler.pkl           # Fitted feature scaler (StandardScaler)
    ├── metadata.pkl         # Validation benchmarks and configuration dict
    └── x_train_vals.pkl     # Reference training distribution for confidence score math
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Custom HTML/CSS Styling
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
- **Machine Learning**: Scikit-Learn
- **Model Serialization**: Joblib

---

## 🚀 Getting Started (Run Locally)

Follow these steps to run the application on your computer:

### 1. Prerequisites
Make sure you have **Python 3.8+** installed.

### 2. Clone or Extract the Project
Open your terminal or command prompt in the project root folder.

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
streamlit run app.py
```
After executing, Streamlit will print the local URL (typically `http://localhost:8501`) which will open automatically in your browser.

---

## 🧠 Behind the Scenes: Machine Learning Pipeline

### Feature Columns
The models use the following features to forecast temperature:
- **Humidity (%)**
- **Wind Speed (km/h)**
- **Atmospheric Pressure (hPa)**
- **Rainfall (mm)**

### Models Evaluated
- **Linear Regression**: Fits a linear model with coefficients to minimize residual sum of squares.
- **Decision Tree Regressor**: Non-parametric tree model that divides data based on feature thresholds.
- **Random Forest Regressor**: Ensemble learning method that averages predictions of 100 decision trees to control overfitting.

### Scientific Confidence Score
Rather than outputting a static score, this application computes a dynamic **Confidence Score** based on the input values' statistical distance from the training distribution:
1. **Baseline Confidence**: Derived directly from the best model's $R^2$ validation score.
2. **Outlier Penalty**: Computes the Z-score for all user inputs relative to the mean and standard deviation of the training dataset.
3. If the input parameters have an average Z-score higher than $1.5$ standard deviations, the confidence degrades proportionally. This warns the user when they are attempting to extrapolate predictions using unrealistic weather inputs.
