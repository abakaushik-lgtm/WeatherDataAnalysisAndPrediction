import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define directory paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_synthetic_weather_data(filepath, num_days=1000, seed=42):
    """
    Generates a realistic synthetic weather dataset and saves it to CSV.
    Intentionally introduces some duplicates and missing values to demonstrate data cleaning.
    """
    np.random.seed(seed)
    
    # Generate dates
    start_date = pd.to_datetime("2023-01-01")
    dates = pd.date_range(start_date, periods=num_days, freq="D")
    
    # 1. Temperature: Seasonal pattern (sinusoidal) + random noise
    # Standard temp around 20 degrees with +/- 10 degrees seasonal swing
    day_of_year = dates.dayofyear
    seasonal_temp = 20.0 + 10.0 * np.sin(2 * np.pi * day_of_year / 365.0 - np.pi / 2)
    temp_noise = np.random.normal(0, 3.0, size=num_days)
    temperature = seasonal_temp + temp_noise
    
    # 2. Humidity: Inversely correlated with temperature, with noise
    # Hotter days tend to have lower relative humidity, colder days higher
    humidity_base = 75.0 - 1.2 * (temperature - 20.0)
    humidity_noise = np.random.normal(0, 8.0, size=num_days)
    humidity = np.clip(humidity_base + humidity_noise, 15.0, 100.0)
    
    # 3. Pressure: Inversely correlated with humidity and temperature (low pressure system -> rain/humid)
    pressure_base = 1013.25 - 0.15 * (humidity - 60.0)
    pressure_noise = np.random.normal(0, 5.0, size=num_days)
    pressure = pressure_base + pressure_noise
    
    # 4. Rainfall: Higher probability when humidity is high and pressure is low
    # P(Rain) = sigmoid(0.1 * (Humidity - 75) - 0.05 * (Pressure - 1013.25))
    logit = 0.1 * (humidity - 75.0) - 0.08 * (pressure - 1013.25)
    prob_rain = 1.0 / (1.0 + np.exp(-logit))
    
    rainfall = []
    for p in prob_rain:
        if np.random.random() < p:
            # If it rains, generate amount using exponential distribution (average 8mm, max 50mm)
            rain_amt = min(50.0, np.random.exponential(scale=8.0))
            rainfall.append(round(rain_amt, 1))
        else:
            rainfall.append(0.0)
    rainfall = np.array(rainfall)
    
    # Adjust temperature slightly down if it rains
    temperature = temperature - 0.15 * rainfall
    
    # 5. Wind Speed: Random variation, slightly higher when pressure is low (storm systems)
    wind_base = 12.0 + 0.3 * np.maximum(0, 1013.25 - pressure)
    wind_noise = np.random.normal(0, 4.0, size=num_days)
    wind_speed = np.clip(wind_base + wind_noise, 0.0, 55.0)
    
    # Build dataframe
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Temperature": np.round(temperature, 1),
        "Humidity": np.round(humidity, 1),
        "Wind Speed": np.round(wind_speed, 1),
        "Pressure": np.round(pressure, 1),
        "Rainfall": rainfall
    })
    
    # Inject intentional duplicates (approx 2% duplicates)
    dup_indices = np.random.choice(num_days, size=int(num_days * 0.02), replace=False)
    duplicates = df.iloc[dup_indices].copy()
    # Modify date slightly for duplicates or keep identical
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Inject intentional missing values (approx 3% in random columns, except Date)
    columns_to_corrupt = ["Temperature", "Humidity", "Wind Speed", "Pressure", "Rainfall"]
    for col in columns_to_corrupt:
        nan_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        df.loc[nan_indices, col] = np.nan
        
    # Sort by date (duplicates will be clustered near original locations)
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath

def clean_weather_data(df):
    """
    Cleans the weather dataframe:
    1. Removes duplicate rows.
    2. Imputes missing values in numerical columns with their median.
    3. Formats the Date column.
    Returns: cleaned_df, num_duplicates_removed, missing_values_imputed_count
    """
    cleaned_df = df.copy()
    
    # Format Date
    if "Date" in cleaned_df.columns:
        cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"]).dt.strftime("%Y-%m-%d")
        
    # Count duplicates before removal
    duplicates_mask = cleaned_df.duplicated()
    num_duplicates = duplicates_mask.sum()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Count missing values
    missing_counts = cleaned_df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    
    # Impute missing values for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if cleaned_df[col].isnull().sum() > 0:
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            
    return cleaned_df, num_duplicates, total_missing, missing_counts

def train_and_evaluate_models(df, test_size=0.2, random_state=42):
    """
    Trains Linear Regression, Random Forest, and Decision Tree regressors.
    Features: Humidity, Wind Speed, Pressure, Rainfall
    Target: Temperature
    Saves scaler and best model to 'models/' directory.
    Returns: models_dict, scaler, metrics_df, best_model_name, feature_names
    """
    feature_names = ["Humidity", "Wind Speed", "Pressure", "Rainfall"]
    target_name = "Temperature"
    
    # Drop rows where feature/target columns have NaNs (should be clean already, but safe check)
    clean_df = df.dropna(subset=feature_names + [target_name])
    
    X = clean_df[feature_names]
    y = clean_df[target_name]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=7, random_state=random_state)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R² Score": round(r2, 4)
        })
        
    metrics_df = pd.DataFrame(results)
    
    # Select best model based on R² Score
    best_idx = metrics_df["R² Score"].idxmax()
    best_model_name = metrics_df.loc[best_idx, "Model"]
    best_model = trained_models[best_model_name]
    
    # Save the scaler and the best model
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    
    # Save model details/metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "metrics": metrics_df.to_dict(orient="records"),
        "r2_score": float(metrics_df.loc[best_idx, "R² Score"]),
        "train_mean": float(y_train.mean()),
        "train_std": float(y_train.std())
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))
    
    # Also save training data for confidence calculation
    joblib.dump(X_train.values, os.path.join(MODELS_DIR, "x_train_vals.pkl"))
    
    return trained_models, scaler, metrics_df, best_model_name, feature_names

def load_best_model():
    """
    Loads the saved best model, scaler, and metadata.
    Returns: model, scaler, metadata, x_train_vals
    """
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    metadata_path = os.path.join(MODELS_DIR, "metadata.pkl")
    x_train_path = os.path.join(MODELS_DIR, "x_train_vals.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path)):
        return None, None, None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    
    x_train_vals = None
    if os.path.exists(x_train_path):
        x_train_vals = joblib.load(x_train_path)
        
    return model, scaler, metadata, x_train_vals

def calculate_confidence_score(user_inputs, x_train_vals, r2_score):
    """
    Calculates a prediction confidence score (0-100%) based on:
    1. Model's test R² score (representing overall baseline confidence).
    2. Proximity of inputs to the training data distribution (using Z-scores).
    """
    # Baseline confidence starts at R2 score * 100
    baseline_conf = max(10.0, r2_score * 100.0)
    
    if x_train_vals is None:
        return round(baseline_conf, 1)
        
    # Calculate Z-scores for each input parameter compared to the training set distribution
    z_scores = []
    # user_inputs is a dictionary/list: [Humidity, Wind Speed, Pressure, Rainfall]
    for i in range(len(user_inputs)):
        col_vals = x_train_vals[:, i]
        mean = np.mean(col_vals)
        std = np.std(col_vals)
        if std > 0:
            z = abs(user_inputs[i] - mean) / std
        else:
            z = 0.0
        z_scores.append(z)
        
    # Penalize confidence for outlier inputs (Z-score > 1.5)
    # Average Z-score penalty: if avg_z is high, decrease confidence
    avg_z = np.mean(z_scores)
    
    # Penalty function: Z-scores up to 1.5 have no penalty. Above 1.5, we scale down
    penalty = 0.0
    if avg_z > 1.5:
        penalty = (avg_z - 1.5) * 15.0  # 15% reduction per unit Z-score deviation above 1.5
        
    confidence = baseline_conf - penalty
    # Ensure confidence stays within a logical range [10%, 98%] (never 100% since models have residual error)
    confidence = np.clip(confidence, 10.0, 98.5)
    
    return round(float(confidence), 1)
