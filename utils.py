import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'water_quality.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.joblib')

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    df.fillna(df.mean(), inplace=True)
    
    X = df.drop("WQI", axis=1)
    y = df["WQI"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
        "Support Vector Regressor": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100))
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}
        
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = models[best_model_name]
    
    joblib.dump(best_model, MODEL_PATH)
    
    metrics = {
        "best_model": best_model_name,
        "results": results
    }
    joblib.dump(metrics, METRICS_PATH)
    
    return metrics

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)

def get_metrics():
    if not os.path.exists(METRICS_PATH):
        train_and_save_model()
    return joblib.load(METRICS_PATH)

def water_quality_advice(wqi_value):
    if wqi_value >= 90:
        return "Excellent water quality. Safe for drinking."
    elif wqi_value >= 70:
        return "Good water quality. Use basic filtration if needed."
    elif wqi_value >= 50:
        return "Medium quality. Apply filtration and chemical treatment."
    elif wqi_value >= 25:
        return "Poor quality. Needs advanced treatment + disinfection."
    else:
        return "Very poor. Not suitable for drinking. Requires full purification."
