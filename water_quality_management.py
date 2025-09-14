import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("water_quality.csv")
df.fillna(df.mean(), inplace=True)
X = df.drop("WQI", axis=1)
y = df["WQI"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel="rbf")
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name} -> MSE: {mse:.2f}, R²: {r2:.2f}")
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name}")
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
print("\n--- Water Quality Prediction System ---")
feature_names = list(X.columns)
user_values = []
for feature in feature_names:
    while True:
        try:
            val = float(input(f"Enter value for {feature}: "))
            user_values.append(val)
            break
        except ValueError:
            print("❌ Please enter a valid numeric value.")
user_df = pd.DataFrame([user_values], columns=feature_names)
predicted_wqi = best_model.predict(user_df)[0]
print(f"\n🔮 Predicted WQI = {predicted_wqi:.2f}")
print(f"💡 Suggestion: {water_quality_advice(predicted_wqi)}")
plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test)
    plt.subplot(1, len(models), i)
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(name)
    plt.xlabel("Actual WQI")
    plt.ylabel("Predicted WQI")
plt.tight_layout()
plt.show()
