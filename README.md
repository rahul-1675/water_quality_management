# Water Quality Prediction System

A Python-based system to predict **Water Quality Index (WQI)** using machine learning models. This project allows users to:

- Train and evaluate multiple regression models (Linear Regression, Random Forest, Support Vector Regressor).  
- Predict WQI for single inputs or batch data from a CSV file.  
- Provide actionable advice based on predicted WQI.

---

## Features

- **Data Preprocessing:** Handles missing values automatically.  
- **Model Training:** Splits data into 80% training and 20% testing.  
- **Multiple Models:** Linear Regression, Random Forest, and Support Vector Regressor.  
- **Evaluation Metrics:** Mean Squared Error (MSE) and R² score.  
- **Visualization:** Predicted vs Actual WQI plots for each model.  
- **Best Model Selection:** Automatically selects and saves the best model.  
- **Interactive Predictions:**  
  - Manual input for single sample prediction.  
  - Batch prediction using a CSV file.  
- **Water Quality Advice:** Provides recommendations based on WQI values.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/water-quality-prediction.git
cd water-quality-prediction
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1   # For PowerShell
# OR
.venv\Scripts\activate.bat       # For CMD
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not available, install manually:*

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## Usage

1. Place your dataset CSV file (e.g., `water_quality.csv`) in the project folder.  
2. Run the script:

```bash
python Water_Quality_Management.py
```

3. Follow prompts:

- **Single Prediction:** Enter values for each feature.  
- **Batch Prediction:** Enter the path to a CSV file containing multiple rows of features. The script will generate `batch_predictions.csv` with predicted WQI and advice.

---

## Input CSV Format (for batch predictions)

CSV file should have the **same column names as features used for training**, for example:

| pH  | DO  | BOD | Nitrates | Turbidity |
|-----|-----|-----|----------|-----------|
| 7.2 | 8.1 | 3.5 | 4.0      | 2.1       |
| 6.9 | 7.8 | 4.0 | 3.8      | 2.5       |

---

## Output

- Single sample prediction:

```
Predicted WQI = 78.45
Suggestion: Good water quality. Use basic filtration if needed.
```

- Batch predictions:

```
CSV file 'batch_predictions.csv' with predicted WQI and advice.
```

---

## Visualization

- Predicted vs Actual WQI scatter plots for all models.  
- Helps identify which model performs best.

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgements

- Python Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`.  
- Machine learning techniques for regression and water quality prediction.

