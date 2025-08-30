# Water Quality Management Project 💧🌍

## Overview  
This project applies **Artificial Intelligence (AI)** and **Machine Learning (ML)** to monitor and analyze water quality.  
The goal is to evaluate **water quality index (WQI)** using parameters such as pH, turbidity, dissolved oxygen, TDS, and temperature, and prepare the dataset for predictive modeling.  

---

## Dataset  
**Source:** Custom sample dataset (`water_quality.csv`)  

**Features:**  
- pH – Acidity/alkalinity of water  
- Turbidity – Clarity of water  
- DO – Dissolved Oxygen (mg/L)  
- TDS – Total Dissolved Solids (mg/L)  
- Temperature – °C  

**Target:**  
- WQI – Water Quality Index (0–100 scale)  

---

## Tools & Technologies  
- **Python Libraries:** Pandas, Matplotlib, Seaborn  

---

## Project Workflow (Up to Heatmap)  

### 1. Import Libraries  
Load required libraries for data handling and visualization.  

### 2. Load Dataset  
Read `water_quality.csv` into a Pandas DataFrame.  

### 3. Explore Dataset  
- View first few rows (`head()`)  
- Check dataset shape and structure  
- Identify missing values (`isnull().sum()`)  

### 4. Data Cleaning  
- Fill missing values with column means for consistency  

### 5. Visualization  
- **Pair Plot:** To analyze feature distributions and relationships  
- **Correlation Heatmap:** To identify relationships between water parameters  

---

## Visualization Example  

**Pairplot Example:**  
```python
sns.pairplot(df, diag_kind="kde")
plt.show()
```

**Heatmap Example:**  
```python
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation of Water Quality Parameters")
plt.show()
```

These plots help understand how features like pH, DO, and TDS affect WQI.  

---

✅ By the end of this phase (30%), the dataset is **cleaned, visualized, and ready for model training** in the next stage.  
