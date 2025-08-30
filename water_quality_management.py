import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("water_quality.csv")
print("Dataset Shape:", df.shape)
print(df.head())
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
sns.pairplot(df, diag_kind="kde")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation of Water Quality Parameters")
plt.show()
X = df.drop("WQI", axis=1)
y = df["WQI"]
print("✅ Dataset Ready for Model Training")
