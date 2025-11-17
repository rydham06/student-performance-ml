
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_excel("C:\\zUsers\\HP\\Downloads\\student_performance_project\\data.excel")

print("First 5 rows of data:")
print(df.head())

print(" Summary Statistics:")
print(df.describe())

df.hist(figsize=(10,6))
plt.suptitle("Histograms of Features", fontsize=14)
plt.show()


plt.figure(figsize=(7,5))
plt.bar(df.index, df["final_grade"])
plt.xlabel("Student Index")
plt.ylabel("Final Grade")
plt.title("Bar Plot – Final Grade")
plt.show()

cols = ["attendance", "assignment_score", "midterm_score"]

for col in cols:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df["final_grade"])
    plt.xlabel(col)
    plt.ylabel("final_grade")
    plt.title(f"Scatter Plot: {col} vs final_grade")
    plt.show()


print(" Correlation with final_grade:")
print(df.corr()["final_grade"].sort_values(ascending=False))




X = df[["attendance", "assignment_score", "midterm_score"]]
y = df["final_grade"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\ Linear Regression Results:")
print("R2 Score:", r2_score(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print(" Random Forest Results:")
print("R2 Score:", r2_score(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

