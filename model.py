
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel("C:\\zUsers\\HP\\Downloads\\student_performance_project\\data.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe())


corr = df.corr(numeric_only=True)
print("\n===== CORRELATION MATRIX =====")
print(corr)

print("\n===== CORRELATION WITH final_grade =====")
print(corr["final_grade"])


plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(6,4))
plt.hist(df["final_grade"], bins=10)
plt.title("Histogram of Final Grade")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(6,4))
df["final_grade"].value_counts().sort_index().plot(kind='bar')
plt.title("Bar Graph of Final Grade")
plt.xlabel("Final Grade")
plt.ylabel("Count")
plt.show()


features = ["attendance", "assignment_score", "midterm_score"]

for col in features:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df["final_grade"])
    plt.title(f"{col} vs Final Grade")
    plt.xlabel(col)
    plt.ylabel("Final Grade")
    plt.show()
