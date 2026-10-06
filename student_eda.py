import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Create the dataset (12 rows: 2 missing Marks, 1 exact duplicate row)
data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female",
               "Male", "Female", "Male", "Female", "Male", "Male"],
    "StudyHours": [5, 8, 3, 7, 6, 9, 2, 6, 4, 7, 5, 5],
    "Attendance(%)": [80, 92, 70, 88, 82, 95, 65, 85, 75, 90, 78, 78],
    "Marks": [68, 85, 52, 78, np.nan, 90, 45, 74, np.nan, 80, 66, 66]
}
pd.DataFrame(data).to_csv("student_performance.csv", index=False)
# (a) Load the CSV, show shape and info
df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
df.info()
# (b) Missing values and duplicates, then clean
print("Missing values per column:")
print(df.isnull().sum())
print("Missing values in Marks:", df["Marks"].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
df["Marks"] = df["Marks"].fillna(df["Marks"].median())
print("After cleaning:", df.shape)
# (c) Histogram and boxplot of Marks
plt.figure(figsize=(6, 4))
plt.hist(df["Marks"], bins=6, color="steelblue", edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Marks Distribution")
plt.tight_layout()
plt.savefig("marks_histogram.png")

plt.figure(figsize=(4, 4))
plt.boxplot(df["Marks"])
plt.title("Marks Boxplot")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
# (d) Scatter plot and correlation matrix
plt.figure(figsize=(6, 4))
plt.scatter(df["StudyHours"], df["Marks"], color="darkorange")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")

print("Correlation matrix:")
print(df[["StudyHours", "Attendance(%)", "Marks"]].corr())
# (e) Average Marks and Attendance by Gender
print("Average Marks and Attendance by Gender:")
print(df.groupby("Gender")[["Marks", "Attendance(%)"]].mean().round(1))

print("Saved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")