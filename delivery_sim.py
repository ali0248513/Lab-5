import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
# Create the dataset: 100 synthetic deliveries (with 3 very slow ones)
times = np.round(np.random.normal(loc=32, scale=8, size=100)).astype(int)
times = np.clip(times, 10, None)
times[[10, 45, 80]] = [68, 71, 75]

pd.DataFrame({
    "OrderID": range(1, 101),
    "DeliveryTime(min)": times
}).to_csv("delivery_times.csv", index=False)

df = pd.read_csv("delivery_times.csv")
real = df["DeliveryTime(min)"]
# (a) Mean and standard deviation of real data
mean = real.mean()
std = real.std()
print(f"Real delivery times -- Mean: {mean:.1f} min Std: {std:.1f} min")
# (b) Simulate 100 values from Normal(mean, std)
simulated = np.random.normal(loc=mean, scale=std, size=100)
print(f"Simulated 100 values from Normal(mean={mean:.1f}, std={std:.1f})")
# (c) Overlapping histograms
bins = np.linspace(min(real.min(), simulated.min()),
                   max(real.max(), simulated.max()), 16)
plt.figure(figsize=(7, 4))
plt.hist(real, bins=bins, alpha=0.6, color="steelblue", label="Real")
plt.hist(simulated, bins=bins, alpha=0.6, color="indianred", label="Simulated")
plt.xlabel("Delivery Time (min)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
# (d) EDA: missing values, boxplot, percentiles
print("Missing values:", real.isnull().sum())

plt.figure(figsize=(4, 4))
plt.boxplot(real)
plt.title("Delivery Time Boxplot")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")

q1 = real.quantile(0.25)
q2 = real.quantile(0.50)
q3 = real.quantile(0.75)
print(f"Percentiles -- 25th: {q1} 50th: {q2} 75th: {q3}")
# (e) IQR outliers
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f"IQR outlier bounds: {lower:.1f} - {upper:.1f}")

outliers = real[(real < lower) | (real > upper)]
print(f"Outliers detected: {len(outliers)} deliveries ({outliers.tolist()} minutes)")
print(f"Percentage flagged as outliers: {len(outliers) / len(real) * 100:.1f}%")

print("Saved delivery_comparison_histogram.png, delivery_boxplot.png")