import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
# Roll a fair six-sided die 1000 times
rolls = np.random.randint(1, 7, size=1000)
# (a) First 10 rolls
print("First 10 rolls:", rolls[:10])
# (b) Count of each face
faces, counts = np.unique(rolls, return_counts=True)
# (c) Empirical vs theoretical probability
print("Face Count Empirical P Theoretical P")
for face, count in zip(faces, counts):
    print(f"{face:<5}{count:<6}{count / 1000:<12.3f}{1 / 6:.4f}")
# (d) Histogram of the rolls
plt.figure(figsize=(6, 4))
plt.hist(rolls, bins=np.arange(0.5, 7.5, 1), rwidth=0.8,
         color="steelblue", edgecolor="black")
plt.xticks(range(1, 7))
plt.xlabel("Die Face")
plt.ylabel("Frequency")
plt.title("1000 Rolls of a Fair Die")
plt.tight_layout()
plt.savefig("die_rolls.png")
print("Saved die_rolls.png")