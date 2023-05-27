from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")

def exponential_smoothing(x, alpha=0.6):
    smoothed = [x[0]]

    for i in range(1, len(x)):
        smoothed_val = (1 - alpha) * x[i] + alpha * smoothed[i-1]
        smoothed.append(smoothed_val)

    return np.array(smoothed)

for category in ["pathmnist", "dermamnist", "bloodmnist"]:
    fig, ax = plt.subplots()
    colors = ["#e64980", "#4c6ef5", "#74b816"]

    for i, depth in enumerate([18, 34, 50]):
        data = pd.read_csv(f"data/resnet-choice/val-acc/baseline-{category}-{depth}_version_0.csv")

        steps = data["Step"].values
        values = data["Value"].values
        
        smoothed_values = exponential_smoothing(values)

        # Smoothed values
        ax.plot(steps, smoothed_values, color=colors[i], label=f"resnet-{depth}")
        # Unsmoothed values
        ax.plot(steps, values, color=colors[i], alpha=0.25)
    
    ax.set_xlabel("steps")
    ax.set_ylabel("validation accuracy")
    ax.legend()

    # Update ticks
    ax.set_xticklabels([f"{int(tick/1000)}k" for tick in ax.get_xticks()])

    plt.show()
