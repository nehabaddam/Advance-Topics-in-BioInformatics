import pandas as pd
import matplotlib.pyplot as plt

# Load the feature importance data from the CSV file
file_path = 'feature_importances_ML_DL.csv'  # Adjust the path if necessary
feature_importances = pd.read_csv(file_path)

# Sort the features by importance and select the top 20
top_features = feature_importances.sort_values(by='Importance', ascending=False).head(20)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance', fontsize=14)
plt.title('Top 20 Features Importance', fontsize=16)
plt.gca().invert_yaxis()  # Invert y axis to show the highest importance on top
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the figure
output_file_path = 'feature_importance.png'  # Adjust the path if necessary
plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data from the models
# Data from the models
models = ['Random Forest', 'XGBoost', 'Neural Network', 'CNN']
accuracy = [0.8949, 0.8453, 0.8672, 0.9152]
precision = [0.9264, 0.8568, 0.8666, 0.9191]
recall = [0.8579, 0.8290, 0.8680, 0.9105]
f1_score = [0.8908, 0.8427, 0.8673, 0.9148]


# Set up bar positions
x = np.arange(len(models))

# Width of the bars
width = 0.2

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='#1f77b4')
bars2 = ax.bar(x - 0.5*width, recall, width, label='Recall', color='#6baed6')
bars3 = ax.bar(x + 0.5*width, f1_score, width, label='F1 Score', color='#dbeefb')
bars4 = ax.bar(x + 1.5*width, accuracy, width, label='Accuracy', color='#3182bd')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)

# Position the legend at the bottom
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)

# Adding value labels on the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

# Show the plot
plt.tight_layout()

output_file_path = 'final_results.png'  # Adjust the path if necessary
plt.savefig(output_file_path, bbox_inches='tight', dpi=300)

plt.show()
k=1


import matplotlib.pyplot as plt
import numpy as np

# Data for the plot


metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [0.9444,0.9451050642259364,0.9444,0.9443779731211358]

# Creating the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Bar plot with shades of blue
colors = plt.cm.Blues(np.linspace(0.4, 1, len(values)))
bars = ax.bar(metrics, values, color=colors)

# Adding labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Model Performance: GCN')

# Display the values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

# Display the plot
plt.show()

k=1