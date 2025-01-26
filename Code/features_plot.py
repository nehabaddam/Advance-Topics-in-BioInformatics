import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'feature_importances.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)
data['Feature Length'] = data['Feature'].apply(lambda x: len(x.split('_')[1]))
# Filter features based on their lengths and drop the 'Importance' column
mer_2 = data[data['Feature Length'] == 2].drop(columns=['Feature Length', 'Importance']).head(10)
mer_3 = data[data['Feature Length'] == 3].drop(columns=['Feature Length', 'Importance']).head(10)
mer_4 = data[data['Feature Length'] == 4].drop(columns=['Feature Length', 'Importance']).head(10)

# Create a single DataFrame for the combined table
combined_data = pd.DataFrame({
    '2-mer Features': mer_2['Feature'].values,
    '3-mer Features': mer_3['Feature'].values,
    '4-mer Features': mer_4['Feature'].values
})

# Create a figure to display the table
fig, ax = plt.subplots(figsize=(10, 6))

# Function to create a colorful table
def create_combined_table(ax, data, title):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Smaller font size
    
    # Set color for the table cells
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#1f77b4')  # Blue header
            cell.set_text_props(weight='bold', color='white', fontsize=12)  # White text on header
        else:  # White background for other cells
            cell.set_facecolor('white')
            cell.set_text_props(fontsize=10)  # Adjust font size for cells
    
    ax.set_title(title, fontsize=14, fontname='Times New Roman')

# Create a combined table for 2-mer, 3-mer, and 4-mer features
create_combined_table(ax, combined_data, 'Top 10 Features for 2-mer, 3-mer, and 4-mer')

# Save the figure as an image
plt.tight_layout()
plt.savefig('combined_feature_importance_table.png', bbox_inches='tight')  # Ensure tight layout
plt.show()
