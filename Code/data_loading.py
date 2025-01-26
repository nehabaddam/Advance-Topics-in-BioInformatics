import gzip
import os
import pandas as pd
from Bio import SeqIO

# Path to your dataset directory containing .fasta.gz files
directory_path = "dataset/"

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".fasta.gz"):
        file_path = os.path.join(directory_path, filename)
        
        # Initialize a list to store data for the current file
        data = []
        
        # Read the .fasta.gz file
        with gzip.open(file_path, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                data.append({"ID": record.id, "Sequence": str(record.seq)})
        
        # Create a DataFrame from the list
        df = pd.DataFrame(data)
        
        # Create an output CSV file name based on the original .fasta.gz file name
        output_csv_name = filename.replace(".fasta.gz", ".csv")
        output_csv_path = os.path.join(directory_path, output_csv_name)
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_csv_path, index=False)
        
        print(f"Data from {filename} saved to {output_csv_name}")
