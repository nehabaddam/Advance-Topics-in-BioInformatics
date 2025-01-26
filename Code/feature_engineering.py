import pandas as pd
import os
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# Precompute all possible k-mers for k = 2, 3, 4
all_kmers_dict = {k: [''.join(kmer) for kmer in product('ACGT', repeat=k)] for k in range(2, 5)}

from Bio.SeqUtils import MeltingTemp as mt

# Function to calculate melting temperature (Tm)
def compute_melting_temp(sequence):
    return mt.Tm_Wallace(sequence)

# Function to compute nucleotide skew
def compute_gc_skew(sequence):
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    return (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0

# Function to calculate k-mer variability (Diversity Index)
def compute_kmer_diversity(sequence, k):
    kmer_counts = compute_kmer_counts(sequence, k)
    total_kmers = sum(kmer_counts.values())
    unique_kmers = len(kmer_counts)
    return unique_kmers / total_kmers if total_kmers > 0 else 0

# Function to count dinucleotides
def compute_dinucleotide_frequencies(sequence):
    dinucleotides = [''.join(pair) for pair in product('ACGT', repeat=2)]
    dinuc_freq = {f'dinuc_{dn}': 0 for dn in dinucleotides}
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        if dinuc in dinuc_freq:
            dinuc_freq[f'dinuc_{dinuc}'] += 1
    return dinuc_freq


# Define a function to compute k-mer counts with pre-defined k-mers
def compute_kmer_counts(sequence, k):
    kmer_counts = defaultdict(int)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmer_counts[kmer] += 1
    return kmer_counts

# Define a function to calculate GC content
def compute_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)

import math
from collections import Counter

# Function to calculate AT content
def compute_at_content(sequence):
    at_count = sequence.count('A') + sequence.count('T')
    return at_count / len(sequence)

# Function to detect CpG islands
def compute_cpg_islands(sequence):
    cpg_count = sequence.count('CG')
    return cpg_count

# Function to calculate Shannon entropy (sequence complexity)
def compute_shannon_entropy(sequence):
    nucleotide_counts = Counter(sequence)
    total_bases = len(sequence)
    entropy = 0
    for count in nucleotide_counts.values():
        p = count / total_bases
        entropy -= p * math.log2(p)
    return entropy

# Function to calculate codon usage bias
def compute_codon_usage(sequence):
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3) if len(sequence[i:i+3]) == 3]
    codon_counts = Counter(codons)
    total_codons = len(codons)
    codon_usage = {f'codon_{codon}': count / total_codons for codon, count in codon_counts.items()}
    return codon_usage

# Updated function to generate feature dictionary for a sequence
def get_features(sequence):
    features = {}
    
    # Use precomputed k-mers
    for k, all_kmers in all_kmers_dict.items():
        for kmer in all_kmers:
            features[f'{k}-mer_{kmer}'] = 0
        kmer_counts = compute_kmer_counts(sequence, k)
        for kmer, count in kmer_counts.items():
            features[f'{k}-mer_{kmer}'] = count

    # Existing features
    features['gc_content'] = compute_gc_content(sequence)
    features['at_content'] = compute_at_content(sequence)
    features['cpg_islands'] = compute_cpg_islands(sequence)
    features['shannon_entropy'] = compute_shannon_entropy(sequence)
    codon_usage_features = compute_codon_usage(sequence)
    features.update(codon_usage_features)
    
    # Additional features
    features['gc_skew'] = compute_gc_skew(sequence)
    features['melting_temp'] = compute_melting_temp(sequence)
    features['kmer_diversity'] = compute_kmer_diversity(sequence, 3)
    # dinucleotide_freq = compute_dinucleotide_frequencies(sequence)
    # features.update(dinucleotide_freq)
    
    return features


# Define the main execution block
# Define the main execution block
if __name__ == '__main__':
    # Initialize the output files
    output_file_csv = 'sequence_features_val.csv'
    output_file_parquet = 'sequence_features_val.parquet'
    label_file_csv = 'labels_val.csv'
    label_file_parquet = 'labels_val.parquet'
    
    # Initialize lists to hold features and labels
    all_labels = []

    # Process data in chunks
    chunk_size = 100000  # Adjust based on available memory
    header_written_features = False
    header_written_labels = True

    for file, label in [('dataset/pathogenic_val_vir_hum_nano_filtered.csv', 1)
                        ,('dataset/nonpathogenic_val_vir_all_nano_filtered.csv', 0)]:

        # Read the file in chunks
        chunk_iter = pd.read_csv(file, chunksize=chunk_size)

        with ProcessPoolExecutor() as executor:
            for chunk in chunk_iter:
                # Prepare data for parallel processing
                sequences = chunk['Sequence'].tolist()
                # Collect the label for each sequence
                all_labels.extend([label] * len(sequences))  
                print(f"Processing {len(sequences)} sequences from {file}")

                # Compute features in parallel
                feature_list = list(executor.map(get_features, sequences))

                # Convert to DataFrame
                features_df = pd.DataFrame(feature_list)

                # Append features to CSV and Parquet files
                if not features_df.empty:
                    features_df.to_csv(output_file_csv, mode='a', header=not header_written_features, index=False)

                    if os.path.exists(output_file_parquet):
                        existing_df = pd.read_parquet(output_file_parquet)
                        combined_df = pd.concat([existing_df, features_df], ignore_index=True)
                        combined_df.to_parquet(output_file_parquet, index=False, engine='pyarrow')
                    else:
                        features_df.to_parquet(output_file_parquet, index=False, engine='pyarrow')
                    header_written_features = True  # Ensure header is written only once

    # Save labels to DataFrames
    labels_df = pd.DataFrame({'label': all_labels})

    # Save labels to CSV and Parquet
    labels_df.to_csv(label_file_csv, index=False, header=not header_written_labels)  # Automatically writes headers
    labels_df.to_parquet(label_file_parquet, index=False)  # Headers written automatically
    