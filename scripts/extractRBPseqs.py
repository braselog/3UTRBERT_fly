import pandas as pd
import pyranges as pr
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import gzip
import random
import os
import argparse
import yaml

#parse arguments
parser = argparse.ArgumentParser(description='Extract RBP sequences')
parser.add_argument('--params', default='params.yaml', type=str, help='YAML file with parameters')
parser.add_argument('--rbp_file', type=str, help='file with RBP locations')
parser.add_argument('--gtf_file', type=str, help='gtf file with genome annotations')
parser.add_argument('--ref_genome', type=str, help='reference genome file in fasta')
parser.add_argument('--RBP', type=str, help='RBP name')
parser.add_argument('--window_size', type=str, help='size of window to extract')
parser.add_argument('--output_dir', type=str, help='Output directory')
args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['extractRBPseqs'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()


# File paths
os.makedirs(args.output_dir, exist_ok=True)
output_fasta_path = os.path.join(args.output_dir, args.RBP + "_sequences.fasta")
output_random_fasta_path = os.path.join(args.output_dir, args.RBP+'_sequences_random.fasta')

# Read the RBP file into a pandas dataframe
df = pd.read_csv(args.rbp_file, sep='\t', header=None, compression='gzip')

# Read the GTF file into a PyRanges object
gtf = pr.read_gtf(args.gtf_file)

# Filter the GTF file to only include three_prime_utr regions
gtf_three_prime_utr = gtf[gtf.Feature == "three_prime_utr"]

# Convert the df DataFrame to a PyRanges object including column 6
df2 = df.rename(columns={df.columns[0]: "Chromosome", df.columns[1]: "Start", df.columns[2]: "End", df.columns[4]: "Strand", df.columns[5]: "RBP"})
df2 = df2[['Chromosome', 'Start', 'End', 'Strand', 'RBP']]
df_pr = pr.PyRanges(df2)

# Remove 'chr' prefix from the chromosome names in df_pr
df_pr.Chromosome = df_pr.Chromosome.str.replace('chr', '')

# Find overlapping regions between df_pr and the filtered gtf_three_prime_utr
overlaps = df_pr.join(gtf_three_prime_utr)

# Remove duplicate rows
overlaps = overlaps.df.loc[overlaps.df[['Chromosome', 'Start', 'End', 'Strand', 'RBP']].drop_duplicates().index]

# Remove occurrences where the Start and End are not completely inside the UTR ranges
filtered_overlaps = overlaps[(overlaps['Start'] >= overlaps['Start_b']) & (overlaps['End'] <= overlaps['End_b'])]

# Filter for desired RBP
RBP_hits = filtered_overlaps[filtered_overlaps.RBP.str.contains(args.RBP, na=False)]

# Calculate the center, extractStart, and extractEnd for each row in RBP_hits
RBP_hits['Center'] = (RBP_hits['Start'] + RBP_hits['End']) // 2
RBP_hits['extractStart'] = RBP_hits['Center'] - (args.window_size // 2)
RBP_hits['extractEnd'] = RBP_hits['Center'] + (args.window_size // 2)

# Adjust extractStart and extractEnd based on Start_b and End_b
for index, row in RBP_hits.iterrows():
    if row['extractStart'] < row['Start_b']:
        diff = row['Start_b'] - row['extractStart']
        RBP_hits.at[index, 'extractStart'] = row['Start_b']
        RBP_hits.at[index, 'extractEnd'] += diff
    if row['extractEnd'] > row['End_b']:
        diff = row['extractEnd'] - row['End_b']
        RBP_hits.at[index, 'extractEnd'] = row['End_b']
        RBP_hits.at[index, 'extractStart'] -= diff

# Open the reference genome file
reference_genome = SeqIO.to_dict(SeqIO.parse(gzip.open(args.ref_genome, "rt"), "fasta"))

# Create a list to hold the sequences
sequences = []

# Iterate over the rows in RBP_hits
for index, row in RBP_hits.iterrows():
    chromosome = row['Chromosome']
    start = row['extractStart']
    end = row['extractEnd']
    strand = row['Strand']
    
    # Extract the sequence from the reference genome
    sequence = reference_genome[chromosome].seq[start:end]
    
    # Reverse complement the sequence if the strand is negative
    if strand == '-':
        sequence = sequence.reverse_complement()
    
    # Create a SeqRecord object
    record = SeqRecord(Seq(sequence), id="1", description=f"{row['gene_id']}; {chromosome}:{start}-{end}")
    
    # Add the record to the list
    sequences.append(record)

# Write the sequences to a new fasta file
SeqIO.write(sequences, output_fasta_path, "fasta")

# Get the gene_ids that are in filtered_overlaps but not in RBP_hits
gene_ids_not_in_RBP_hits = set(filtered_overlaps['gene_id']) - set(RBP_hits['gene_id'])

# Filter the rows for genes not in RBP_hits
filtered_rows = filtered_overlaps[filtered_overlaps['gene_id'].isin(gene_ids_not_in_RBP_hits)]

# Group by gene_id and select a single row for each gene
single_rows_per_gene = filtered_rows.groupby('gene_id').first().reset_index()

# Create a list to hold the sequences
random_sequences = []

# Iterate over the rows in single_rows_per_gene
for index, row in single_rows_per_gene.iterrows():
    chromosome = row['Chromosome']
    start_b = row['Start_b']
    end_b = row['End_b']
    
    # Ensure the UTR region is at least 100 base pairs long
    if end_b - start_b >= 110:
        # Calculate the number of non-overlapping 100 basepair sequences that can be extracted
        num_sequences = min(3, (end_b - start_b) // 110)
        selected_positions = set()
        
        segment_length = (end_b - start_b) // num_sequences

        for i in range(num_sequences):
            segment_start = start_b + i * segment_length
            segment_end = segment_start + segment_length

            # Ensure the segment is at least 100 base pairs long
            if segment_end - segment_start < 100:
                continue
            
            # Select a random start position within the segment
            random_start = random.randint(segment_start, segment_end - 100)
            random_end = random_start + 100

            # Extract the sequence from the reference genome
            sequence = reference_genome[chromosome].seq[random_start:random_end]

            # Create a SeqRecord object
            record = SeqRecord(Seq(sequence), id="0", description=f"{row['gene_id']}; {chromosome}:{random_start}-{random_end}")

            # Add the record to the list
            random_sequences.append(record)

# Write the random sequences to a new fasta file
SeqIO.write(random_sequences, output_random_fasta_path, "fasta")