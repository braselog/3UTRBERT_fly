from Bio import SeqIO
import os
import random
import csv
import argparse
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from threading import Thread
import gzip

random.seed(42)

# Load NCBI-to-OMA gene ID mappings
def load_ncbi_oma_mapping(file_path):
    ncbi_to_oma = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            oma_id, ncbi_id = line.strip().split()
            ncbi_to_oma[ncbi_id] = oma_id
    return ncbi_to_oma

def load_ncbi_flybase_mapping(file_path):
    flybase_to_ncbi = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            ncbi_flybase_gene = line.strip().split()
            flybase_to_ncbi[ncbi_flybase_gene[1]] = ncbi_flybase_gene[0]
    return flybase_to_ncbi

# Load ortholog groups and assign to splits
def load_and_split_ortholog_groups(file_path):
    ortholog_splits = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            group_id = parts[0]
            oma_ids = parts[2:]

            # Randomly assign each group to a split
            split = random.choices(['train', 'dev', 'test'], weights=[0.7, 0.15, 0.15], k=1)[0]
            for oma_id in oma_ids:
                ortholog_splits[oma_id] = (split, group_id)
    return ortholog_splits

# Convert sequence to k-mers
def seq2kmer(seq, k):
    return " ".join([seq[x:x+k] for x in range(len(seq)+1-k)])

# Process individual fasta file
def process_fasta_file(fasta_file, ortholog_splits, ncbi_to_oma, flybase_to_ncbi, kmer_size, queue):
    species = "[D" + fasta_file.split('/')[-1].replace('-','_').split('_')[1][:4] + "]"
    with open(fasta_file, 'r') as file:
        for seq_record in SeqIO.parse(file, "fasta"):
            ncbi_id = seq_record.description.split()[-1]  # Extract NCBI gene ID
            if species == "[Dmela]":
                ncbi_id = flybase_to_ncbi.get(ncbi_id, None)
            oma_id = ncbi_to_oma.get(ncbi_id, None)

            # Skip if no matching OMA ID found
            if oma_id is None or oma_id not in ortholog_splits:
                continue
            
            # Determine the split for this sequence
            split, group_id = ortholog_splits[oma_id]
            kmer_sequence = seq2kmer(str(seq_record.seq).replace('T','U'), kmer_size)

            # Put the result in the queue
            queue.put((split, kmer_sequence, species, group_id, oma_id, ncbi_id))

# Writer thread function
def writer_thread(output_dir, queue):
    files = {
        'train': open(os.path.join(output_dir, 'train.tsv'), 'a'),
        'dev': open(os.path.join(output_dir, 'dev.tsv'), 'a'),
        'test': open(os.path.join(output_dir, 'test.tsv'), 'a')
    }
    while True:
        item = queue.get()
        if item is None:
            break
        split, kmer_sequence, species, group_id, oma_id, ncbi_id = item
        files[split].write(f"{kmer_sequence}\t{species}\t{group_id}\t{oma_id}\t{ncbi_id}\n")
    for f in files.values():
        f.close()

# Main function for parallel processing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default='params.yaml', help="Path to the YAML file containing parameters.")
    parser.add_argument("--data_dir", help="Directory with species FASTA files.")
    parser.add_argument("--output_dir", help="Directory to save the .tsv files.")
    parser.add_argument("--kmer", type=int, default=4, help="K-mer size.")
    parser.add_argument("--oma_groups", help="Path to oma-groups.txt file.")
    parser.add_argument("--oma_ncbi_map", help="Path to oma-ncbi.txt file.")
    parser.add_argument("--ncbi_flybase_map", help="Path to ncbi-flybase.txt file.")
    args = parser.parse_args()

    # Load parameters
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['preprocessFlyUTRs'].items():
            parser.set_defaults(**{key: value})
    args = parser.parse_args()

    # Load mappings and ortholog groups
    oma_to_ncbi = load_ncbi_oma_mapping(args.oma_ncbi_map)
    ortholog_splits = load_and_split_ortholog_groups(args.oma_groups)
    flybase_to_ncbi = load_ncbi_flybase_mapping(args.ncbi_flybase_map)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize manager for communication between processes and writer thread
    manager = Manager()
    queue = manager.Queue()

    # Start writer thread
    writer = Thread(target=writer_thread, args=(args.output_dir, queue))
    writer.start()

    # Get list of FASTA files
    fasta_files = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir) if file.endswith(".fa")]

    #process_fasta_file('output/data/3utrFlyChunked/Drosophila_obscura-3utr.fa', ortholog_splits, oma_to_ncbi, flybase_to_ncbi, args.kmer, queue)
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_fasta_file, fasta_file, ortholog_splits, oma_to_ncbi, flybase_to_ncbi, args.kmer, queue)
            for fasta_file in fasta_files
        ]
        for future in tqdm(futures, desc="Processing FASTA files"):
            future.result()  # Wait for each file to complete

    # Signal the writer thread to exit
    queue.put(None)
    writer.join()

if __name__ == "__main__":
    main()