import os
import gzip
import argparse
import yaml
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description="Isolate 3' UTR sequences from cDNA and GTF files.")
    parser.add_argument('--params', default='params.yaml', help='Path to the yaml file with specified arguments.')
    parser.add_argument('--data_dir', default='data/downloaded', help='Path to the directory containing the data files.')
    parser.add_argument('--output_dir', default='data/3UTRs', help='Path to the output directory.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of processes to run concurrently.')

    # Parse initial arguments
    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if os.path.isfile(args.params):
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['isolate3UTRs'].items():
                parser.set_defaults(**{key: value})

    return parser.parse_args()

def reverse_complement(dna_sequence):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    reversed_sequence = dna_sequence[::-1]
    reverse_complement_sequence = ''.join(complement[base] for base in reversed_sequence)
    return reverse_complement_sequence

def read_gtf(gtf_file):
    utr_dict = {}
    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split('\t')
                if parts[2] == 'three_prime_utr':
                    strand = 1 if parts[6] == '+' else -1
                    attributes = parts[8].strip().split(';')
                    gene_id = attributes[0].split('"')[1]
                    transcript_id = attributes[1].split('"')[1]
                    start = int(parts[3])
                    end = int(parts[4])
                    if parts[0] not in utr_dict:
                        utr_dict[parts[0]] = {}
                    if transcript_id not in utr_dict[parts[0]]:
                        utr_dict[parts[0]][(transcript_id, strand, gene_id)] = []
                    utr_dict[parts[0]][(transcript_id, strand, gene_id)].append((start, end))
    return utr_dict

def extract_3utr_sequences(fa_file, utr_dict):
    utr_sequences = {}
    with gzip.open(fa_file, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            name = record.name 
            if name in utr_dict:
                sequence = str(record.seq)
                for tID_strand_gID in utr_dict[name].keys():
                    utr = ''.join([sequence[start - 1:end] for start, end in utr_dict[name][tID_strand_gID]])
                    if 'N' in utr:
                        continue
                    utr_sequences[tID_strand_gID] = utr if tID_strand_gID[1] == 1 else reverse_complement(utr)
    return utr_sequences

def process_files(data_dir, output_dir, files, start_index):
    base_name = files[start_index].split('-')[0]
    utr_dict = read_gtf(os.path.join(data_dir, files[start_index]))
    utr_sequences = extract_3utr_sequences(os.path.join(data_dir, files[start_index + 1]), utr_dict)
    output_file = os.path.join(output_dir, f'{base_name}-3utr.fa')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as out_f:
        for tID_strand_gID, sequence in utr_sequences.items():
            out_f.write(f'>{tID_strand_gID[0]} {tID_strand_gID[2]}\n{sequence}\n')

def main():
    args = parse_args()

    files = os.listdir(args.data_dir)
    files.sort()
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for i in range(0, len(files) - 1, 2):
            #process_files(args.data_dir, args.output_dir, files, i)
            executor.submit(process_files, args.data_dir, args.output_dir, files, i)

if __name__ == '__main__':
    main()