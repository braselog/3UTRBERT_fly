from Bio import SeqIO
import argparse
import yaml
import os

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--params",
    default=None,
    type=str,
    required=True,
    help="Path to the YAML file containing parameters.",
)
parser.add_argument(
    "--input_dir",
    default=None,
    type=str,
    help="The input directory containing fasta files. Should contain the sequences for the task.",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    help="Path where the output fasta files should be saved.",
)
parser.add_argument(
    "--chunk_size",
    default=None,
    type=int,
    help="The chunk size for processing the data.",
)

# Parse initial arguments
args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['chunkData'].items():
            parser.set_defaults(**{key: value})

# Parse arguments again to apply defaults from YAML
args = parser.parse_args()

# make sure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

unique_sequences = {}

for file in os.listdir(args.input_dir):
    input_path = os.path.join(args.input_dir, file)
    output_path = os.path.join(args.output_dir, file)
    with open(output_path, "w") as out_handle:
        for record in SeqIO.parse(input_path, "fasta"):
            chunks = [record.seq[i:i + args.chunk_size] for i in range(0, len(record.seq), args.chunk_size)]
            for i, chunk in enumerate(chunks):
                seq_str = str(chunk)
                if len(seq_str) >= 10 and seq_str not in unique_sequences:
                    unique_sequences[seq_str] = True
                    chunk_id = f"{record.id}_chunk{i+1}"
                    # Extract description without the ID
                    description = record.description.split(' ', 1)[1] if ' ' in record.description else ''
                    chunk_record = SeqIO.SeqRecord(chunk, id=chunk_id, description=description)
                    SeqIO.write(chunk_record, out_handle, "fasta-2line")