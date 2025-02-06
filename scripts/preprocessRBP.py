# extract 20% independent_test_set, do 5-fold split in remaining data
from Bio import SeqIO
import os
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
import argparse
import yaml
import csv
import numpy as np


def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def write_to_tsv(file_path, data, header=['sequence', 'label']):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        if not file_exists:
            tsv_w.writerow(header)
        for row in data:
            tsv_w.writerow(row)

def extract_test_and_split_train_vali(input_path, save_path, kmer, percentKeep=100):
    #create directory if not exist
    os.makedirs(save_path, exist_ok=True)
    # remove preexisting files in output directory
    for file in os.listdir(save_path):
        os.remove(os.path.join(save_path, file))
    
    files = os.listdir(input_path)
    seq_list = []

    for file in files:
        if file.endswith('.fasta'):
            input_file = os.path.join(input_path, file)
            seq_list = []
            for seq_record in tqdm(SeqIO.parse(input_file, "fasta")):
                seq_record.seq = seq_record.seq.upper()
                seq_list.append(seq_record)
            random.shuffle(seq_list)

            length = len(seq_list)
            test_len = length // 5
            independent_test_set = seq_list[:test_len]
            kmer_to_tsv_test = []
            sequences = []
            for seq_record in independent_test_set:
                final_kmer = seq2kmer(str(seq_record.seq), kmer)
                label = seq_record.id
                kmer_to_tsv_test.append([final_kmer.replace('T',"U"), label])
                sequences.append(seq_record)

            write_to_tsv(os.path.join(save_path, "test.tsv"), kmer_to_tsv_test)
            with open(os.path.join(save_path, "test.fasta"), 'a') as output_handle:
                SeqIO.write(sequences, output_handle, "fasta-2line")

            remain_data = seq_list[test_len:]
            dev_len = len(remain_data) // 5
            dev_set = remain_data[:dev_len]
            train_set = remain_data[dev_len:]

            kmer_to_tsv_train = []
            sequences = []
            for seq_record in train_set:
                final_kmer = seq2kmer(str(seq_record.seq), kmer)
                label = seq_record.id
                kmer_to_tsv_train.append([final_kmer.replace('T', "U"), label])
                sequences.append(seq_record)

            indices = random.sample(range(len(kmer_to_tsv_train)), int(np.floor(len(kmer_to_tsv_train) / 100 * percentKeep)))
            kmer_to_tsv_train = [kmer_to_tsv_train[i] for i in indices]
            sequences = [sequences[i] for i in indices]
            write_to_tsv(os.path.join(save_path, "train.tsv"), kmer_to_tsv_train)
            #SeqIO.write(sequences, os.path.join(save_path, "train.fasta"), "fasta") # have both the train and dev data in the same file for training the svm classifier.
            if 'random' in file:
                SeqIO.write(sequences, os.path.join(save_path, "train_neg.fasta"), "fasta-2line")
            else:
                SeqIO.write(sequences, os.path.join(save_path, "train_pos.fasta"), "fasta-2line")

            kmer_to_tsv_dev = []
            devSeqs = []
            for seq_record in dev_set:
                final_kmer = seq2kmer(str(seq_record.seq), kmer)
                label = seq_record.id
                kmer_to_tsv_dev.append([final_kmer.replace('T',"U"), label])
                devSeqs.append(seq_record)

            indices = random.sample(range(len(kmer_to_tsv_dev)), int(np.floor(len(kmer_to_tsv_dev) / 100 * percentKeep)))
            kmer_to_tsv_dev = [kmer_to_tsv_dev[i] for i in indices]
            devSeqs = [devSeqs[i] for i in indices]
            write_to_tsv(os.path.join(save_path, "dev.tsv"), kmer_to_tsv_dev)
            if 'random' in file:
                SeqIO.write(devSeqs, os.path.join(save_path, "dev_neg.fasta"), "fasta-2line")
                SeqIO.write(sequences+devSeqs, os.path.join(save_path, "train_dev_neg.fasta"), "fasta-2line")
            else:
                SeqIO.write(devSeqs, os.path.join(save_path, "dev_pos.fasta"), "fasta-2line")
                SeqIO.write(sequences+devSeqs, os.path.join(save_path, "train_dev_pos.fasta"), "fasta-2line")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--params", default=None, type=str, required=True, help="Path to the YAML file containing parameters.",)
    parser.add_argument( "--data_dir", default=None, type=str, help="The input fasta data dir. Should contain the sequences for the task.",)
    parser.add_argument( "--output_dir", default=None, type=str, help="Path where the processed .tsv files are saved.",)
    parser.add_argument( "--kmer", default=3, type=int, help="The kmer used by the model",)
    parser.add_argument( "--percentTrainDataKeep", default=100, type=int, help="The kmer used by the model",)
    parser.add_argument( "--splitSeed", default=42, type=int, help="Random seed for test, val, train split.",)

    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['preprocessRBP'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    random.seed(args.splitSeed)
    extract_test_and_split_train_vali(args.data_dir, args.output_dir, args.kmer, args.percentTrainDataKeep)

if __name__ == "__main__":
    main()
