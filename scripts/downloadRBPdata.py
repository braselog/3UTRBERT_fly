import subprocess
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='Download RBP data files.')
parser.add_argument('--params', default='params.yaml', type=str, help='YAML file with parameters')
parser.add_argument('--output_dir', type=str, help='Directory to save downloaded files')
args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['downloadRBPdata'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

output_path = os.path.join(args.output_dir, 'fly.txt.gz')
url = 'https://cloud.tsinghua.edu.cn/d/8133e49661e24ef7a915/files/?p=%2Ffly.txt.gz&dl=1'
subprocess.run(['wget', '-O', output_path, url], check=True)

output_path = os.path.join(args.output_dir, 'readme.md')
url = "https://cloud.tsinghua.edu.cn/d/8133e49661e24ef7a915/files/?p=%2FREADME.md&dl=1"
subprocess.run(['wget', '-O', output_path, url], check=True)