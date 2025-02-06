import os
import requests
from bs4 import BeautifulSoup
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

def download_files(species_url, output_dir):
    species_response = requests.get(species_url)
    species_soup = BeautifulSoup(species_response.text, 'html.parser')
    species_link = species_soup.find_all('a')[-1]
    species_href = species_link.get('href')
    if species_href and species_href.endswith('/') and 'GC' in species_href:
        gc_url = f"{species_url}{species_href}"
        gc_response = requests.get(gc_url)
        gc_soup = BeautifulSoup(gc_response.text, 'html.parser')
        for gc_link in gc_soup.find_all('a'):
            gc_href = gc_link.get('href')
            if gc_href and gc_href.endswith('/') and "Parent Directory" not in gc_link.text:
                intermed_url = f"{gc_url}{gc_href}"
                genome_url = f"{intermed_url}genome/"
                refseq_response = requests.get(genome_url)
                refseq_soup = BeautifulSoup(refseq_response.text, 'html.parser')
                for refseq_link in refseq_soup.find_all('a'):
                    file_href = refseq_link.get('href')
                    if file_href.endswith('unmasked.fa.gz'):  # Match the desired file pattern
                        file_url = f"{genome_url}{file_href}"
                        file_name = os.path.join(output_dir, file_href)
                        subprocess.run(['curl', '-o', file_name, file_url])
                        print(f"Downloaded: {file_url}")

                refseq_url = f"{intermed_url}geneset/"
                refseq_response = requests.get(refseq_url)
                refseq_soup = BeautifulSoup(refseq_response.text, 'html.parser')
                refseq_link = refseq_soup.find_all('a')[-1]  # grab the most recent date folder
                refseq_href = refseq_link.get('href')
                if refseq_href and refseq_href.endswith('/'):
                    geneset_url = f"{refseq_url}{refseq_href}"
                    geneset_response = requests.get(geneset_url)
                    geneset_soup = BeautifulSoup(geneset_response.text, 'html.parser')
                    for file_link in geneset_soup.find_all('a'):
                        file_href = file_link.get('href')
                        if file_href.endswith('gtf.gz'):  # Match the desired file pattern
                            file_url = f"{geneset_url}{file_href}"
                            file_name = os.path.join(output_dir, file_href)
                            subprocess.run(['curl', '-o', file_name, file_url])
                            print(f"Downloaded: {file_url}")

def main():
    # Read in the output directory from the command line
    output_dir = sys.argv[1]
    base_output_dir = output_dir.split('/')[0]

    # if the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download files for mapping sequences to ortholog groups
    subprocess.run(['curl', '-o', f"{base_output_dir}/oma-groups.txt.gz", "https://drosoma.unil.ch/All/oma-groups.txt.gz"])
    subprocess.run(['curl', '-o', f"{base_output_dir}/oma-ncbi.txt.gz", "https://drosoma.unil.ch/All/oma-ncbi.txt.gz"])
    subprocess.run(['curl', '-o', f"{output_dir}/Drosophila_melanogaster-GCA_000001215.4-unmasked.fa.gz", "https://ftp.ensembl.org/pub/rapid-release/species/Drosophila_melanogaster/GCA_000001215.4/flybase/genome/Drosophila_melanogaster-GCA_000001215.4-unmasked.fa.gz"])
    subprocess.run(['curl', '-o', f"{output_dir}/Drosophila_melanogaster-GCA_000001215.4-2022_07-genes.gtf.gz", "https://ftp.ensembl.org/pub/rapid-release/species/Drosophila_melanogaster/GCA_000001215.4/flybase/geneset/2022_07/Drosophila_melanogaster-GCA_000001215.4-2022_07-genes.gtf.gz"])

    # # Download files for all Drosophila species
    # # Base URL for Ensembl data
    # base_url = "https://ftp.ensembl.org/pub/rapid-release/species/"
    # response = requests.get(base_url)
    # soup = BeautifulSoup(response.text, 'html.parser')

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = []
    #     for link in soup.find_all('a'):
    #         href = link.get('href')
    #         if href and href.endswith('/') and href.startswith('Drosophila'):  # Only directories
    #             species_url = f"{base_url}{href}"
    #             futures.append(executor.submit(download_files, species_url, output_dir))

    #     for future in futures:
    #         future.result()

if __name__ == "__main__":
    main()