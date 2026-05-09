from Bio import SeqIO
import csv

# File paths
fasta_file = "/home/tzermpou/pratskos/DeepPeptide/data/unseen_sequences.fasta"
tsv_file = "/home/tzermpou/pratskos/DeepPeptide/data/uniprotkb_antimicrobial_OR_antifungal_A_2025_06_03.tsv"
output_file = "filtered_output.tsv"

# Step 1: Extract UniProt accessions from FASTA
fasta_accessions = set()

for record in SeqIO.parse(fasta_file, "fasta"):
    # FASTA headers like: >sp|A0A097PTA8|DEFCO_COPCI
    parts = record.id.split("|")
    if len(parts) >= 2:
        fasta_accessions.add(parts[1])  # e.g., A0A097PTA8

# Step 2: Filter TSV based on these accessions
with open(tsv_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline='', encoding="utf-8") as outfile:

    reader = csv.reader(infile, delimiter="\t")
    writer = csv.writer(outfile, delimiter="\t")

    # Write header
    header = next(reader)
    writer.writerow(header)

    # Write matching rows
    for row in reader:
        entry_id = row[0]  # First column is the UniProt accession
        if entry_id in fasta_accessions:
            writer.writerow(row)

print(f"Filtered TSV written to {output_file}")
