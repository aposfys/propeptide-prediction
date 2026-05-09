# from Bio import SeqIO

# # Load sequences from both FASTA files
# seqs_a = set(str(record.seq) for record in SeqIO.parse("/home/tzermpou/pratskos/DeepPeptide/predictor/uniprotkb_antimicrobial_OR_antifungal_O_2025_06_02.fasta", "fasta"))
# seqs_b = set(str(record.seq) for record in SeqIO.parse("/home/tzermpou/pratskos/DeepPeptide/data/protein_sequences_no_duplicates.fasta", "fasta"))

# # Find intersection
# common_seqs = seqs_a & seqs_b

# print(f"Total sequences in file1: {len(seqs_a)}")
# print(f"Total sequences in file2: {len(seqs_b)}")
# print(f"Sequences from file1 also in file2: {len(common_seqs)}")

from Bio import SeqIO

file1 = "/home/tzermpou/pratskos/DeepPeptide/predictor/uniprotkb_antimicrobial_OR_antifungal_O_2025_06_02.fasta"
file2 = "/home/tzermpou/pratskos/DeepPeptide/data/protein_sequences_no_duplicates.fasta"
output = "unseen_sequences.fasta"

# Load sequences from file2 into a set for quick lookup
seqs_file2 = set(str(record.seq) for record in SeqIO.parse(file2, "fasta"))

# Filter file1: keep only sequences not found in file2
unique_records = [
    record for record in SeqIO.parse(file1, "fasta")
    if str(record.seq) not in seqs_file2
]

# Write unique sequences to a new FASTA file
SeqIO.write(unique_records, output, "fasta")

print(f"Saved {len(unique_records)} sequences unique to {file1} into {output}")

