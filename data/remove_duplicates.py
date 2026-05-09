def remove_duplicate_fasta_entries(input_path, output_path):
    seen_ids = set()
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        write_seq = False
        for line in infile:
            if line.startswith('>'):
                seq_id = line[1:].split('|')[0]  # Extract ID before first "|"
                if seq_id in seen_ids:
                    write_seq = False
                else:
                    seen_ids.add(seq_id)
                    outfile.write(line)
                    write_seq = True
            else:
                if write_seq:
                    outfile.write(line)

# Example usage
remove_duplicate_fasta_entries('protein_sequences.fasta', 'protein_sequences_no_duplicates.fasta')
