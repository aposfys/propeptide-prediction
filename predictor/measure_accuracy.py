# import json
# import csv

# # Load JSON data
# with open('peptide_predictions_from_testrun.json') as f:
#     data = json.load(f)

# # Create a dictionary from JSON PREDICTIONS keyed by the entry name without the '>' prefix
# predictions = {k.lstrip('>'): v for k, v in data["PREDICTIONS"].items()}

# # Parse CSV file and extract relevant info into a list of tuples (sseqid, propeptide_range)
# csv_entries = []
# with open('Precursors.csv') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         sseqid = row['sseqid']  # The ID, e.g., XP_030039163.2
#         propeptide_pos = row['Pro_Domain_Position']  # The last column with positions e.g. "42-483"
        
#         # Check if propeptide_pos includes a '-' and is non-empty
#         if '-' in propeptide_pos and propeptide_pos.strip():
#             start_str, end_str = propeptide_pos.split('-')
#             try:
#                 start = int(start_str)
#                 end = int(end_str)
#                 csv_entries.append((sseqid, start, end))
#             except ValueError:
#                 # Ignore malformed propeptide positions if necessary
#                 pass

# # Count how many JSON entries match CSV entries on Propeptide positions
# count = 0

# for sseqid, start_csv, end_csv in csv_entries:
#     if sseqid in predictions:
#         peptides = predictions[sseqid].get('peptides', [])
#         # Find peptides of type "Propeptide" with matching start and end
#         for peptide in peptides:
#             if peptide.get('type') == 'Propeptide':
#                 start_json = peptide.get('start')
#                 end_json = peptide.get('end')
#                 if start_json == start_csv and end_json == end_csv:
#                     count += 1
#                     break  # Only count once per sseqid

# print(f"Number of JSON entries with matching Propeptide positions found in CSV: {count}")
import json
import csv

with open('peptide_predictions_from_testrun.json') as f:
    data = json.load(f)

predictions = {k.lstrip('>'): v for k, v in data["PREDICTIONS"].items()}

csv_entries = []
with open('Precursors.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sseqid = row['sseqid']
        propeptide_pos = row['Pro_Domain_Position']
        if '-' in propeptide_pos and propeptide_pos.strip():
            start_str, end_str = propeptide_pos.split('-')
            try:
                start = int(start_str)
                end = int(end_str)
                csv_entries.append((sseqid, start, end))
            except ValueError:
                pass

count = 0
print_limit = 10  # max number of printed entries
printed = 0
seq_len=0
num_entries=0
count_one=0
valid_count_one=0
for sseqid, start_csv, end_csv in csv_entries:
    if ((end_csv-start_csv>=5) and (end_csv-start_csv<=50)):
        if sseqid in predictions:
            peptides = predictions[sseqid].get('peptides', [])
            for peptide in peptides:
                if peptide.get('type') == 'Propeptide':
                    start_json = peptide.get('start')
                    end_json = peptide.get('end')
                    if (start_json >= start_csv-1 and start_json <= start_csv+1) and (end_json >= end_csv-1 and end_json <= end_csv+1):
                        count += 1
                        
                        if printed < print_limit:
                            print(f"Match {printed+1}: sseqid={sseqid}, start={start_json}, end={end_json}, type={peptide.get('type')}")
                            printed += 1
                        break
                    else:
                    # print(f"Match {printed+1}: sseqid={sseqid}, start={start_json}, end={end_json}, type={peptide.get('type')}")
                        seq_len+=(end_csv-start_csv)
                        num_entries+=1
    else:
        count_one+=1
mean=seq_len/num_entries
#print(f"\nMean seq len {mean}")
print(f"\nCount of AminoAcids with Propeptide of length >=5 and <=50: {count_one}")
print(f"\nTotal matching entries found: {count}")
