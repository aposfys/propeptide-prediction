import csv
import json

# File paths
tsv_file = "/home/tzermpou/pratskos/DeepPeptide/data/filtered_output.tsv"
json_file = "/home/tzermpou/pratskos/DeepPeptide/predictor/predict_outputs/sequence_outputs.json"
output_file = "predicted_and_real_positions.tsv"

# Load JSON
with open(json_file, "r", encoding="utf-8") as jf:
    json_data = json.load(jf)

# Map Entry ID → (start, end) by searching JSON keys
def get_predicted_propeptide(entry_id):
    for header, info in json_data.get("PREDICTIONS", {}).items():
        if entry_id in header:
            for peptide in info.get("peptides", []):
                if peptide.get("type") == "Propeptide":
                    return peptide["start"], peptide["end"]
    return "", ""

# Read TSV and write output
with open(tsv_file, newline='', encoding="utf-8") as infile, \
     open(output_file, "w", newline='', encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = ["Entry", "Propeptide", "Predicted_Propeptide_Start", "Predicted_Propeptide_End"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for row in reader:
        entry_id = row["Entry"]
        propep = row["Propeptide"]
        start, end = get_predicted_propeptide(entry_id)
        writer.writerow({
            "Entry": entry_id,
            "Propeptide": propep,
            "Predicted_Propeptide_Start": start,
            "Predicted_Propeptide_End": end
        })

print(f"✅ Done. Output saved to: {output_file}")
