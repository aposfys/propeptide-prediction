import pandas as pd
import re

# Load TSV
df = pd.read_csv('filtered_output.tsv', sep='\t')

# Function to extract start and end from the 'Propeptide' column
def extract_positions(propeptide_str):
    match = re.search(r'(\d+)\.\.(\d+)', str(propeptide_str))
    if match:
        return match.group(1), match.group(2)
    return None, None

# Apply the extraction
df['Propeptide_Start'], df['Propeptide_End'] = zip(*df['Propeptide'].apply(extract_positions))

# Drop rows where positions couldn't be extracted
df_cleaned = df.dropna(subset=['Propeptide_Start', 'Propeptide_End'])

# Keep only desired columns
df_final = df_cleaned[['Entry', 'Propeptide_Start', 'Propeptide_End']]

# Save to a new TSV file
df_final.to_csv('propeptides_cleaned.tsv', sep='\t', index=False)
