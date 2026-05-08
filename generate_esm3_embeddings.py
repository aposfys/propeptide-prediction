import os
import pandas as pd
import torch
from tqdm import tqdm
import hashlib
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

# Configuration
CSV_PATH = "data/labeled_sequences_esm3.csv" 
OUTPUT_DIR = "data/esm3_embeddings"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_hash(seq):
    # Clean the sequence exactly how the DataLoader cleans it
    clean_seq = str(seq).strip().replace('\n', '').replace(' ', '')
    return hashlib.md5(clean_seq.encode()).hexdigest()

print(f"Loading ESM3 on {DEVICE}...")
model = ESM3.from_pretrained("esm3_sm_open_v1").to(DEVICE)
model.eval()

df = pd.read_csv(CSV_PATH)
print(f"Processing {len(df)} sequences...")

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['sequence']
        seq_hash = get_hash(seq)
        output_file = os.path.join(OUTPUT_DIR, f"{seq_hash}.pt")
        
        if os.path.exists(output_file):
            continue
            
        try:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            
            embeddings = output.embeddings
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)
            
            # SLICE: Remove BOS/EOS 
            final_tensor = embeddings[1:-1, :].float().cpu()
            
            # --- QUALITY CHECK ---
            if len(seq) != final_tensor.shape[0]:
                print(f"Length Mismatch in {seq_hash}: Seq({len(seq)}) vs Emb({final_tensor.shape[0]})")
                continue

            torch.save(final_tensor, output_file)

            # Optional: Clear Mac memory cache
            if DEVICE == "mps":
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"Error processing sequence: {e}")

print(f"Done! Embeddings saved to {OUTPUT_DIR}")