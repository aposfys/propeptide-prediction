import pandas as pd
import re

def parse_propep_field(propep_str):
    """Convert 'PROPEP 19..23; ...; PROPEP 25..31' → '(19-23),(25-31)'"""
    if pd.isna(propep_str) or str(propep_str).strip() == "":
        return ""
    matches = re.findall(r"PROPEP\s+(\d+)\.\.(\d+)", str(propep_str))
    return ",".join(f"({s}-{e})" for s, e in matches) if matches else ""


def parse_signal_field(signal_str):
    if pd.isna(signal_str) or str(signal_str).strip() == "":
        return ""
    matches = re.findall(r"SIGNAL\s+(\d+)\.\.(\d+)", str(signal_str))
    return ",".join(f"({s}-{e})" for s, e in matches) if matches else ""


def parse_peptide_field(peptide_str):
    if pd.isna(peptide_str) or str(peptide_str).strip() == "":
        return ""
    matches = re.findall(r"(?:PEPTIDE|CHAIN)\s+(\d+)\.\.(\d+)", str(peptide_str))
    return ",".join(f"({s}-{e})" for s, e in matches) if matches else ""


def build_propeptide_dataset(input_tsv, output_csv):
    df = pd.read_csv(input_tsv, sep="\t")
    print(f"Loaded {len(df)} sequences from UniProt")
    print(f"Columns available: {list(df.columns)}")

    # UniProt TSV column names (with spaces)
    df_out = pd.DataFrame({
        "protein_id": df["Entry"],
        "protein_name": df["Protein names"],
        "organism": df["Organism"],
        "sequence": df["Sequence"],
        "propeptide_coordinates": df["Propeptide"].apply(parse_propep_field),
        "signal_coordinates": df["Signal peptide"].apply(parse_signal_field),
    })

    # Add peptide/chain if present
    pep_col = "Peptide" if "Peptide" in df.columns else None
    chain_col = "Chain" if "Chain" in df.columns else None
    if pep_col:
        df_out["peptide_coordinates"] = df[pep_col].apply(parse_peptide_field)
    if chain_col and not pep_col:
        df_out["peptide_coordinates"] = df[chain_col].apply(parse_peptide_field)

    # Filter: must have at least one propeptide
    before = len(df_out)
    df_out = df_out[df_out["propeptide_coordinates"] != ""].reset_index(drop=True)
    print(f"After requiring propeptide annotation: {len(df_out)} (dropped {before - len(df_out)})")

    # Filter: drop oversized sequences (memory limit for ESM3)
    df_out["length"] = df_out["sequence"].str.len()
    before = len(df_out)
    df_out = df_out[df_out["length"] <= 1500].reset_index(drop=True)
    print(f"After length <= 1500: {len(df_out)} (dropped {before - len(df_out)})")

    # Filter: drop sequences with non-standard amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    before = len(df_out)
    df_out = df_out[df_out["sequence"].apply(lambda s: set(s).issubset(valid_aa))].reset_index(drop=True)
    print(f"After valid AA only: {len(df_out)} (dropped {before - len(df_out)})")

    print(f"\nFinal stats:")
    print(f"  Total sequences: {len(df_out)}")
    print(f"  Median length: {df_out['length'].median():.0f}")
    print(f"  Max length: {df_out['length'].max()}")
    print(f"  Mean length: {df_out['length'].mean():.1f}")

    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    build_propeptide_dataset(
        "data/uniprot_propeptides_2026.tsv",
        "data/labeled_sequences_uniprot_2026.csv"
    )
