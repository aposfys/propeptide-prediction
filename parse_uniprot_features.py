# scripts/parse_uniprot_features.py
import pandas as pd
import re

def parse_propep_field(propep_str: str) -> str:
    """Convert UniProt PROPEP feature string to (start-end),(start-end) format."""
    if pd.isna(propep_str) or propep_str == "":
        return ""

    # Match patterns like "PROPEP 19..23" or "PROPEP 1..14"
    matches = re.findall(r"PROPEP\s+(\d+)\.\.(\d+)", str(propep_str))
    if not matches:
        return ""
    return ",".join(f"({s}-{e})" for s, e in matches)


def parse_signal_field(signal_str: str) -> str:
    """Same logic for signal peptides — needed to mask them as State 0."""
    if pd.isna(signal_str) or signal_str == "":
        return ""
    matches = re.findall(r"SIGNAL\s+(\d+)\.\.(\d+)", str(signal_str))
    if not matches:
        return ""
    return ",".join(f"({s}-{e})" for s, e in matches)


def parse_peptide_field(peptide_str: str) -> str:
    """Mature peptides — also masked to State 0 in propeptides_only mode."""
    if pd.isna(peptide_str) or peptide_str == "":
        return ""
    matches = re.findall(r"(?:PEPTIDE|CHAIN)\s+(\d+)\.\.(\d+)", str(peptide_str))
    if not matches:
        return ""
    return ",".join(f"({s}-{e})" for s, e in matches)


def build_propeptide_dataset(input_tsv: str, output_csv: str):
    df = pd.read_csv(input_tsv, sep="\t")

    print(f"Loaded {len(df)} sequences from UniProt")

    # Required columns (rename to match your existing CSV format)
    df_out = pd.DataFrame({
        "protein_id": df["Entry"],
        "protein_name": df["Protein names"],
        "organism": df["Organism"],
        "sequence": df["Sequence"],
        "propeptide_coordinates": df["Propeptide"].apply(parse_propep_field),
        "signal_coordinates": df["Signal peptide"].apply(parse_signal_field),
        "peptide_coordinates": df.get("Peptide", df.get("Chain", pd.Series([""]*len(df)))).apply(parse_peptide_field),
    })

    # Filter: keep only sequences with at least one propeptide annotation
    df_out = df_out[df_out["propeptide_coordinates"] != ""].reset_index(drop=True)

    # Filter: drop sequences longer than 1500 residues (memory limit for ESM3)
    df_out["length"] = df_out["sequence"].str.len()
    df_out = df_out[df_out["length"] <= 1500].reset_index(drop=True)

    # Filter: drop sequences with non-standard amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    df_out["clean"] = df_out["sequence"].apply(lambda s: set(s).issubset(valid_aa))
    df_out = df_out[df_out["clean"]].drop(columns=["clean"]).reset_index(drop=True)

    print(f"After filtering: {len(df_out)} sequences with valid propeptide annotations")
    print(f"Median length: {df_out['length'].median():.0f}, max: {df_out['length'].max()}")

    df_out.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    build_propeptide_dataset(
        "data/uniprot_propeptides_2026.tsv",
        "data/labeled_sequences_uniprot_2026.csv"
    )