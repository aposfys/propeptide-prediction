import requests
import pandas as pd
import time

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
QUERY = "(reviewed:true) AND (ft_propep:*) NOT (fragment:true) NOT (taxonomy_id:10239)"
FIELDS = "accession,id,protein_name,organism_name,sequence,length,ft_signal,ft_propep,ft_peptide,ft_chain"

def fetch_all():
    all_rows = []
    next_link = None
    page = 0

    while True:
        params = {"query": QUERY, "format": "tsv", "fields": FIELDS, "size": 500}
        if next_link:
            response = requests.get(next_link)
        else:
            response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text[:200]}")
            break

        lines = response.text.strip().split("\n")
        if page == 0:
            header = lines[0].split("\t")
            all_rows.append(header)
            data_lines = lines[1:]
        else:
            data_lines = lines[1:]

        for line in data_lines:
            all_rows.append(line.split("\t"))

        link_header = response.headers.get("Link", "")
        if 'rel="next"' in link_header:
            next_link = link_header.split(";")[0].strip("<>")
            page += 1
            print(f"Page {page}: {len(all_rows)-1} sequences so far...")
            time.sleep(0.5)
        else:
            break

    df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
    df.to_csv("data/uniprot_propeptides_2026.tsv", sep="\t", index=False)
    print(f"\nDone. Saved {len(df)} sequences to data/uniprot_propeptides_2026.tsv")
    return df

if __name__ == "__main__":
    fetch_all()
