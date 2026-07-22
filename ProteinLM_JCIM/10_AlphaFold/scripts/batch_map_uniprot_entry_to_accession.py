import time
import requests
import pandas as pd
from pathlib import Path

MAPPING = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_mapping.tsv")
OUT = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_mapping_with_accession.tsv")

df = pd.read_csv(MAPPING, sep="\t")
entry_names = df["entry_name"].dropna().astype(str).unique().tolist()

print("Entry names:", len(entry_names))

RUN_URL = "https://rest.uniprot.org/idmapping/run"
STATUS_URL = "https://rest.uniprot.org/idmapping/status/"
RESULTS_URL = "https://rest.uniprot.org/idmapping/results/"

data = {
    "from": "UniProtKB_AC-ID",
    "to": "UniProtKB",
    "ids": ",".join(entry_names)
}

r = requests.post(RUN_URL, data=data, timeout=60)
r.raise_for_status()

job_id = r.json()["jobId"]
print("Job ID:", job_id)

while True:
    s = requests.get(STATUS_URL + job_id, timeout=60)
    s.raise_for_status()
    js = s.json()

    if "jobStatus" in js:
        print("Status:", js["jobStatus"])
        if js["jobStatus"] in ["NEW", "RUNNING"]:
            time.sleep(3)
            continue
        elif js["jobStatus"] == "FINISHED":
            break
        else:
            raise RuntimeError(js)
    else:
        break

result_url = RESULTS_URL + job_id + "?format=tsv&fields=accession,id,protein_name,gene_names&size=500"

all_rows = []

while result_url:
    res = requests.get(result_url, timeout=60)
    res.raise_for_status()

    lines = res.text.strip().splitlines()
    if len(lines) > 1:
        header = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t")
            row = dict(zip(header, parts))
            all_rows.append(row)

    next_link = None
    if "Link" in res.headers:
        links = res.headers["Link"].split(",")
        for link in links:
            if 'rel="next"' in link:
                next_link = link.split(";")[0].strip()[1:-1]
                break

    result_url = next_link

map_df = pd.DataFrame(all_rows)

print("Mapped rows from UniProt:", len(map_df))
print(map_df.head())

# Expected UniProt TSV columns usually include:
# From, Entry, Entry Name, Protein names, Gene Names

if "From" not in map_df.columns:
    raise RuntimeError("Column 'From' not found in UniProt result.")

accession_col = "Entry" if "Entry" in map_df.columns else None
id_col = "Entry Name" if "Entry Name" in map_df.columns else None

if accession_col is None:
    raise RuntimeError("Accession column not found in UniProt result.")

rename = {
    "From": "entry_name",
    accession_col: "uniprot_accession"
}

if id_col:
    rename[id_col] = "uniprot_id"

map_df = map_df.rename(columns=rename)

keep_cols = ["entry_name", "uniprot_accession"]
if "uniprot_id" in map_df.columns:
    keep_cols.append("uniprot_id")
if "Protein names" in map_df.columns:
    keep_cols.append("Protein names")
if "Gene Names" in map_df.columns:
    keep_cols.append("Gene Names")

map_df = map_df[keep_cols].drop_duplicates("entry_name")

out = df.merge(map_df, on="entry_name", how="left")

out["alphafold_id"] = out["uniprot_accession"].apply(
    lambda x: f"AF-{x}-F1-model_v4" if pd.notna(x) and str(x) != "" else ""
)

out["alphafold_pdb_url"] = out["uniprot_accession"].apply(
    lambda x: f"https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.pdb"
    if pd.notna(x) and str(x) != "" else ""
)

out["alphafold_cif_url"] = out["uniprot_accession"].apply(
    lambda x: f"https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.cif"
    if pd.notna(x) and str(x) != "" else ""
)

out.to_csv(OUT, sep="\t", index=False)

print()
print("Final proteins:", len(out))
print("Mapped accessions:", out["uniprot_accession"].notna().sum())
print("Unmapped:", out["uniprot_accession"].isna().sum())
print("Saved:", OUT)
