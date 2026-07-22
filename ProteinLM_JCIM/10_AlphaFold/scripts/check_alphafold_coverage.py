import pandas as pd
import re
from pathlib import Path

PROTEIN_TABLE = Path("ProteinLM_JCIM/00_data/protein_master_table.tsv")
OUTDIR = Path("ProteinLM_JCIM/10_AlphaFold/04_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "alphafold_mapping.tsv"

df = pd.read_csv(PROTEIN_TABLE, sep="\t")

rows = []

for _, r in df.iterrows():
    gene = r["gene"]
    entry_name = str(r["entry_name"])

    # UniProt entry name usually looks like SEC61_YEAST.
    # AlphaFold DB file usually requires UniProt accession,
    # so this first script only prepares mapping candidates.
    organism_ok = entry_name.endswith("_YEAST")

    rows.append({
        "gene": gene,
        "entry_name": entry_name,
        "protein_name": r["protein_name"],
        "organism_entry_is_yeast": organism_ok,
        "alphafold_status": "needs_uniprot_accession",
        "notes": "Entry name available; UniProt accession still required for AlphaFold DB download."
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)

print("Proteins:", len(out))
print("Yeast UniProt-style entry names:", int(out["organism_entry_is_yeast"].sum()))
print("Need UniProt accession:", len(out))
print("Saved:", OUT)

print()
print(out.head())
