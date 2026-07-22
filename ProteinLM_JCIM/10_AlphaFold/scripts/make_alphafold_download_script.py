import pandas as pd
from pathlib import Path

MAPPING = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_mapping_with_accession.tsv")
OUTDIR = Path("ProteinLM_JCIM/10_AlphaFold/01_pdb")
SCRIPT = Path("ProteinLM_JCIM/10_AlphaFold/scripts/download_alphafold_pdb.sh")
LIST = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_download_list.tsv")

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MAPPING, sep="\t")

rows = []
lines = [
    "#!/bin/bash",
    "set -e",
    "",
    "cd ~/yeast_project/Paper2_ML",
    "mkdir -p ProteinLM_JCIM/10_AlphaFold/01_pdb",
    ""
]

for _, r in df.iterrows():
    gene = r["gene"]
    acc = r["uniprot_accession"]
    url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
    outfile = f"ProteinLM_JCIM/10_AlphaFold/01_pdb/{gene}__{acc}.pdb"

    rows.append({
        "gene": gene,
        "entry_name": r["entry_name"],
        "uniprot_accession": acc,
        "alphafold_pdb_url": url,
        "local_pdb": outfile
    })

    lines.append(f'curl -L --fail "{url}" -o "{outfile}" || echo "FAILED {gene} {acc}" >> ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_download_failed.txt')

pd.DataFrame(rows).to_csv(LIST, sep="\t", index=False)

with open(SCRIPT, "w") as f:
    f.write("\n".join(lines) + "\n")

print("Proteins:", len(df))
print("Saved download list:", LIST)
print("Saved shell script:", SCRIPT)
print()
print("First 5:")
print(pd.DataFrame(rows).head())
