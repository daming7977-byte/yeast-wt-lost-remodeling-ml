import pandas as pd
from pathlib import Path

MASTER = Path("master_feature_table_v2.tsv")
AFMAP = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_mapping_with_accession.tsv")
PDB_DIR = Path("ProteinLM_JCIM/10_AlphaFold/01_pdb_v6")

OUT = Path("ProteinLM_JCIM/11_TMD_AlphaFold/01_mapping/tmd_alphafold_residue_mapping.tsv")
OUT.parent.mkdir(parents=True, exist_ok=True)

master = pd.read_csv(MASTER, sep="\t")
afmap = pd.read_csv(AFMAP, sep="\t")[["gene", "uniprot_accession"]]

df = master.merge(afmap, on="gene", how="left")

rows = []

for _, r in df.iterrows():
    gene = r["gene"]
    acc = r["uniprot_accession"]

    pdb_file = PDB_DIR / f"{gene}__{acc}.pdb"

    af_available = int(pdb_file.exists())

    rows.append({
        "gene": gene,
        "key": r["key"],
        "tmd_index": r["tmd_index"],
        "tmd_start_aa": int(r["tmd_start_aa"]),
        "tmd_end_aa": int(r["tmd_end_aa"]),
        "tmd_length_aa": int(r["tmd_length_aa"]),
        "protein_length": int(r["protein_length"]),
        "classification": r["classification"],
        "label_WT_lost": int(r["label_WT_lost"]),
        "uniprot_accession": acc,
        "alphafold_available": af_available,
        "local_pdb": str(pdb_file) if af_available else ""
    })

out = pd.DataFrame(rows)

out.to_csv(OUT, sep="\t", index=False)

print("TMD rows:", len(out))
print("Unique proteins:", out["gene"].nunique())
print("WT-lost TMDs:", int(out["label_WT_lost"].sum()))
print("AlphaFold available TMDs:", int(out["alphafold_available"].sum()))
print("Missing AlphaFold TMDs:", int((out["alphafold_available"] == 0).sum()))
print("Saved:", OUT)

print()
print(out.head())
