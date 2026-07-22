import pandas as pd
from pathlib import Path

PROTEIN = Path("ProteinLM_JCIM/00_data/protein_master_table.tsv")
UNIPROT = Path("ProteinLM_JCIM/10_AlphaFold/00_data/uniprot_yeast_reference.tsv")
OUT = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_mapping_with_accession.tsv")

protein = pd.read_csv(PROTEIN, sep="\t")
uni = pd.read_csv(UNIPROT, sep="\t")

uni = uni.rename(columns={
    "Entry": "uniprot_accession",
    "Entry Name": "entry_name",
    "Gene Names": "uniprot_gene_names",
    "Protein names": "uniprot_protein_name",
    "Length": "uniprot_length"
})

merged = protein.merge(
    uni[["uniprot_accession", "entry_name", "uniprot_gene_names", "uniprot_protein_name", "uniprot_length"]],
    on="entry_name",
    how="left"
)

merged["alphafold_id"] = merged["uniprot_accession"].apply(
    lambda x: f"AF-{x}-F1-model_v4" if pd.notna(x) else ""
)

merged["alphafold_pdb_url"] = merged["uniprot_accession"].apply(
    lambda x: f"https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.pdb" if pd.notna(x) else ""
)

merged["alphafold_cif_url"] = merged["uniprot_accession"].apply(
    lambda x: f"https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.cif" if pd.notna(x) else ""
)

merged.to_csv(OUT, sep="\t", index=False)

print("Project proteins:", len(protein))
print("UniProt reference rows:", len(uni))
print("Mapped accessions:", merged["uniprot_accession"].notna().sum())
print("Unmapped:", merged["uniprot_accession"].isna().sum())
print("Saved:", OUT)

if merged["uniprot_accession"].isna().sum() > 0:
    print("\nUnmapped examples:")
    print(merged.loc[merged["uniprot_accession"].isna(), ["gene", "entry_name", "protein_name"]].head(20))
