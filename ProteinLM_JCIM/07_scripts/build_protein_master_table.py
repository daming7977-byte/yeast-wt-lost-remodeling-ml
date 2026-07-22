import pandas as pd
from pathlib import Path

INFILE = Path("master_feature_table_v2.tsv")
OUTFILE = Path("ProteinLM_JCIM/00_data/protein_master_table.tsv")

df = pd.read_csv(INFILE, sep="\t")

protein = (
    df.groupby("gene")
    .agg(
        protein_length=("protein_length", "max"),
        n_tmd=("tmd_index", "count"),
        n_WT_lost=("label_WT_lost", "sum"),
        has_WT_lost=("label_WT_lost", "max"),
        n_any_eS7_event=("label_any_eS7_event", "sum"),
        has_any_eS7_event=("label_any_eS7_event", "max"),
        n_4KR_specific=("label_4KR_specific", "sum"),
        has_4KR_specific=("label_4KR_specific", "max"),
        n_shared=("label_shared", "sum"),
        has_shared=("label_shared", "max"),
        single_pass=("single_pass", "max"),
        multi_pass=("multi_pass", "max"),
        loc_er=("loc_er", "max"),
        loc_golgi=("loc_golgi", "max"),
        loc_cell_membrane=("loc_cell_membrane", "max"),
        loc_mitochondrion=("loc_mitochondrion", "max"),
        loc_vacuole=("loc_vacuole", "max"),
        loc_nucleus=("loc_nucleus", "max"),
        is_transporter_like=("is_transporter_like", "max"),
        is_transferase_like=("is_transferase_like", "max"),
        mean_tmd_length_aa=("tmd_length_aa", "mean"),
        mean_relative_tmd_position=("relative_tmd_position", "mean"),
        min_relative_tmd_position=("relative_tmd_position", "min"),
        max_relative_tmd_position=("relative_tmd_position", "max"),
        subcellular_location=("subcellular_location", "first"),
        protein_name=("protein_name", "first"),
        entry_name=("entry_name", "first"),
    )
    .reset_index()
)

protein["has_WT_lost"] = protein["has_WT_lost"].astype(int)
protein["has_any_eS7_event"] = protein["has_any_eS7_event"].astype(int)
protein["has_4KR_specific"] = protein["has_4KR_specific"].astype(int)
protein["has_shared"] = protein["has_shared"].astype(int)

protein.to_csv(OUTFILE, sep="\t", index=False)

print("Input TMD rows:", len(df))
print("Output proteins:", len(protein))
print("WT-lost positive proteins:", int(protein["has_WT_lost"].sum()))
print("Background proteins:", int((protein["has_WT_lost"] == 0).sum()))
print("Any eS7-event proteins:", int(protein["has_any_eS7_event"].sum()))
print("Saved:", OUTFILE)

print()
print("Top rows:")
print(protein.head())
