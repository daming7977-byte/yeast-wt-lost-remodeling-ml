import numpy as np
import pandas as pd
from pathlib import Path

PDB_DIR = Path("ProteinLM_JCIM/10_AlphaFold/01_pdb_v6")
MAPPING = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_mapping_with_accession.tsv")
OUT = Path("ProteinLM_JCIM/10_AlphaFold/02_features/alphafold_basic_features.tsv")

OUT.parent.mkdir(parents=True, exist_ok=True)

mapping = pd.read_csv(MAPPING, sep="\t")

def parse_ca_atoms(pdb_file):
    coords = []
    plddt = []

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # In AlphaFold PDB files, B-factor column stores pLDDT.
            bfactor = float(line[60:66])

            coords.append([x, y, z])
            plddt.append(bfactor)

    return np.array(coords, dtype=float), np.array(plddt, dtype=float)

rows = []

for _, r in mapping.iterrows():
    gene = r["gene"]
    acc = r["uniprot_accession"]
    pdb_file = PDB_DIR / f"{gene}__{acc}.pdb"

    if not pdb_file.exists():
        rows.append({
            "gene": gene,
            "uniprot_accession": acc,
            "alphafold_available": 0
        })
        continue

    coords, plddt = parse_ca_atoms(pdb_file)

    if len(coords) == 0:
        rows.append({
            "gene": gene,
            "uniprot_accession": acc,
            "alphafold_available": 0
        })
        continue

    center = coords.mean(axis=0)
    centered = coords - center

    rg = np.sqrt((centered ** 2).sum(axis=1).mean())

    dmax = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)).max()

    rows.append({
        "gene": gene,
        "uniprot_accession": acc,
        "alphafold_available": 1,
        "af_n_residues": len(coords),

        "af_plddt_mean": float(np.mean(plddt)),
        "af_plddt_median": float(np.median(plddt)),
        "af_plddt_std": float(np.std(plddt)),
        "af_plddt_min": float(np.min(plddt)),
        "af_plddt_max": float(np.max(plddt)),

        "af_frac_plddt_gt90": float(np.mean(plddt > 90)),
        "af_frac_plddt_70_90": float(np.mean((plddt >= 70) & (plddt <= 90))),
        "af_frac_plddt_50_70": float(np.mean((plddt >= 50) & (plddt < 70))),
        "af_frac_plddt_lt50": float(np.mean(plddt < 50)),

        "af_radius_of_gyration": float(rg),
        "af_max_dimension": float(dmax),
        "af_compactness_rg_per_residue": float(rg / len(coords)),
        "af_compactness_dmax_per_residue": float(dmax / len(coords)),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)

print("Mapped proteins:", len(mapping))
print("Rows written:", len(out))
print("AlphaFold available:", int(out["alphafold_available"].sum()))
print("Missing:", int((out["alphafold_available"] == 0).sum()))
print("Saved:", OUT)
print()
print(out.head())
