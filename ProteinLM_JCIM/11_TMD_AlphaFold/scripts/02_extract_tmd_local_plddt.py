import numpy as np
import pandas as pd
from pathlib import Path

MAP = Path("ProteinLM_JCIM/11_TMD_AlphaFold/01_mapping/tmd_alphafold_residue_mapping.tsv")
OUT = Path("ProteinLM_JCIM/11_TMD_AlphaFold/02_features/tmd_local_plddt_features.tsv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MAP, sep="\t")

def parse_ca_plddt(pdb_file):
    plddt = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resi = int(line[22:26])
            bfactor = float(line[60:66])
            plddt[resi] = bfactor
    return plddt

cache = {}

def get_region_stats(plddt_dict, start, end, prefix):
    vals = [plddt_dict[i] for i in range(start, end + 1) if i in plddt_dict]
    if len(vals) == 0:
        return {
            f"{prefix}_n_residues": 0,
            f"{prefix}_plddt_mean": np.nan,
            f"{prefix}_plddt_median": np.nan,
            f"{prefix}_plddt_std": np.nan,
            f"{prefix}_plddt_min": np.nan,
            f"{prefix}_plddt_max": np.nan,
            f"{prefix}_frac_plddt_gt90": np.nan,
            f"{prefix}_frac_plddt_lt50": np.nan,
        }

    vals = np.array(vals, dtype=float)
    return {
        f"{prefix}_n_residues": int(len(vals)),
        f"{prefix}_plddt_mean": float(np.mean(vals)),
        f"{prefix}_plddt_median": float(np.median(vals)),
        f"{prefix}_plddt_std": float(np.std(vals)),
        f"{prefix}_plddt_min": float(np.min(vals)),
        f"{prefix}_plddt_max": float(np.max(vals)),
        f"{prefix}_frac_plddt_gt90": float(np.mean(vals > 90)),
        f"{prefix}_frac_plddt_lt50": float(np.mean(vals < 50)),
    }

rows = []

for _, r in df.iterrows():
    gene = r["gene"]
    pdb_file = r["local_pdb"]

    base = {
        "gene": gene,
        "key": r["key"],
        "tmd_index": r["tmd_index"],
        "tmd_start_aa": r["tmd_start_aa"],
        "tmd_end_aa": r["tmd_end_aa"],
        "classification": r["classification"],
        "label_WT_lost": r["label_WT_lost"],
        "alphafold_available": r["alphafold_available"],
    }

    if r["alphafold_available"] != 1 or not isinstance(pdb_file, str) or not Path(pdb_file).exists():
        base["local_plddt_available"] = 0
        rows.append(base)
        continue

    if pdb_file not in cache:
        cache[pdb_file] = parse_ca_plddt(pdb_file)

    plddt = cache[pdb_file]

    start = int(r["tmd_start_aa"])
    end = int(r["tmd_end_aa"])
    plen = int(r["protein_length"])

    upstream_start = max(1, start - 20)
    upstream_end = start - 1

    downstream_start = end + 1
    downstream_end = min(plen, end + 20)

    window_start = max(1, start - 20)
    window_end = min(plen, end + 20)

    base["local_plddt_available"] = 1

    base.update(get_region_stats(plddt, start, end, "tmd"))
    base.update(get_region_stats(plddt, upstream_start, upstream_end, "up20"))
    base.update(get_region_stats(plddt, downstream_start, downstream_end, "down20"))
    base.update(get_region_stats(plddt, window_start, window_end, "window20"))

    base["tmd_vs_window20_plddt_delta"] = base["tmd_plddt_mean"] - base["window20_plddt_mean"]
    base["tmd_vs_up20_plddt_delta"] = base["tmd_plddt_mean"] - base["up20_plddt_mean"]
    base["tmd_vs_down20_plddt_delta"] = base["tmd_plddt_mean"] - base["down20_plddt_mean"]

    rows.append(base)

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)

print("TMD rows:", len(out))
print("Available:", int(out["local_plddt_available"].sum()))
print("Missing:", int((out["local_plddt_available"] == 0).sum()))
print("WT-lost available:", int(out.loc[out["local_plddt_available"] == 1, "label_WT_lost"].sum()))
print("Saved:", OUT)
print()
print(out.head())
