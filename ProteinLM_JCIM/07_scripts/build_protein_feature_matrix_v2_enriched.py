import re
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO

MASTER = Path("master_feature_table_v2.tsv")
PROTEIN_MASTER = Path("ProteinLM_JCIM/00_data/protein_master_table.tsv")
FASTA = Path("~/yeast_project/Saccharomyces_cerevisiae.R64-1-1.pep.all.fa").expanduser()
OUT = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")

HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
KR = set("KR")
DE = set("DE")

def mean_hydro(seq):
    vals = [HYDRO[a] for a in seq if a in HYDRO]
    return np.mean(vals) if vals else np.nan

def frac(seq, aa_set):
    seq = [a for a in seq if a in HYDRO]
    if not seq:
        return np.nan
    return sum(a in aa_set for a in seq) / len(seq)

def seq_stats(seq, prefix):
    return {
        f"{prefix}_hydrophobicity_mean": mean_hydro(seq),
        f"{prefix}_aromatic_fraction": frac(seq, AROMATIC),
        f"{prefix}_positive_fraction": frac(seq, POSITIVE),
        f"{prefix}_negative_fraction": frac(seq, NEGATIVE),
        f"{prefix}_KR_fraction": frac(seq, KR),
        f"{prefix}_DE_fraction": frac(seq, DE),
        f"{prefix}_charge_balance_pos_minus_neg": frac(seq, POSITIVE) - frac(seq, NEGATIVE),
    }

def add_key(d, key, seq):
    if key and key not in d:
        d[key] = seq

seqs = {}

for record in SeqIO.parse(str(FASTA), "fasta"):
    seq = str(record.seq)
    desc = record.description

    add_key(seqs, record.id, seq)

    m = re.search(r"\bgene:([^\s]+)", desc)
    if m:
        add_key(seqs, m.group(1), seq)

    m = re.search(r"\bgene_symbol:([^\s]+)", desc)
    if m:
        add_key(seqs, m.group(1), seq)

tmd = pd.read_csv(MASTER, sep="\t")
protein = pd.read_csv(PROTEIN_MASTER, sep="\t")

rows = []

for gene, g in tmd.groupby("gene"):
    seq = seqs.get(gene)

    if seq is None:
        continue

    tmd_seqs = []
    intervals = []

    for _, r in g.iterrows():
        start = int(r["tmd_start_aa"])
        end = int(r["tmd_end_aa"])
        tmd_seqs.append(seq[start - 1:end])
        intervals.append((start, end))

    lengths = np.array([len(s) for s in tmd_seqs], dtype=float)
    positions = g["relative_tmd_position"].astype(float).values

    tmd_hydro = np.array([mean_hydro(s) for s in tmd_seqs], dtype=float)
    tmd_arom = np.array([frac(s, AROMATIC) for s in tmd_seqs], dtype=float)
    tmd_pos = np.array([frac(s, POSITIVE) for s in tmd_seqs], dtype=float)
    tmd_neg = np.array([frac(s, NEGATIVE) for s in tmd_seqs], dtype=float)
    tmd_charge = tmd_pos - tmd_neg

    intervals_sorted = sorted(intervals)
    gaps = []
    for i in range(1, len(intervals_sorted)):
        prev_end = intervals_sorted[i-1][1]
        cur_start = intervals_sorted[i][0]
        gaps.append(max(0, cur_start - prev_end - 1))

    row = {
        "gene": gene,

        # topology / architecture
        "tmd_length_median": float(np.median(lengths)),
        "tmd_length_std": float(np.std(lengths)),
        "tmd_length_min": float(np.min(lengths)),
        "tmd_length_max": float(np.max(lengths)),
        "relative_position_std": float(np.std(positions)),
        "first_tmd_position": float(np.min(positions)),
        "last_tmd_position": float(np.max(positions)),
        "tmd_density": float(len(tmd_seqs) / len(seq)),
        "tmd_aa_fraction": float(np.sum(lengths) / len(seq)),
        "mean_tmd_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_tmd_gap": float(np.max(gaps)) if gaps else 0.0,

        # protein-level sequence composition
        **seq_stats(seq, "protein"),

        # TMD-level composition summarized per protein
        "tmd_hydrophobicity_mean": float(np.nanmean(tmd_hydro)),
        "tmd_hydrophobicity_std": float(np.nanstd(tmd_hydro)),
        "tmd_hydrophobicity_min": float(np.nanmin(tmd_hydro)),
        "tmd_hydrophobicity_max": float(np.nanmax(tmd_hydro)),
        "tmd_aromatic_fraction_mean": float(np.nanmean(tmd_arom)),
        "tmd_aromatic_fraction_std": float(np.nanstd(tmd_arom)),
        "tmd_positive_fraction_mean": float(np.nanmean(tmd_pos)),
        "tmd_positive_fraction_std": float(np.nanstd(tmd_pos)),
        "tmd_negative_fraction_mean": float(np.nanmean(tmd_neg)),
        "tmd_negative_fraction_std": float(np.nanstd(tmd_neg)),
        "tmd_charge_balance_mean": float(np.nanmean(tmd_charge)),
        "tmd_charge_balance_std": float(np.nanstd(tmd_charge)),
    }

    rows.append(row)

extra = pd.DataFrame(rows)

out = protein.merge(extra, on="gene", how="left")
out.to_csv(OUT, sep="\t", index=False)

print("Proteins:", len(out))
print("Columns:", len(out.columns))
print("New feature columns:", len(extra.columns) - 1)
print("Missing values total:", int(out.isna().sum().sum()))
print("Positive proteins:", int(out["has_WT_lost"].sum()))
print("Saved:", OUT)

print()
print("Added columns:")
for c in extra.columns:
    if c != "gene":
        print(c)
