import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MODEL = "facebook/esm2_t12_35M_UR50D"
MASTER = Path("master_feature_table_v2.tsv")
FASTA = Path("~/yeast_project/Saccharomyces_cerevisiae.R64-1-1.pep.all.fa").expanduser()
OUT = Path("ProteinLM_JCIM/01_embeddings/esm2_t12_35M_full_protein_embeddings.tsv")

df = pd.read_csv(MASTER, sep="\t")
genes = sorted(df["gene"].unique())

seqs = {}

def add_key(k, seq):
    if k and k not in seqs:
        seqs[k] = seq

for record in SeqIO.parse(str(FASTA), "fasta"):
    seq = str(record.seq)
    desc = record.description
    add_key(record.id, seq)

    m = re.search(r"\bgene:([^\s]+)", desc)
    if m:
        add_key(m.group(1), seq)

    m = re.search(r"\bgene_symbol:([^\s]+)", desc)
    if m:
        add_key(m.group(1), seq)

protein_rows = []
missing = []

for gene in genes:
    seq = seqs.get(gene)
    if seq is None:
        missing.append(gene)
    else:
        protein_rows.append({"gene": gene, "sequence": seq, "length": len(seq)})

proteins = pd.DataFrame(protein_rows)

print("Unique genes:", len(genes))
print("Matched proteins:", len(proteins))
print("Missing:", len(missing))
print("Max length:", proteins["length"].max())
print("Median length:", proteins["length"].median())

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.eval()

rows = []
batch_size = 4

for start in tqdm(range(0, len(proteins), batch_size)):
    batch = proteins.iloc[start:start + batch_size]
    seq_batch = batch["sequence"].tolist()

    inputs = tokenizer(
        seq_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1022
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    attention = inputs["attention_mask"]

    for i in range(hidden.shape[0]):
        mask = attention[i].bool()
        valid_idx = torch.where(mask)[0]

        if len(valid_idx) > 2:
            token_idx = valid_idx[1:-1]
        else:
            token_idx = valid_idx

        emb = hidden[i, token_idx, :].mean(dim=0).detach().cpu().numpy()

        row = {
            "gene": batch.iloc[i]["gene"],
            "protein_length": int(batch.iloc[i]["length"])
        }

        for j, val in enumerate(emb):
            row[f"esm2_full_{j}"] = float(val)

        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)

print("Saved:", OUT)
print("Shape:", out.shape)
