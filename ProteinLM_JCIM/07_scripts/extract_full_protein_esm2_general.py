import argparse
import re
import torch
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1022)
    return p.parse_args()

args = parse_args()

MASTER = Path("ProteinLM_JCIM/00_data/protein_master_table.tsv")
FASTA = Path("~/yeast_project/Saccharomyces_cerevisiae.R64-1-1.pep.all.fa").expanduser()
OUT = Path(f"ProteinLM_JCIM/01_embeddings/{args.tag}_full_protein_embeddings.tsv")

protein = pd.read_csv(MASTER, sep="\t")
genes = sorted(protein["gene"].unique())

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

rows = []
missing = []

for gene in genes:
    seq = seqs.get(gene)
    if seq is None:
        missing.append(gene)
    else:
        rows.append({
            "gene": gene,
            "sequence": seq,
            "protein_length_from_fasta": len(seq)
        })

seqdf = pd.DataFrame(rows)

print("Model:", args.model)
print("Tag:", args.tag)
print("Proteins in master:", len(genes))
print("Matched proteins:", len(seqdf))
print("Missing:", len(missing))
print("Max length:", seqdf["protein_length_from_fasta"].max())
print("Median length:", seqdf["protein_length_from_fasta"].median())

if missing:
    print("Missing examples:", missing[:20])

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model).to(device)
model.eval()

out_rows = []

for start in tqdm(range(0, len(seqdf), args.batch_size)):
    batch = seqdf.iloc[start:start + args.batch_size]
    seq_batch = batch["sequence"].astype(str).tolist()

    inputs = tokenizer(
        seq_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length
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
            "protein_length_from_fasta": int(batch.iloc[i]["protein_length_from_fasta"])
        }

        for j, val in enumerate(emb):
            row[f"{args.tag}_{j}"] = float(val)

        out_rows.append(row)

out = pd.DataFrame(out_rows)
out.to_csv(OUT, sep="\t", index=False)

print("Saved:", OUT)
print("Shape:", out.shape)
