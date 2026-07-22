import time
import requests
import pandas as pd
from pathlib import Path

INFILE = Path("ProteinLM_JCIM/10_AlphaFold/04_results/uniprot_entry_names.txt")
OUT = Path("ProteinLM_JCIM/10_AlphaFold/04_results/uniprot_entry_to_accession.tsv")

entries = [x.strip() for x in open(INFILE) if x.strip()]

rows = []

for i, entry in enumerate(entries, 1):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=id:{entry}&fields=accession,id,protein_name,gene_names&format=tsv&size=1"
    r = requests.get(url, timeout=30)

    if r.status_code != 200:
        rows.append({
            "entry_name": entry,
            "accession": "",
            "status": f"HTTP_{r.status_code}"
        })
        continue

    lines = r.text.strip().splitlines()

    if len(lines) < 2:
        rows.append({
            "entry_name": entry,
            "accession": "",
            "status": "not_found"
        })
    else:
        parts = lines[1].split("\t")
        rows.append({
            "entry_name": entry,
            "accession": parts[0],
            "uniprot_id": parts[1] if len(parts) > 1 else "",
            "protein_name": parts[2] if len(parts) > 2 else "",
            "gene_names": parts[3] if len(parts) > 3 else "",
            "status": "mapped"
        })

    if i % 50 == 0:
        print(f"Processed {i}/{len(entries)}")

    time.sleep(0.1)

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)

print("Entries:", len(entries))
print("Mapped:", int((out["status"] == "mapped").sum()))
print("Unmapped:", int((out["status"] != "mapped").sum()))
print("Saved:", OUT)
