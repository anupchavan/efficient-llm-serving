# Evaluation

Three benchmark studies, in the order they were run. Each folder has its scripts, raw result files, generated tables and a README with the setup, results and limits.

| Folder | Model, GPU | Runs | Dates (2026) | Role |
|---|---|---|---|---|
| [`01_qwen2.5-3b_1024tok`](01_qwen2.5-3b_1024tok) | Qwen2.5-3B, RTX PRO 4000 24 GB | 201 | 7–10 Oct | First pipeline; week-6 report |
| [`02_qwen2.5-3b_lengths`](02_qwen2.5-3b_lengths) | Qwen2.5-3B, RTX PRO 4000 24 GB | 864 (216 used) | 15–26 Oct | Prompt lengths; progress presentation |
| [`03_qwen3-14b_sharegpt`](03_qwen3-14b_sharegpt) | Qwen3-14B, A100 80 GB | 504 | 8–9 Nov | Final results |

The final report's conclusions rest on study 3. Studies 1 and 2 are reported for what they showed and where study 3 contradicts them.

Regenerate every table from the raw files (no GPU needed):

```bash
pip install -r requirements.txt
python3 evaluation/01_qwen2.5-3b_1024tok/analysis/summarise.py
python3 evaluation/02_qwen2.5-3b_lengths/analysis/summarise.py
python3 evaluation/02_qwen2.5-3b_lengths/analysis/tables.py
cd evaluation/03_qwen3-14b_sharegpt
python3 analysis/summarise.py && python3 analysis/goodput.py && python3 analysis/extract_per_run.py && python3 analysis/analyse.py
```

`pytest` runs the same steps and compares the output with the committed files.
