# Progress documents

Earlier deliverables, kept as dated records. They are not updated. Where a later study changed a result, the final report (`../Project_Report.pdf`) is the one to read.

| File | Date | Based on | What later changed |
|---|---|---|---|
| `2026-09_project_understanding_report.pdf` | September 2026 | Project statement | Nothing |
| `2026-10-15_progress_report.pdf` | 15 October 2026 | Early runs of study 2 | Its preliminary claim that speculative decoding collapses for lack of GPU memory and that AWQ restores it came from runs later excluded (ADR-005). It is not supported by the final data. |
| `2026-10-16_week6_report.pdf`, `2026-10-16_week6_slides.pdf` | 16 October 2026 | Study 1 | Covers one prompt length on the 3B model. Superseded by study 3. |
| `2026-10-31_progress_presentation.pptx` | 31 October 2026 | Study 2 | Speculative-decoding gains (lowest TPOT with AWQ + speculative; more capacity at long prompts) were not reproduced on the 14B model. The prefix-sharing slide compares two sweeps and is not a valid caching measurement (see the study 2 README). Goodput there is per run, not per request. |
