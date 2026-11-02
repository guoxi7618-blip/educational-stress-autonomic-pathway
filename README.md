# Within-person autonomic pathways link educational stress to subclinical psychological distress in university students

Code, summary-level data and figure source data for the manuscript of the same title.

## Contents

| Folder | Content |
| --- | --- |
| `code/` | Acquisition, preprocessing, analysis and figure code. See `code/README.md`. |
| `data_for_reviewers/` | Summary-level results, codebooks and manuscript verification tables. See `data_for_reviewers/README.md`. |
| `source_data/` | `Source_Data.xlsx`, the data behind each table and figure panel. |
| `figures/` | Fig. 1–6 and Supplementary Fig. 1 as vector PDF. |

## Reproduce the figures

```bash
pip install matplotlib==3.10.0 python-pptx openpyxl lxml numpy
python code/figures/make_figures.py
```

The script reads `source_data/Source_Data.xlsx` and `data_for_reviewers/` and writes PDF, PNG and an editable PowerPoint deck to `figures/`. It uses the Arial font.

## Names used in the code

Analysis scripts and `code/export_data_for_reviewers.py` keep earlier working names: `crisis` or `Y` is subclinical psychological distress in the manuscript, and `subclinical state` is the above-threshold state. Running the export script writes those earlier names to its output.

## Data not in this repository

Record-level tables, raw physiological recordings, suicide-risk variables and person-level model outputs are not included, so the analysis scripts cannot be re-run from this repository alone. `data_for_reviewers/09_codebook/` documents those tables variable by variable. Access follows the Data availability statement of the manuscript.
