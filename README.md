# find_proteasome.py

### v0.0.1

A Python script to identify proteasome subunits (or other genes with HMM models) from protein FASTA files using **HMMER3** (`hmmsearch`).  
The main usage is to filter hits **based on provided gene-specific cutoffs** (`--gene_cutoff_tsv`).  
As an optional feature, the script can also estimate cutoffs from score distributions using **Gaussian Mixture Models (GMM)**.

---

## Features
- Runs `hmmsearch` on all `.hmm` files in a given directory  
- Extracts **per-gene scores** and **hit information** from `--tblout` output  
- **Main mode**: filter hits using provided cutoffs via `--gene_cutoff_tsv`  
- **Optional mode**: estimate gene-specific cutoffs from score distributions using Gaussian Mixture Models (sklearn)  
- Outputs:
  - `*_filtered_hits.tsv`: filtered hits above the cutoff, including sequences  
  - `*_filtered_hits.fasta`: FASTA sequences of filtered hits  
  - `*_gene_cutoffs.tsv`: used or estimated cutoffs per gene  
  - (optional) Histograms of score distributions with cutoff overlays (`plots/`)  

---

## Requirements
- Python ≥ 3.7  
- [HMMER3](http://hmmer.org/) (`hmmsearch` available in PATH)  
- Python packages:
  - `numpy`
  - `scikit-learn`
  - `scipy`
  - `biopython`
  - `matplotlib` (optional, only if using `--plot`)  

Install dependencies with:
```bash
pip install numpy scikit-learn scipy biopython matplotlib
````

---

## Usage

```bash
python find_proteasome.py \
  --models_dir MODELS_DIR \
  --fasta_path INPUT_FASTA \
  --output_dir OUTPUT_DIR \
  --gene_cutoff_tsv TSV \
  [--cpu N] \
  [--plot]
```

### Arguments

* `--models_dir`
  Directory containing `.hmm` files

* `--fasta_path`
  Input protein FASTA file to search

* `--output_dir`
  Base directory for outputs. Results of `hmmsearch` will be placed under `OUTPUT_DIR/hmmsearch/`

* `--gene_cutoff_tsv`
  TSV file with predefined cutoffs (`gene<TAB>cutoff`).
  **This is the main mode of usage.**

* `--cpu` *(default: 4)*
  Number of CPUs to use for hmmsearch

* `--plot`
  Save score histograms (`plots/`) with cutoffs and (optionally) fitted GMMs

---

## Example Commands

### 1. Main usage: apply predefined cutoffs

```bash
python find_proteasome.py \
  --models_dir models \
  --fasta_path sample_proteome.faa \
  --output_dir sample_output \
  --cpu 8 \
  --plot \
  --gene_cutoff_tsv estimated_cutoffs.tsv
```

Here, `estimated_cutoffs.tsv` (provided in this repository) is used to filter proteasome hits from the input proteome.

---

### 2. Optional usage: estimate cutoffs from a reference proteome

```bash
python find_proteasome.py \
  --models_dir models \
  --fasta_path reference_proteome.faa \
  --output_dir estimate_cutoff \
  --cpu 8 \
  --plot
```

If no `--gene_cutoff_tsv` is provided, the script estimates cutoffs using Gaussian Mixture Models.
This mode is intended for **deriving cutoffs once**, which can then be reused across other datasets.

---

## Input Files

* **HMM models**: stored in the `models/` directory (included in this repository, e.g., `RPT1.hmm`, `RPN3.hmm`, …)
* **Predefined cutoffs**: `estimated_cutoffs.tsv` is also included in this repository with format:

  ```
  Gene    Cutoff
  RPT1    152.4
  RPN3    128.7
  ...
  ```

---

## Output Files

* `output_dir/hmmsearch/*.tblout` — raw HMMER table outputs
* `output_dir/*_filtered_hits.tsv` — filtered hits with scores, cutoffs, and sequences
* `output_dir/*_filtered_hits.fasta` — FASTA sequences of hits above cutoff
* `output_dir/*_gene_cutoffs.tsv` — used or estimated gene cutoffs
* `output_dir/plots/*.png` — (if `--plot`) histograms with cutoffs (and optional GMM fits)

---

### Cutoff estimation details

The cutoff values were empirically optimized based on HMM score distributions from plant RefSeq proteomes:

- **RPN3**: estimated with a 2-component Gaussian Mixture Model (GMM)  
- **RPT1–RPT6**: estimated with a 4-component GMM  
- **Other subunits** (`PAA, PAB, PAC, PAD, PAE, PAF, PAG, PBA, PBB, PBC, PBD, PBE, PBF, PBG, RPN1, RPN2, RPN5–RPN7, RPN9–RPN13, DSS1, RAD23`): estimated with a 3-component GMM  
- **RPN8**: cutoff manually increased to **300** to reduce false positives  
- **RAD23**: fixed cutoff of **207.3**, requiring hits to pass this threshold in both the full-length score and the best single-domain score  

These empirical rules are reflected in the provided `estimated_cutoffs.tsv`.

