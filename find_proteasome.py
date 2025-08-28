#! /usr/bin/env python3
import os
import sys
import glob
import subprocess
import argparse
import numpy as np
import shutil
from Bio import SeqIO
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def run_hmmsearch(models_dir, fasta_path, output_dir, cpu=8):
    # Create a dedicated subdirectory for hmmsearch outputs inside output_dir
    hmm_out_dir = os.path.join(output_dir, "hmmsearch")
    os.makedirs(hmm_out_dir, exist_ok=True)
    hmm_files = sorted(glob.glob(os.path.join(models_dir, "*.hmm")))
    total = len(hmm_files)
    print(f"Found {total} HMM models. Logs: {hmm_out_dir}", flush=True)
    if total == 0:
        sys.stderr.write(f"No HMM files found in {models_dir}\n")
        sys.exit(1)
    fasta_base = os.path.splitext(os.path.basename(fasta_path))[0]
    for idx, hmm in enumerate(hmm_files, start=1):
        gene = os.path.splitext(os.path.basename(hmm))[0]
        tblout = os.path.join(hmm_out_dir, f"{fasta_base}_{gene}.tblout")
        domtblout = os.path.join(hmm_out_dir, f"{fasta_base}_{gene}.domtblout")
        log_path = os.path.join(hmm_out_dir, f"{fasta_base}_{gene}.log")
        print(f"[{idx}/{total}]\tRunning hmmsearch for {gene}...", flush=True)
        cmd = [
            "hmmsearch",
            "--cpu",
            str(cpu),
            "--noali",
            "--tblout",
            tblout,
            "--domtblout",
            domtblout,
            hmm,
            fasta_path,
        ]
        with open(log_path, "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=logf)


def extract_scores_from_tblout(tblout_path):
    scores = []
    with open(tblout_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) > 5:
                try:
                    score = float(parts[5])
                    scores.append(score)
                except ValueError:
                    sys.stderr.write(f"Error parsing score in {tblout_path}: {line}")
                    sys.exit(1)
    return scores


def extract_all_scores_from_dir(tblout_dir, fasta_base=None):
    tblout_files = glob.glob(os.path.join(tblout_dir, "*.tblout"))
    all_scores = {}
    for tblout in tblout_files:
        fname = os.path.basename(tblout)
        if fasta_base and fname.startswith(fasta_base + "_"):
            hmm_base = fname[len(fasta_base) + 1 : -len(".tblout")]
        else:
            hmm_base = os.path.splitext(fname)[0]
        scores = extract_scores_from_tblout(tblout)
        all_scores[hmm_base] = scores
    return all_scores


def extract_hits_from_tblout(tblout_path):
    """Extract per-hit gene id and scores from HMMER --tblout file.
    Returns list of tuples: (gene_id, full_seq_score, best1dom_score)
    """
    hits = []
    with open(tblout_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            # Expect both full sequence score (parts[5]) and best 1 domain score (parts[8])
            if len(parts) > 8:
                try:
                    gene_id = parts[0]
                    full_seq_score = float(parts[5])
                    best1dom_score = float(parts[8])
                    hits.append((gene_id, full_seq_score, best1dom_score))
                except ValueError:
                    sys.stderr.write(f"Error parsing score in {tblout_path}: {line}")
                    sys.exit(1)
    return hits


def collect_gene_hits(tblout_dir, fasta_base=None):
    """Collect hits from tblout grouped by HMM model (gene) name.
    Returns dict: { gene: [(gene_id, score), ...] }
    """
    tblout_files = glob.glob(os.path.join(tblout_dir, "*.tblout"))
    gene_hits = {}
    for tblout in tblout_files:
        fname = os.path.basename(tblout)
        if fasta_base and fname.startswith(fasta_base + "_"):
            gene = fname[len(fasta_base) + 1 : -len(".tblout")]
        else:
            gene = os.path.splitext(fname)[0]
        gene_hits[gene] = extract_hits_from_tblout(tblout)
    return gene_hits


def calculate_hmm_score_cutoffs(output_scores, null_posterior_threshold):
    hmm_genes = sorted(output_scores.keys())
    cutoffs = {}
    gmm_infos = {}
    for gene in hmm_genes:
        all_scores = np.array(output_scores.get(gene, []))
        if len(all_scores) == 0:
            cutoffs[gene] = None
            gmm_infos[gene] = None
            continue
        if gene == "RPN3":
            ncomp = 2
        elif gene.startswith("RPT"):
            ncomp = 4
        else:
            ncomp = 3
        gmm = GaussianMixture(n_components=ncomp).fit(all_scores.reshape(-1, 1))
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()
        # Sort means in ascending order
        sorted_idx = np.argsort(means)
        sorted_means = means[sorted_idx]
        sorted_covariances = covariances[sorted_idx]
        sorted_weights = weights[sorted_idx]
        # Target component: rightmost (maximum mean)
        target_idx = ncomp - 1

        left_mean = sorted_means[-2]
        left_std = float(np.sqrt(sorted_covariances[-2]))

        right_mean = sorted_means[-1]
        right_std = float(np.sqrt(sorted_covariances[-1]))

        data_min = float(np.min(all_scores))
        data_max = float(np.max(all_scores))
        margin_left = min(left_mean - 6 * left_std, data_min - 1.0)
        margin_right = max(right_mean + 6 * right_std, data_max + 1.0)

        x = np.linspace(margin_left, margin_right, 4000)

        comp_pdfs = [
            sorted_weights[i]
            * norm.pdf(x, sorted_means[i], float(np.sqrt(sorted_covariances[i])))
            for i in range(ncomp)
        ]
        total_pdf = np.sum(comp_pdfs, axis=0)
        total_pdf = np.maximum(total_pdf, 1e-300)  # Numerical stability
        target_pdf = comp_pdfs[target_idx]
        target_posterior = target_pdf / total_pdf

        # null posterior <= t  <=>  target posterior >= 1 - t
        t = float(null_posterior_threshold)
        thresh = 1.0 - t
        idxs = np.where(target_posterior >= thresh)[0]
        cutoff = x[idxs[0]]
        cutoffs[gene] = cutoff
        gmm_infos[gene] = {
            "ncomp": ncomp,
            "means": sorted_means,
            "covariances": sorted_covariances,
            "weights": sorted_weights,
            "x": x,
            "comp_pdfs": comp_pdfs,
            "total_pdf": total_pdf,
            "target_pdf": target_pdf,
            "target_posterior": target_posterior,
        }
        print(
            f"{gene} HMM score cutoff: {cutoff} (GMM n_components={ncomp}, null posterior <= {t})"
        )
    return cutoffs, gmm_infos


def build_filtered_hits_rows(output_scores, output_hits, cutoffs, seqs):
    """Build rows for output TSV.
    Returns list of tuples: (query_gene, hit_gene_id, score, cutoff, sequence)
    """
    rows = []
    for gene in sorted(output_scores.keys()):
        cutoff = cutoffs.get(gene)
        if cutoff is None:
            continue
        hits = output_hits.get(gene, [])
        # hits tuples: (gene_id, full_seq_score, best1dom_score)
        for gene_id, full_score, best1_score in sorted(
            hits, key=lambda x: (-x[1], x[0])
        ):
            if gene == "RAD23":
                ok = (full_score >= cutoff) and (best1_score >= cutoff)
            else:
                ok = full_score >= cutoff
            if ok:
                seq = str(seqs.get(gene_id, ""))
                rows.append((gene, gene_id, full_score, round(cutoff, 2), seq))
    return rows


def write_tsv(out_path, header_cols, rows):
    """Write rows to a TSV file with a header. Returns number of data rows written."""
    with open(out_path, "w") as wf:
        wf.write("\t".join(header_cols) + "\n")
        for r in rows:
            wf.write("\t".join(map(str, r)) + "\n")
    return len(rows)


def write_fasta(out_path, rows):
    """Write sequences from rows to FASTA. Returns number of sequences written.
    Expects each row as: (gene, gene_id, score, cutoff, sequence)
    """
    count = 0
    with open(out_path, "w") as wf:
        for gene, gene_id, score, cutoff, seq in rows:
            if not seq:
                continue
            wf.write(f">{gene_id} gene={gene} score={score} cutoff={cutoff}\n")
            wf.write(str(seq) + "\n")
            count += 1
    return count


def read_gene_cutoffs_tsv(tsv_path):
    """Read a TSV with two columns: gene<TAB>cutoff. Returns dict {gene: float_cutoff}."""
    cutoffs = {}
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Gene\tCutoff"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            gene, val = parts[0], parts[1]
            try:
                cutoffs[gene] = float(val)
            except ValueError:
                # Likely a header row; skip
                sys.stderr.write(f"Skipping invalid cutoff line: {line}\n")
                sys.exit(1)
    return cutoffs


def plot_cutoff_histograms(
    output_scores,
    cutoffs,
    gmm_infos,
    out_dir,
    fasta_base=None,
    null_posterior_threshold=None,
):
    """Plot score histograms for output scores per gene and mark cutoff.
    Also overlays fitted GMM distributions if possible (using precomputed GMM info).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return 0

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    n_plots = 0
    genes = sorted(output_scores.keys())
    for gene in genes:
        out_vals = np.array(output_scores.get(gene, []), dtype=float)
        if out_vals.size == 0:
            continue
        combined = out_vals

        # bins for hist
        bins = min(50, max(10, int(np.sqrt(combined.size))))
        bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            out_vals,
            bins=bin_edges,
            alpha=0.6,
            label="output scores",
            color="tab:orange",
        )

        cutoff = cutoffs.get(gene)
        if cutoff is not None:
            if null_posterior_threshold is None:
                cutoff_label = f"cutoff = {cutoff:.3f}"
            else:
                cutoff_label = f"cutoff = {cutoff:.3f} (null <= {float(null_posterior_threshold):.2f})"
            ax.axvline(
                cutoff,
                color="red",
                linestyle="--",
                linewidth=2,
                label=cutoff_label,
            )

        # === GMM overlay (use precomputed info) ===
        gmm_info = gmm_infos.get(gene)
        if gmm_info is not None:
            x = gmm_info["x"]
            ncomp = gmm_info["ncomp"]
            comp_pdfs = gmm_info["comp_pdfs"]
            total_pdf = gmm_info["total_pdf"]
            scale = len(combined) * (bin_edges[1] - bin_edges[0])
            ax.plot(
                x,
                total_pdf * scale,
                color="black",
                lw=2,
                label=f"GMM total (n={ncomp})",
            )
            for i, comp_pdf in enumerate(comp_pdfs):
                ax.plot(x, comp_pdf * scale, lw=1.5, ls="--", label=f"comp {i+1}")

        prefix = f"{gene} - " if gene else ""
        ax.set_title(f"{prefix}HMM score distribution")
        ax.set_xlabel("HMM score")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()

        out_name = f"{fasta_base + '_' if fasta_base else ''}{gene}_score_hist.png"
        fig.savefig(os.path.join(plots_dir, out_name), dpi=150)
        plt.close(fig)
        n_plots += 1

    print(f"Saved {n_plots} plots to {plots_dir}")
    return n_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hmmsearch for all HMMs in a directory and extract scores."
    )
    parser.add_argument(
        "--models_dir",
        required=True,
        help="Directory containing .hmm and .tblout files",
    )
    parser.add_argument("--fasta_path", required=True, help="Input protein FASTA file")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base directory to write outputs (hmmsearch results saved under 'hmmsearch/')",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=4,
        help="Number of CPUs for hmmsearch (default: 4)",
    )
    parser.add_argument(
        "--null_posterior_threshold",
        dest="null_posterior_threshold",
        type=float,
        default=0.5,
        help=("Threshold on posterior null probability. " "Default: 0.5"),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save histograms of score distributions with cutoffs to PNG files.",
    )
    parser.add_argument(
        "--gene_cutoff_tsv",
        help="TSV file with two columns: gene\tcutoff. If provided, skip model extraction and GMM cutoff estimation.",
    )
    args = parser.parse_args()

    fasta_base = os.path.splitext(os.path.basename(args.fasta_path))[0]

    # 1. Run hmmsearch for each hmm model (results in output_dir/hmmsearch)
    run_hmmsearch(args.models_dir, args.fasta_path, args.output_dir, cpu=args.cpu)

    # 2. Extract scores or load cutoffs from TSV
    hmm_out_dir = os.path.join(args.output_dir, "hmmsearch")
    output_scores = extract_all_scores_from_dir(hmm_out_dir, fasta_base=fasta_base)
    if args.gene_cutoff_tsv:
        print(f"Using provided gene cutoff TSV: {args.gene_cutoff_tsv}")
        cutoffs = read_gene_cutoffs_tsv(args.gene_cutoff_tsv)
        gmm_infos = {}
        if args.plot:
            plot_cutoff_histograms(
                output_scores,
                cutoffs,
                gmm_infos,
                out_dir=args.output_dir,
                fasta_base=fasta_base,
                null_posterior_threshold=args.null_posterior_threshold,
            )
    else:
        cutoffs, gmm_infos = calculate_hmm_score_cutoffs(
            output_scores,
            null_posterior_threshold=args.null_posterior_threshold,
        )
        print(f"Fixed RAD23 cutoff: {cutoffs['RAD23']} -> 207.3")
        cutoffs["RAD23"] = 207.3
        if args.plot:
            plot_cutoff_histograms(
                output_scores,
                cutoffs,
                gmm_infos,
                out_dir=args.output_dir,
                fasta_base=fasta_base,
                null_posterior_threshold=args.null_posterior_threshold,
            )

    # 3. Read output tblout hits and summarize those exceeding the cutoff
    output_hits = collect_gene_hits(hmm_out_dir, fasta_base=fasta_base)
    # Build a mapping from hit gene_id to sequence from input FASTA
    seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.fasta_path, "fasta")}
    # Apply fixed cutoff for RAD23 only if not already provided
    if args.gene_cutoff_tsv:
        cutoff_tsv = os.path.join(
            args.output_dir, f"{os.path.basename(args.gene_cutoff_tsv)}"
        )
        shutil.copyfile(args.gene_cutoff_tsv, cutoff_tsv)
        print(f"Copied provided gene cutoffs to {cutoff_tsv}")
    else:
        cutoff_tsv = os.path.join(args.output_dir, f"{fasta_base}_gene_cutoffs.tsv")
        cutoff_rows = [
            (gene, round(cutoff, 2))
            for gene, cutoff in sorted(cutoffs.items())
            if cutoff is not None
        ]
        n_cutoff_rows = write_tsv(cutoff_tsv, ["Gene", "Cutoff"], cutoff_rows)
        print(f"Wrote {n_cutoff_rows} gene cutoffs to {cutoff_tsv}")
    rows = build_filtered_hits_rows(output_scores, output_hits, cutoffs, seqs)
    out_tsv = os.path.join(args.output_dir, f"{fasta_base}_filtered_hits.tsv")
    total_rows = write_tsv(
        out_tsv,
        ["Query gene ID", "Hit gene ID", "Score", "Cutoff", "Sequence"],
        rows,
    )
    print(f"Wrote {total_rows} hits to {out_tsv}")

    # Write FASTA of filtered hits
    out_fa = os.path.join(args.output_dir, f"{fasta_base}_filtered_hits.fasta")
    total_seqs = write_fasta(out_fa, rows)
    print(f"Wrote {total_seqs} sequences to {out_fa}")
    out_fa = os.path.join(args.output_dir, f"{fasta_base}_filtered_hits.fasta")
    total_seqs = write_fasta(out_fa, rows)
    print(f"Wrote {total_seqs} sequences to {out_fa}")
    print(f"Wrote {total_rows} hits to {out_tsv}")
    out_tsv = os.path.join(args.output_dir, f"{fasta_base}_filtered_hits.tsv")
    total_rows = write_tsv(
        out_tsv,
        ["Query gene ID", "Hit gene ID", "Score", "Cutoff", "Sequence"],
        rows,
    )
    print(f"Wrote {total_rows} hits to {out_tsv}")
