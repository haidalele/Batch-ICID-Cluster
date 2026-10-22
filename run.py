from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from batch_icid import BatchICIDCluster, graph_to_adjacency


PROJECT_DIR = Path(__file__).resolve().parent


def project_path(path: Path) -> Path:
    """Resolve relative paths from the directory containing run.py."""
    return path if path.is_absolute() else PROJECT_DIR / path


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open(encoding="utf-8-sig", newline="")


def read_edge_label_files(edge_path: Path, label_path: Path):
    labels_by_node: dict[str, str] = {}
    label_suffix = label_path.with_suffix("").suffix.lower() if label_path.suffix.lower() == ".gz" \
        else label_path.suffix.lower()
    with open_text(label_path) as f:
        if label_suffix in {".csv", ".tsv"}:
            delimiter = "\t" if label_suffix == ".tsv" else ","
            for row in csv.reader(f, delimiter=delimiter):
                if len(row) >= 2:
                    labels_by_node[row[0].strip()] = row[1].strip()
        else:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and not line.lstrip().startswith("#"):
                    labels_by_node[parts[0]] = parts[1]

    nodes = list(labels_by_node)
    node_index = {node: i for i, node in enumerate(nodes)}
    edges: list[tuple[int, int]] = []
    with open_text(edge_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or line.lstrip().startswith("#"):
                continue
            u, v = parts[0], parts[1]
            for node in (u, v):
                if node not in node_index:
                    node_index[node] = len(nodes)
                    nodes.append(node)
            edges.append((node_index[u], node_index[v]))

    missing = [node for node in nodes if node not in labels_by_node]
    if missing:
        raise ValueError(f"{label_path}: missing labels for {len(missing)} nodes; first: {missing[0]}")
    if not edges:
        raise ValueError(f"{edge_path}: no edges found")
    return nodes, edges, [labels_by_node[node] for node in nodes]


def read_gml_dataset(path: Path, truth_attribute: str):
    # These supplied GML files use simple node/edge blocks. Read the IDs,
    # labels and one numeric ground-truth attribute without another dependency.
    content = path.read_text(encoding="utf-8-sig")
    node_blocks = re.findall(r"\bnode\s*\[(.*?)\]", content, flags=re.DOTALL)
    edge_blocks = re.findall(r"\bedge\s*\[(.*?)\]", content, flags=re.DOTALL)
    nodes, labels = [], []
    for block in node_blocks:
        node_id = re.search(r"\bid\s+(-?\d+)", block)
        label = re.search(r'\blabel\s+"([^"]+)"', block)
        truth_label = re.search(rf"\b{re.escape(truth_attribute)}\s+(-?\d+)", block)
        if node_id is None or label is None or truth_label is None:
            raise ValueError(f"{path}: expected node id, label and '{truth_attribute}' fields in every node block")
        nodes.append(label.group(1))
        labels.append(truth_label.group(1))
    if not nodes:
        raise ValueError(f"{path}: no GML node blocks found")
    index = {node_id.group(1): i for i, block in enumerate(node_blocks)
             if (node_id := re.search(r"\bid\s+(-?\d+)", block)) is not None}
    edges = []
    for block in edge_blocks:
        source = re.search(r"\bsource\s+(-?\d+)", block)
        target = re.search(r"\btarget\s+(-?\d+)", block)
        if source is None or target is None:
            raise ValueError(f"{path}: malformed GML edge block")
        edges.append((index[source.group(1)], index[target.group(1)]))
    return nodes, edges, labels


def pair_scores(truth, predicted):
    # Pairwise micro F1 and Jaccard for node pairs.
    _, true_ids = np.unique(truth, return_inverse=True)
    _, pred_ids = np.unique(predicted, return_inverse=True)
    contingency = np.zeros((true_ids.max() + 1, pred_ids.max() + 1), dtype=np.int64)
    np.add.at(contingency, (true_ids, pred_ids), 1)
    choose2 = lambda x: x * (x - 1) // 2
    tp = int(choose2(contingency).sum())
    same_pred = int(choose2(contingency.sum(axis=0)).sum())
    same_true = int(choose2(contingency.sum(axis=1)).sum())
    fp, fn = same_pred - tp, same_true - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return f1, jaccard


def run_dataset(name: str, nodes, edges, truth, args):
    classes = sorted(set(truth))
    class_index = {label: i for i, label in enumerate(classes)}
    truth_ids = np.asarray([class_index[label] for label in truth])
    adjacency = graph_to_adjacency(edges, len(nodes))
    undirected_edge_count = int(adjacency.nnz // 2)
    model = BatchICIDCluster(args.steps, args.neighbor_max, args.beta0, args.dim, args.seed)
    predicted, features = model.fit_predict(adjacency, len(classes))
    f1, jaccard = pair_scores(truth_ids, predicted)

    output_dir = project_path(args.output) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "true_label", "predicted_cluster"])
        writer.writerows(zip(nodes, truth, predicted.tolist()))
    np.save(output_dir / "embeddings.npy", features)
    metrics = {
        "dataset": name,
        "n_nodes": len(nodes),
        "n_edges": undirected_edge_count,
        "n_communities": len(classes),
        "NMI": float(normalized_mutual_info_score(truth_ids, predicted)),
        "F_measure_pairwise": f1,
        "Jaccard_pairwise": jaccard,
        "effective_dim": int(features.shape[1]),
        "seed": args.seed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Run Batch-ICID-Cluster; default dataset paths are relative to run.py."
    )
    parser.add_argument("--dataset", choices=("all", "email", "football", "karate", "dolphins"), default="all",
                        help="Dataset to run (default: all)")
    parser.add_argument("--email-edges", type=Path, default=Path("data/email-Eu-core.txt.gz"))
    parser.add_argument("--email-labels", type=Path,
                        default=Path("data/email-Eu-core-department-labels.txt.gz"))
    parser.add_argument("--football-gml", type=Path, default=Path("data/football/football.gml"))
    parser.add_argument("--dolphins-gml", type=Path, default=Path("data/dolphins/dolphins.gml"))
    parser.add_argument("--karate-edges", type=Path, default=Path("data/karate/edges.txt"))
    parser.add_argument("--karate-labels", type=Path, default=Path("data/karate/labels.csv"))
    parser.add_argument("--edges", type=Path, help="Custom whitespace-separated edge list")
    parser.add_argument("--labels", type=Path, help="Custom node labels (CSV, TSV, whitespace or .gz)")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--neighbor-max", type=int, default=1000)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if (args.edges is None) != (args.labels is None):
        parser.error("--edges and --labels must be supplied together")
    if args.edges is not None:
        nodes, edges, labels = read_edge_label_files(project_path(args.edges), project_path(args.labels))
        run_dataset("custom", nodes, edges, labels, args)
        return

    if args.dataset in ("all", "email"):
        nodes, edges, labels = read_edge_label_files(
            project_path(args.email_edges), project_path(args.email_labels)
        )
        run_dataset("email-Eu-core", nodes, edges, labels, args)
    if args.dataset in ("all", "football"):
        nodes, edges, labels = read_gml_dataset(project_path(args.football_gml), "value")
        run_dataset("football", nodes, edges, labels, args)
    if args.dataset in ("all", "karate"):
        nodes, edges, labels = read_edge_label_files(
            project_path(args.karate_edges), project_path(args.karate_labels)
        )
        run_dataset("karate", nodes, edges, labels, args)
    if args.dataset in ("all", "dolphins"):
        nodes, edges, labels = read_gml_dataset(project_path(args.dolphins_gml), "groundTruth")
        run_dataset("dolphins", nodes, edges, labels, args)


if __name__ == "__main__":
    main()
