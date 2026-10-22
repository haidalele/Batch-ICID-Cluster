from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD

from batch_icid import BatchICIDCluster, graph_to_adjacency
from run import PROJECT_DIR, read_edge_label_files, read_gml_dataset

RESULTS = PROJECT_DIR / "results" / "chapter5_review_revision"
DATA = PROJECT_DIR / "data" / "lfr"
SEED = 42
REPEAT_SEEDS = list(range(10))
BASE = {"steps": 8, "neighbor_max": 1000, "beta0": 1.0, "dim": 100, "random_state": SEED}
BASELINE_PREPROCESSING = "binary_undirected_no_selfloops_v1"


def normalize_adjacency(adjacency):
    adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1.0
    return adjacency

# The manuscript reports n, mean/max degree, community-size ranges and mu,
# but omits tau1/tau2 and random seeds. We record these explicit generator
# assumptions rather than presenting generated graphs as the authors' files.
LFR_CASES = {
    "Syn1": {"n": 1000, "average_degree": 15, "max_degree": 50,
             "min_community": 20, "max_community": 50, "mus": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "Syn2": {"n": 1500, "average_degree": 10, "max_degree": 200,
             "min_community": 50, "max_community": 400, "mus": [0.05, 0.1, 0.15, 0.2]},
    "Syn3": {"n": 5000, "average_degree": 10, "max_degree": 200,
             "min_community": 50, "max_community": 400, "mus": [0.05, 0.1]},
    "Syn4": {"n": 10000, "average_degree": 10, "max_degree": 200,
             "min_community": 50, "max_community": 400, "mus": [0.05, 0.1]},
}
TAU1, TAU2 = 3.0, 1.5


def dataset_dir(name: str, mu: float) -> Path:
    return DATA / name / ("mu_" + f"{mu:.2f}".replace(".", "p"))


def make_lfr(name: str, mu: float, config: dict):
    folder = dataset_dir(name, mu)
    edge_file, label_file, meta_file = folder / "edges.txt", folder / "labels.csv", folder / "metadata.json"
    if edge_file.exists() and label_file.exists() and meta_file.exists():
        return read_edge_label_files(edge_file, label_file), json.loads(meta_file.read_text(encoding="utf-8"))

    folder.mkdir(parents=True, exist_ok=True)
    graph = nx.LFR_benchmark_graph(
        config["n"], TAU1, TAU2, mu,
        average_degree=config["average_degree"], max_degree=config["max_degree"],
        min_community=config["min_community"], max_community=config["max_community"],
        max_iters=1000, seed=SEED,
    )
    groups = sorted({frozenset(graph.nodes[v]["community"]) for v in graph}, key=lambda group: min(group))
    labels = {v: str(i) for i, group in enumerate(groups) for v in group}
    with edge_file.open("w", encoding="utf-8") as f:
        for u, v in graph.edges():
            if u < v:
                f.write(f"{u} {v}\n")
    with label_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for v in sorted(graph.nodes()):
            writer.writerow([v, labels[v]])
    metadata = {
        "dataset": name, "mu": mu, "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(), "n_communities": len(groups),
        "tau1_assumed": TAU1, "tau2_assumed": TAU2, "seed": SEED,
        "generator": "networkx.LFR_benchmark_graph",
        "requested_parameters": config,
    }
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return read_edge_label_files(edge_file, label_file), metadata


def score(truth, predicted):
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    true_vals, true_ids = np.unique(truth, return_inverse=True)
    pred_vals, pred_ids = np.unique(predicted, return_inverse=True)
    contingency = np.zeros((len(true_vals), len(pred_vals)), dtype=np.int64)
    np.add.at(contingency, (true_ids, pred_ids), 1)
    c2 = lambda x: x * (x - 1) // 2
    tp = int(c2(contingency).sum())
    same_p, same_t = int(c2(contingency.sum(axis=0)).sum()), int(c2(contingency.sum(axis=1)).sum())
    fp, fn = same_p - tp, same_t - tp
    f_measure = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    fmi = np.sqrt(tp * tp / (same_p * same_t)) if same_p and same_t else 0.0
    row, col = linear_sum_assignment(-contingency)
    coverage = float(contingency[row, col].sum() / len(truth))
    from sklearn.metrics import normalized_mutual_info_score
    return {"NMI": float(normalized_mutual_info_score(truth, predicted)),
            "F_measure": float(f_measure), "Jaccard": float(jaccard),
            "FMI": float(fmi), "Coverage_hungarian_accuracy": coverage,
            "n_communities_true": len(true_vals), "n_communities_predicted": len(pred_vals)}


def deepwalk_ppmi_embedding(graph: nx.Graph, dimension: int, seed: int):
    """Small, explicit DeepWalk-style baseline using walks, PPMI and SVD."""
    rng = np.random.RandomState(seed)
    node_count = graph.number_of_nodes()
    neighbors = [np.asarray(sorted(graph.neighbors(v)), dtype=np.int64)
                 for v in range(node_count)]
    counts = {}
    for _ in range(5):
        for start in rng.permutation(node_count):
            walk = [int(start)]
            for _step in range(19):
                adjacent = neighbors[walk[-1]]
                if adjacent.size == 0:
                    break
                walk.append(int(rng.choice(adjacent)))
            for position, center in enumerate(walk):
                left, right = max(0, position - 5), min(len(walk), position + 6)
                for context_position in range(left, right):
                    if context_position == position:
                        continue
                    pair = (center, walk[context_position])
                    counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        return np.zeros((node_count, 1), dtype=np.float64)
    rows = np.fromiter((pair[0] for pair in counts), dtype=np.int64)
    cols = np.fromiter((pair[1] for pair in counts), dtype=np.int64)
    values = np.fromiter(counts.values(), dtype=np.float64)
    row_sum = np.bincount(rows, weights=values, minlength=node_count)
    col_sum = np.bincount(cols, weights=values, minlength=node_count)
    total = float(values.sum())
    ppmi = np.log(values * total / (row_sum[rows] * col_sum[cols]))
    keep = ppmi > 0
    matrix = sp.csr_matrix((ppmi[keep], (rows[keep], cols[keep])),
                           shape=(node_count, node_count))
    effective_dim = min(dimension, node_count - 1)
    if effective_dim < 1 or matrix.nnz == 0:
        return np.zeros((node_count, 1), dtype=np.float64)
    return TruncatedSVD(n_components=effective_dim, random_state=seed).fit_transform(matrix)


def run_reproducible_baselines(lfr_cache):
    """Rerun accessible classical/embedding baselines on identical graph snapshots."""
    folder = RESULTS / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "baseline_metrics.json"
    previous = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
    results = [row for row in previous if row.get("preprocessing") == BASELINE_PREPROCESSING]
    completed = {(row["dataset"], row["method"]) for row in results}

    def save_results():
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        fields = ["dataset", "method", "n_nodes", "n_edges", "n_communities_true",
                  "n_communities_predicted_mean", "NMI", "NMI_std", "F_measure", "F_measure_std",
                  "Jaccard", "Jaccard_std", "FMI", "FMI_std", "Coverage_hungarian_accuracy",
                  "Coverage_hungarian_accuracy_std", "seconds"]
        with (folder / "baseline_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    datasets = []
    for name, config in LFR_CASES.items():
        if name not in ("Syn1", "Syn2"):
            continue
        for mu in config["mus"]:
            data, _metadata = lfr_cache[(name, mu)]
            datasets.append((name + f"_mu{mu:.2f}", data))
    for name in ("football", "karate", "dolphins", "email-Eu-core"):
        datasets.append((name, read_real(name)))

    for dataset, (nodes, edges, truth) in datasets:
        edges = list(edges)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from((int(u), int(v)) for u, v in edges)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        adjacency = normalize_adjacency(graph_to_adjacency(graph.edges(), len(nodes)))
        k = len(set(truth))
        methods = {
            "Greedy modularity": (1, lambda seed: _partition_labels(
                nx.algorithms.community.greedy_modularity_communities(graph), len(nodes))),
            "Asynchronous LPA": (len(REPEAT_SEEDS), lambda seed: _partition_labels(
                nx.algorithms.community.asyn_lpa_communities(graph, seed=seed), len(nodes))),
            "Spectral clustering": (len(REPEAT_SEEDS), lambda seed: SpectralClustering(
                n_clusters=k, affinity="precomputed", assign_labels="kmeans",
                n_init=10, random_state=seed).fit_predict(adjacency.toarray())),
            "DeepWalk-PPMI-SVD": (len(REPEAT_SEEDS), lambda seed: KMeans(
                n_clusters=k, n_init=10, random_state=seed).fit_predict(
                    deepwalk_ppmi_embedding(graph, 100, seed))),
        }
        for method, (repeat_count, predict) in methods.items():
            if (dataset, method) in completed:
                continue
            seeds = REPEAT_SEEDS[:repeat_count]
            samples = []
            start = time.perf_counter()
            for seed in seeds:
                prediction = predict(seed)
                samples.append(score(truth, prediction))
            elapsed = time.perf_counter() - start
            row = {"suite": "reproduced_baseline", "dataset": dataset,
                   "method": method, "repeat_seeds": seeds, "seconds": elapsed,
                   "preprocessing": BASELINE_PREPROCESSING,
                   "n_nodes": len(nodes), "n_edges": int(adjacency.nnz // 2)}
            for key in ("NMI", "F_measure", "Jaccard", "FMI", "Coverage_hungarian_accuracy"):
                values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
                row[key] = float(values.mean())
                row[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row["n_communities_true"] = k
            row["n_communities_predicted_mean"] = float(np.mean(
                [sample["n_communities_predicted"] for sample in samples]))
            results.append(row)
            completed.add((dataset, method))
            save_results()
            print(f"Completed baseline: {dataset} / {method}", flush=True)
    return len(results)


def _partition_labels(communities, n_nodes):
    labels = np.full(n_nodes, -1, dtype=np.int64)
    for label, community in enumerate(communities):
        for node in community:
            labels[int(node)] = label
    if np.any(labels < 0):
        labels[labels < 0] = labels.max() + 1 + np.arange(np.count_nonzero(labels < 0))
    return labels


def fit_one(name: str, nodes, edges, truth, params: dict, suite: str, config: dict | None = None):
    adjacency = normalize_adjacency(graph_to_adjacency(edges, len(nodes)))
    model = BatchICIDCluster(**params)
    start = time.perf_counter()
    _, embeddings = model.fit_predict(adjacency, len(set(truth)))
    feature_seconds = time.perf_counter() - start
    replicate_metrics = []
    labels = None
    kmeans_start = time.perf_counter()
    for seed in REPEAT_SEEDS:
        prediction = KMeans(n_clusters=len(set(truth)), random_state=seed,
                            n_init=10).fit_predict(embeddings)
        if seed == REPEAT_SEEDS[0]:
            labels = prediction
        replicate_metrics.append(score(truth, prediction))
    repeat_seconds = time.perf_counter() - kmeans_start
    metrics = {}
    for key in ("NMI", "F_measure", "Jaccard", "FMI", "Coverage_hungarian_accuracy"):
        values = np.asarray([result[key] for result in replicate_metrics], dtype=np.float64)
        metrics[key] = float(values.mean())
        metrics[key + "_std"] = float(values.std(ddof=1))
    metrics.update({"dataset": name, "suite": suite, "seconds": feature_seconds,
                    "repeat_kmeans_seconds": repeat_seconds,
                    "effective_dim": int(embeddings.shape[1]),
                    "repeat_seeds": REPEAT_SEEDS,
                    "params": params, "generator": config or {}})
    metrics.update({key: value for key, value in replicate_metrics[0].items()
                    if key.startswith("n_communities_")})
    return metrics, labels, embeddings


def write_run(metrics: dict, labels=None, embeddings=None):
    folder = result_folder(metrics["suite"], metrics["dataset"], metrics["params"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if labels is not None:
        np.save(folder / "predictions.npy", labels)
    if embeddings is not None:
        np.save(folder / "embeddings.npy", embeddings)


def result_folder(suite: str, dataset: str, params: dict) -> Path:
    key = params.get("initialization", "pca") + "_" + params.get("threshold_mode", "dynamic") + "_" + params.get("scope_mode", "local")
    key += "_d" + str(params.get("dim", 100))
    return RESULTS / suite / dataset / key


def saved_result(suite: str, dataset: str, params: dict):
    path = result_folder(suite, dataset, params) / "metrics.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    embedding_path = path.parent / "embeddings.npy"
    if embedding_path.exists():
        result["effective_dim"] = int(np.load(embedding_path, mmap_mode="r").shape[1])
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result if len(result.get("repeat_seeds", [])) == len(REPEAT_SEEDS) else None


def read_real(name: str):
    if name == "email-Eu-core":
        return read_edge_label_files(PROJECT_DIR / "data" / "email-Eu-core.txt.gz",
                                     PROJECT_DIR / "data" / "email-Eu-core-department-labels.txt.gz")
    if name == "football":
        return read_gml_dataset(PROJECT_DIR / "data" / "football" / "football.gml", "value")
    if name == "karate":
        return read_edge_label_files(PROJECT_DIR / "data" / "karate" / "edges.txt",
                                     PROJECT_DIR / "data" / "karate" / "labels.csv")
    return read_gml_dataset(PROJECT_DIR / "data" / "dolphins" / "dolphins.gml", "groundTruth")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    lfr_cache = {}

    # Chapter 5 comparative batch results for all synthetic conditions.
    for name, config in LFR_CASES.items():
        for mu in config["mus"]:
            data, metadata = make_lfr(name, mu, config)
            lfr_cache[(name, mu)] = (data, metadata)
            nodes, edges, truth = data
            params = dict(BASE)
            dataset = name + f"_mu{mu:.2f}"
            result = saved_result("comparative_synthetic", dataset, params)
            if result is None:
                result, labels, embeddings = fit_one(dataset, nodes, edges, truth, params,
                                                      "comparative_synthetic", metadata)
            else:
                labels = embeddings = None
            result["n_nodes"] = len(nodes)
            result["n_edges"] = int(normalize_adjacency(graph_to_adjacency(edges, len(nodes))).nnz // 2)
            if labels is not None:
                write_run(result, labels, embeddings)
            all_metrics.append(result)

    # Batch method on all four real networks (Chapter 5 Tables 5-5/5-6).
    for name in ("football", "karate", "dolphins", "email-Eu-core"):
        nodes, edges, truth = read_real(name)
        params = dict(BASE)
        result = saved_result("comparative_real", name, params)
        if result is None:
            result, labels, embeddings = fit_one(name, nodes, edges, truth, params, "comparative_real")
        else:
            labels = embeddings = None
        result["n_nodes"] = len(nodes)
        result["n_edges"] = int(normalize_adjacency(graph_to_adjacency(edges, len(nodes))).nnz // 2)
        if labels is not None:
            write_run(result, labels, embeddings)
        all_metrics.append(result)

    # Ablations corresponding to the variant families in Figure 5-6.
    ablations = {
        "BRDL": ("random", "dynamic", "local"),
        "BNSL": ("ones", "static", "local"),
        "BNDF": ("ones", "dynamic", "global"),
        "BNSF": ("ones", "static", "global"),
        "BNDL": ("ones", "dynamic", "local"),
        "BPSL": ("pca", "static", "local"),
        "BSSL": ("schmidt", "static", "local"),
        "BSDL": ("schmidt", "dynamic", "local"),
        "BPDF": ("pca", "dynamic", "global"),
        "Batch-ICID": ("pca", "dynamic", "local"),
    }
    for mu in (0.05, 0.1, 0.15, 0.2):
        nodes, edges, truth = lfr_cache[("Syn2", mu)][0]
        for variant, (initialization, threshold_mode, scope_mode) in ablations.items():
            params = dict(BASE, initialization=initialization, threshold_mode=threshold_mode,
                          scope_mode=scope_mode)
            dataset = "Syn2_mu" + f"{mu:.2f}" + "_" + variant
            result = saved_result("ablation_syn2", dataset, params)
            if result is None:
                result, labels, embeddings = fit_one(dataset, nodes, edges, truth, params, "ablation_syn2")
            else:
                labels = embeddings = None
            result["variant"] = variant
            if labels is not None:
                write_run(result, labels, embeddings)
            all_metrics.append(result)

    # Dimensional sensitivity on Syn1 at mu=.1 (Figure 5-7).
    nodes, edges, truth = lfr_cache[("Syn1", 0.1)][0]
    for dim in range(100, 1001, 100):
        params = dict(BASE, dim=dim)
        dataset = f"Syn1_mu0.10_dim{dim}"
        result = saved_result("sensitivity_syn1", dataset, params)
        if result is None:
            result, labels, embeddings = fit_one(dataset, nodes, edges, truth, params,
                                                  "sensitivity_syn1")
        else:
            labels = embeddings = None
        result["configured_dim"] = dim
        if labels is not None:
            write_run(result, labels, embeddings)
        all_metrics.append(result)

    baseline_count = run_reproducible_baselines(lfr_cache)
    summary = RESULTS / "all_metrics.json"
    summary.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    fields = ["suite", "dataset", "variant", "n_nodes", "n_edges", "n_communities_true",
              "n_communities_predicted", "configured_dim", "effective_dim", "NMI", "F_measure", "Jaccard", "FMI",
              "Coverage_hungarian_accuracy", "NMI_std", "F_measure_std", "Jaccard_std", "FMI_std",
              "Coverage_hungarian_accuracy_std", "seconds", "repeat_kmeans_seconds"]
    with (RESULTS / "all_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)
    print(json.dumps({"result_count": len(all_metrics), "summary": str(summary),
                      "summary_csv": str(RESULTS / "all_metrics.csv"),
                      "baseline_result_count": baseline_count,
                      "baseline_summary_csv": str(RESULTS / "baselines" / "baseline_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
