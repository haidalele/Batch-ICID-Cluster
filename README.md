# Batch-ICID-Cluster reproduction package

This package contains the revised mean-field implementation used for the CMC major-revision experiments.

## Install

```powershell
py -3.8 -m pip install -r requirements.txt
```

## Single-dataset CLI

The default data paths are resolved relative to the folder containing `run.py`.

```powershell
py -3.8 run.py
py -3.8 run.py --dataset email
py -3.8 run.py --dataset football
```

The package also includes Karate, Dolphins, and the email-Eu-core data files. For custom networks, pass `--edges` and `--labels`; labels are used only to set K and evaluate the predicted partition. Relative paths are resolved from the project folder.

## Full revised experiments

```powershell
py -3.8 -u run_all_experiments.py
```

The runner executes 67 Batch-ICID configurations: 13 LFR conditions, four real networks, 40 ablations, and 10 target dimensions. It also reruns 52 baseline configurations on Syn1, Syn2, and all four real networks: Greedy modularity, asynchronous label propagation, spectral clustering, and a DeepWalk-style PPMI-SVD baseline. Random baselines use ten seeds; Greedy modularity is deterministic. All methods receive the same binary, undirected, self-loop-free graph snapshot and the same reference partition.

LFR graphs and labels are saved under `data/lfr/` using NetworkX with seed 42, tau1=3, tau2=1.5, and the listed node, degree, community-size, and mixing settings. These are generated instances because the manuscript did not supply the original LFR files or seeds. Exact graph statistics and realized community counts are recorded in each metadata file and in the result table.

Results are written under `results/chapter5_review_revision/`. `all_metrics.csv` and `all_metrics.json` contain the 67 proposed-method runs, including mean and sample standard deviation over ten K-means initializations, effective PCA dimension, graph statistics, and runtime. Baseline results are saved incrementally in `baselines/baseline_metrics.csv` and `baselines/baseline_metrics.json`; interrupted runs resume completed dataset/method pairs. Standard deviations quantify K-means initialization sensitivity on a fixed embedding, not variability across independently generated graphs or diffusion samples.

## Revised method

For each source node, the method builds a bounded BFS scope and estimates first-activation probabilities with a mean-field independent-cascade update. The bounded edge-transmission schedule is `beta_t = beta0 * (1 - (t-1)/L)`, and a cumulative activation probability prevents repeated counting. The sparse source-by-node expected-reachability profile matrix is centered and reduced with data-dependent PCA. The effective dimension is capped by the configured target and numerical rank; the runner records the retained embedding width. K-means produces the final partition.

The current default configuration is `steps=8`, `neighbor_max=1000`, `beta0=1`, and `dim=100`. The revised implementation does not claim that the proposed method is more accurate than the tested baselines. Round-ICID is discussed as the predecessor but is not numerically compared because a compatible executable implementation and exact source graph snapshots were unavailable. Leiden, Infomap, node2vec, and learned GNN/Transformer methods are not included.

Runtime refers to feature construction and PCA; the separate repeated K-means evaluation time is also recorded. The paper gives asymptotic time and working-space bounds, but this package does not record controlled peak memory.
