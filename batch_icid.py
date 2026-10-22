from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.cluster import KMeans


@dataclass
class BatchICIDCluster:
    """Mean-field batch independent-cascade community detector.

    At each step, edge transmissions are combined in log space. This avoids
    the degenerate all-rows-identical cascade matrix in the manuscript's
    printed equations and makes the dynamic threshold affect activation.
    """

    steps: int = 8
    neighbor_max: int = 1000
    beta0: float = 1.0
    dim: int = 100
    random_state: int = 42
    n_init: int = 10
    initialization: str = "pca"
    threshold_mode: str = "dynamic"
    scope_mode: str = "local"

    def fit_predict(self, adjacency: sp.spmatrix, n_communities: int) -> tuple[np.ndarray, np.ndarray]:
        adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        n = adjacency.shape[0]
        if n < 2:
            raise ValueError("at least two nodes are required")
        if not 1 <= n_communities < n:
            raise ValueError("n_communities must be between 1 and n_nodes - 1")
        if self.steps < 1 or self.neighbor_max < 1 or self.dim < 1:
            raise ValueError("steps, neighbor_max and dim must be positive")
        if not 0.0 <= self.beta0 <= 1.0:
            raise ValueError("beta0 must be a probability in [0, 1]")
        if self.initialization not in {"pca", "random", "ones", "schmidt"}:
            raise ValueError("initialization must be pca, random, ones or schmidt")
        if self.threshold_mode not in {"dynamic", "static"}:
            raise ValueError("threshold_mode must be dynamic or static")
        if self.scope_mode not in {"local", "global"}:
            raise ValueError("scope_mode must be local or global")

        adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        adjacency.data[:] = 1.0

        effective_dim = min(self.dim, n - 1)
        rng = np.random.RandomState(self.random_state)
        profile_rows, profile_cols, profile_data = [], [], []
        for source in range(n):
            scope = (list(range(n)) if self.scope_mode == "global"
                     else self._bounded_bfs(adjacency, source))
            local_adj = adjacency[scope][:, scope].tocsr()
            local_degrees = np.asarray(local_adj.sum(axis=1)).ravel()
            local_index = {node: i for i, node in enumerate(scope)}
            frontier = np.zeros(len(scope), dtype=np.float64)
            frontier[local_index[source]] = 1.0
            cumulative = frontier.copy()
            local_vec = frontier.copy()

            for t in range(1, self.steps + 1):
                # The printed recurrence in thesis Eq. (5-1) subtracts
                # (t-1)/T cumulatively, which becomes negative before T=8.
                # We use the bounded schedule described in the text.
                beta_t = (self.beta0 * (1.0 - (t - 1.0) / self.steps)
                          if self.threshold_mode == "dynamic" else self.beta0)

                # Independent-cascade mean-field update. For node v, the
                # probability of receiving no successful transmission from
                # the previous frontier is prod_u(1 - beta_t * frontier[u])
                # over neighbors u. A.T @ log(1 - beta_t * frontier) computes
                # this for all nodes at once. The factor (1-cumulative[v])
                # excludes nodes already activated in an earlier step.
                safe_probability = np.minimum(beta_t * frontier, 1.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_no_transmission = np.log1p(-safe_probability)
                log_no_transmission[~np.isfinite(log_no_transmission)] = -np.inf
                log_no_transmission[local_degrees == 0] = 0.0
                neighbor_log_no = local_adj.T @ log_no_transmission
                activation_probability = -np.expm1(neighbor_log_no)
                activation_probability = np.clip(activation_probability, 0.0, 1.0)
                next_frontier = (1.0 - cumulative) * activation_probability
                local_vec += next_frontier
                cumulative = np.clip(cumulative + next_frontier, 0.0, 1.0)
                frontier = next_frontier
                if not np.any(frontier > 1e-15):
                    break

            scope_array = np.asarray(scope, dtype=np.int64)
            nonzero = np.flatnonzero(local_vec)
            profile_rows.extend([source] * len(nonzero))
            profile_cols.extend(scope_array[nonzero].tolist())
            profile_data.extend(local_vec[nonzero].tolist())

        # PCA is applied to the centered source-by-node cascade-profile
        # matrix, rather than to I_n. This makes the retained subspace reflect
        # variation in the actual diffusion features and avoids the isotropic
        # identity-PCA ambiguity raised by the reviewers. The covariance
        # operator is evaluated sparsely, without densifying the n x n matrix.
        profiles = sp.csr_matrix((profile_data, (profile_rows, profile_cols)),
                                 shape=(n, n), dtype=np.float64)
        if self.initialization == "pca":
            mean = np.asarray(profiles.mean(axis=0)).ravel()

            def covariance_product(vector):
                projected = profiles.dot(vector)
                result = profiles.T.dot(projected)
                result = np.asarray(result).ravel()
                result -= n * mean * float(np.dot(mean, vector))
                return result

            covariance = LinearOperator((n, n), matvec=covariance_product,
                                        rmatvec=covariance_product, dtype=np.float64)
            initial_vector = rng.normal(size=n)
            eigenvalues, components = eigsh(covariance, k=effective_dim, which="LM",
                                             v0=initial_vector, tol=1e-7)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, components = eigenvalues[order], components[:, order]
            rank = int(np.count_nonzero(eigenvalues > max(float(eigenvalues[0]), 0.0) * 1e-10))
            if rank:
                components = components[:, :rank]
                features = profiles.dot(components) - np.outer(np.ones(n), mean @ components)
            else:
                features = np.zeros((n, 1), dtype=np.float64)
        else:
            if self.initialization == "random":
                projection = rng.normal(size=(n, effective_dim))
            elif self.initialization == "ones":
                projection = np.ones((n, effective_dim), dtype=np.float64)
            else:
                projection, _ = np.linalg.qr(rng.normal(size=(n, effective_dim)), mode="reduced")
            features = profiles.dot(projection)

        labels = KMeans(n_clusters=n_communities, random_state=self.random_state,
                        n_init=self.n_init).fit_predict(features)
        return labels, features

    def _bounded_bfs(self, adjacency: sp.csr_matrix, source: int) -> list[int]:
        """Source-first deterministic BFS, capped at neighbor_max scope nodes."""
        cap = min(self.neighbor_max, adjacency.shape[0])
        seen = {source}
        ordered = [source]
        queue = deque([source])
        while queue and len(ordered) < cap:
            u = queue.popleft()
            start, end = adjacency.indptr[u], adjacency.indptr[u + 1]
            for v in adjacency.indices[start:end]:
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    ordered.append(v)
                    if len(ordered) == cap:
                        break
                    queue.append(v)
        return ordered


def graph_to_adjacency(edges: Iterable[tuple[int, int]], n_nodes: int) -> sp.csr_matrix:
    rows, cols = [], []
    for u, v in edges:
        rows.extend((int(u), int(v)))
        cols.extend((int(v), int(u)))
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


