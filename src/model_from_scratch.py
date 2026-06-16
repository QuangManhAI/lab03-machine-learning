import numpy as np
import warnings

class KMeans:
    """K-Means clustering from scratch."""
    def __init__(self, n_clusters=8, init='random', max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        
        # Guard rails
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")
            
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(self.n_init):
            if self.init == 'random':
                indices = rng.choice(n_samples, self.n_clusters, replace=False)
                centers = X[indices].copy()
            else:
                indices = rng.choice(n_samples, self.n_clusters, replace=False)
                centers = X[indices].copy()
                
            labels = np.zeros(n_samples, dtype=int)
            
            for iter_idx in range(self.max_iter):
                # E-step: Assign points to nearest centroid
                distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                new_labels = np.argmin(distances, axis=1)
                
                # M-step: Update centroids
                new_centers = np.zeros_like(centers)
                for k in range(self.n_clusters):
                    members = X[new_labels == k]
                    if len(members) > 0:
                        new_centers[k] = members.mean(axis=0)
                    else:
                        new_centers[k] = X[rng.choice(n_samples)]
                        
                # Check convergence
                center_shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                labels = new_labels
                if center_shift < self.tol:
                    break
            
            # Compute inertia
            distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            min_dist = np.min(distances, axis=1)
            inertia = np.sum(min_dist ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)


class KMedoids:
    """K-Medoids clustering from scratch (PAM-like implementation)."""
    def __init__(self, n_clusters=8, metric='euclidean', max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        
        self.medoids_ = None
        self.cluster_centers_ = None  # Alias for compatibility
        self.labels_ = None
        self.inertia_ = None

    def _compute_distances(self, X, centers):
        if self.metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis, :] - centers[np.newaxis, :, :]), axis=2)
        else:
            return np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")
            
        best_inertia = np.inf
        best_medoids = None
        best_labels = None
        
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(self.n_init):
            medoid_indices = rng.choice(n_samples, self.n_clusters, replace=False)
            medoids = X[medoid_indices].copy()
            
            labels = np.zeros(n_samples, dtype=int)
            
            for iter_idx in range(self.max_iter):
                distances = self._compute_distances(X, medoids)
                new_labels = np.argmin(distances, axis=1)
                
                new_medoids = np.zeros_like(medoids)
                new_medoid_indices = np.zeros(self.n_clusters, dtype=int)
                
                for k in range(self.n_clusters):
                    cluster_indices = np.where(new_labels == k)[0]
                    if len(cluster_indices) > 0:
                        cluster_points = X[cluster_indices]
                        if self.metric == 'manhattan':
                            dist_matrix = np.sum(np.abs(cluster_points[:, np.newaxis, :] - cluster_points[np.newaxis, :, :]), axis=2)
                        else:
                            dist_matrix = np.linalg.norm(cluster_points[:, np.newaxis, :] - cluster_points[np.newaxis, :, :], axis=2)
                        
                        sum_dists = np.sum(dist_matrix, axis=1)
                        best_idx_in_cluster = np.argmin(sum_dists)
                        new_medoid_indices[k] = cluster_indices[best_idx_in_cluster]
                        new_medoids[k] = X[new_medoid_indices[k]]
                    else:
                        rand_idx = rng.choice(n_samples)
                        new_medoid_indices[k] = rand_idx
                        new_medoids[k] = X[rand_idx]
                
                if np.array_equal(np.sort(medoid_indices), np.sort(new_medoid_indices)):
                    break
                medoids = new_medoids
                medoid_indices = new_medoid_indices
                labels = new_labels
                
            # Compute total distance (inertia)
            distances = self._compute_distances(X, medoids)
            min_dist = np.min(distances, axis=1)
            inertia = np.sum(min_dist)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_medoids = medoids.copy()
                best_labels = labels.copy()
                
        self.medoids_ = best_medoids
        self.cluster_centers_ = best_medoids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = self._compute_distances(X, self.medoids_)
        return np.argmin(distances, axis=1)


class HierarchicalClustering:
    """Agglomerative Hierarchical Clustering from scratch using Lance-Williams update formula."""
    def __init__(self, n_clusters=2, linkage='average'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")
            
        # Compute pairwise distance matrix (Euclidean)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        sizes = np.ones(n_samples, dtype=int)
        members = [[i] for i in range(n_samples)]
        active_clusters = set(range(n_samples))
        
        for _ in range(n_samples - self.n_clusters):
            active_list = list(active_clusters)
            sub_matrix = dist_matrix[np.ix_(active_list, active_list)]
            
            min_idx = np.argmin(sub_matrix)
            r_idx, c_idx = np.unravel_index(min_idx, sub_matrix.shape)
            
            u = active_list[r_idx]
            v = active_list[c_idx]
            
            if u > v:
                u, v = v, u
                
            size_u = sizes[u]
            size_v = sizes[v]
            
            # Update distances to remaining active clusters via Lance-Williams
            for k in active_clusters:
                if k != u and k != v:
                    if self.linkage == 'single':
                        new_d = min(dist_matrix[u, k], dist_matrix[v, k])
                    elif self.linkage == 'complete':
                        new_d = max(dist_matrix[u, k], dist_matrix[v, k])
                    elif self.linkage == 'average':
                        new_d = (dist_matrix[u, k] * size_u + dist_matrix[v, k] * size_v) / (size_u + size_v)
                    else:
                        raise ValueError(f"Unsupported linkage: {self.linkage}")
                    dist_matrix[u, k] = dist_matrix[k, u] = new_d
                    
            dist_matrix[u, u] = np.inf
            active_clusters.remove(v)
            sizes[u] += size_v
            members[u].extend(members[v])
            members[v] = []
            
        labels = np.zeros(n_samples, dtype=int)
        for label_idx, cluster_idx in enumerate(sorted(active_clusters)):
            for member in members[cluster_idx]:
                labels[member] = label_idx
                
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# Library wrappers for verification and testing
class KMeansLibrary:
    """Wrapper around sklearn.cluster.KMeans."""
    def __init__(self, n_clusters=8, init='random', max_iter=300, tol=1e-4, random_state=None, n_init=10):
        from sklearn.cluster import KMeans as SKKMeans
        self.estimator = SKKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=n_init
        )
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        self.estimator.fit(X)
        self.cluster_centers_ = self.estimator.cluster_centers_
        self.labels_ = self.estimator.labels_
        self.inertia_ = self.estimator.inertia_
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        return self.estimator.predict(X)


class HierarchicalClusteringLibrary:
    """Wrapper around sklearn.cluster.AgglomerativeClustering."""
    def __init__(self, n_clusters=2, linkage='average'):
        from sklearn.cluster import AgglomerativeClustering as SKAgglomerativeClustering
        self.estimator = SKAgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        self.labels_ = None

    def fit(self, X):
        self.estimator.fit(X)
        self.labels_ = self.estimator.labels_
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class KMedoidsLibrary:
    """Wrapper around sklearn_extra.cluster.KMedoids with fallback to scratch."""
    def __init__(self, n_clusters=8, metric='euclidean', max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        
        self.estimator = None
        self.medoids_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        
        try:
            from sklearn_extra.cluster import KMedoids as SKKMedoids
            self.estimator = SKKMedoids(
                n_clusters=self.n_clusters,
                metric=self.metric,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                init='random'
            )
        except ImportError:
            warnings.warn("sklearn_extra is not installed. KMedoidsLibrary will fallback to KMedoids (scratch).")
            self.estimator = KMedoids(
                n_clusters=self.n_clusters,
                metric=self.metric,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_init=self.n_init
            )

    def fit(self, X):
        self.estimator.fit(X)
        if hasattr(self.estimator, 'medoids_'):
            self.medoids_ = self.estimator.medoids_
            self.cluster_centers_ = self.estimator.cluster_centers_
            self.labels_ = self.estimator.labels_
            self.inertia_ = self.estimator.inertia_
        else:
            # Fallback to scratch attributes
            self.medoids_ = self.estimator.medoids_
            self.cluster_centers_ = self.estimator.cluster_centers_
            self.labels_ = self.estimator.labels_
            self.inertia_ = self.estimator.inertia_
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        return self.estimator.predict(X)
