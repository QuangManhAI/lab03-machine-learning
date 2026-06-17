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


class GaussianMixtureModel:
    """Gaussian Mixture Model clustering from scratch using EM algorithm."""
    def __init__(self, n_components=4, max_iter=100, tol=1e-4, random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg_covar = reg_covar
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.labels_ = None

    def _multivariate_normal_pdf(self, X, mean, cov):
        n_samples, n_features = X.shape
        cov_reg = cov + np.eye(n_features) * self.reg_covar
        try:
            inv_cov = np.linalg.inv(cov_reg)
            det_cov = np.linalg.det(cov_reg)
        except np.linalg.LinAlgError:
            cov_reg += np.eye(n_features) * 1e-4
            inv_cov = np.linalg.inv(cov_reg)
            det_cov = np.linalg.det(cov_reg)
            
        if det_cov <= 0:
            det_cov = 1e-30
            
        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** n_features) * det_cov)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        return norm_const * np.exp(exponent)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            pdfs = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                pdfs[:, k] = self._multivariate_normal_pdf(X, self.means_[k], self.covariances_[k])
                
            weighted_pdfs = pdfs * self.weights_
            sum_weighted_pdfs = np.sum(weighted_pdfs, axis=1, keepdims=True)
            sum_weighted_pdfs = np.where(sum_weighted_pdfs == 0, 1e-30, sum_weighted_pdfs)
            responsibilities = weighted_pdfs / sum_weighted_pdfs
            
            log_likelihood = float(np.sum(np.log(np.maximum(np.sum(weighted_pdfs, axis=1), 1e-30))))
            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood
            
            Nk = np.sum(responsibilities, axis=0)
            Nk_safe = np.where(Nk == 0, 1e-10, Nk)
            
            self.weights_ = Nk / n_samples
            
            for k in range(self.n_components):
                self.means_[k] = np.sum(responsibilities[:, k, np.newaxis] * X, axis=0) / Nk_safe[k]
                diff = X - self.means_[k]
                self.covariances_[k] = (diff.T @ (responsibilities[:, k, np.newaxis] * diff)) / Nk_safe[k]
                
        pdfs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            pdfs[:, k] = self._multivariate_normal_pdf(X, self.means_[k], self.covariances_[k])
        self.labels_ = np.argmax(pdfs * self.weights_, axis=1)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        pdfs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            pdfs[:, k] = self._multivariate_normal_pdf(X, self.means_[k], self.covariances_[k])
        return np.argmax(pdfs * self.weights_, axis=1)


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


class GaussianMixtureModelLibrary:
    """Wrapper around sklearn.mixture.GaussianMixture."""
    def __init__(self, n_components=4, max_iter=100, tol=1e-4, random_state=None):
        from sklearn.mixture import GaussianMixture as SKGaussianMixture
        self.estimator = SKGaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        self.labels_ = None

    def fit(self, X):
        self.estimator.fit(X)
        self.labels_ = self.estimator.predict(X)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        return self.estimator.predict(X)
