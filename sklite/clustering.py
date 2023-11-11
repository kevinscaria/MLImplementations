import numpy as np
import pandas as pd
from sklite.utils import NList

# class KMeansEstimator:
#     def __init__(self, mu_k, distortion, n_iter) -> None:
#         self.mu_k = mu_k
#         self.distortion = distortion
#         self.n_iter = n_iter


class KMeans:
    def __init__(self, 
                 n_clusters=8, 
                 max_iter=10, 
                 verbose=0, 
                 random_state=None
                 ) -> None:
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.verbose=verbose
        self.random_state=random_state
        self.n_list = NList(4)
        self.mu_k = None
        self.n_iter = None
        self.distortion = 0
        self.estimator = None


    def fit(self, X_train):
        """
        This method minimizes the KMeans objective function.

        Parameters
        ----------
        X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """

        # Set Random Seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Convert X_train to numpy array if dataframe
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()

        # Set Default Values
        n_samples = X_train.shape[0]
        prev_n_distortions_changed = True
        self.n_iter = 0

        # STEP1: Select n_clusters random points as self.mu_k
        index = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.mu_k = X_train[index] ## mu_k is the cluster centers

        while prev_n_distortions_changed and self.n_iter < self.max_iter:
            # STEP2: Given self.mu_k, optimize the r_nk
            ## Iterate through all n_samples and find the r_nk value for each, 
            ## where, n = n_samples and k = n_clusters
            r = np.empty(shape=(n_samples, self.n_clusters))

            for n_val in range(n_samples):
                for k_val in range(self.n_clusters):
                    r[n_val][k_val] = np.linalg.norm(X_train[n_val] - self.mu_k[k_val])

            # Get the k index that has the minimum distance value
            min_dist_idx = np.argmin(r, axis=1)

            # Create a Matrix of Zeros like the r_nk matrix
            binary_r = np.zeros_like(r)   

            # Convert the index with min. distance as one
            binary_r[np.arange(n_samples), min_dist_idx] = 1

            r = binary_r
            del binary_r

            # STEP3: Given r_nk, optimize the mu_k
            self.distortion = 0
            for k_val in range(self.n_clusters):
                samples_closest_to_k = r[:, k_val] == 1
                self.mu_k[k_val] = np.mean(X_train[samples_closest_to_k], axis=0)
                self.distortion+=np.linalg.norm(X_train[samples_closest_to_k] - self.mu_k[k_val])
            
            if self.verbose:
                print(f"Num iterations: {self.n_iter} | Distortion: {self.distortion}", flush=True)

            self.n_list.append(self.distortion)
            prev_n_distortions_changed = not self.n_list.is_same()
            self.n_iter+=1

        # Return KMeansEstimator
        return self
        
    def __repr__(self, ):
        return f"""KMeans(mu_k={self.mu_k}, distortion={self.distortion}, n_iter={self.n_iter},
        n_clusters={self.n_clusters}, max_iter={self.max_iter}, verbose={self.verbose},
        random_state={self.random_state})
         """

    def predict(self, X):
        # Convert X_train to numpy array if dataframe
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Set Default Values
        n_samples = X.shape[0]
        r = np.empty(shape=(n_samples, self.n_clusters))

        for n_val in range(n_samples):
            for k_val in range(self.n_clusters):
                r[n_val][k_val] = np.linalg.norm(X[n_val] - self.mu_k[k_val])

        # Return labels 
        return np.argmin(r, axis=1)

    def score(self, ):
        pass

    def transform(self, ):
        pass




        