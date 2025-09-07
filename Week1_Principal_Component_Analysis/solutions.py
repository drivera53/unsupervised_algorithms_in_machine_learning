from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, target_explained_variance=None):
        """
        explained_variance: float, the target level of explained variance
        """
        self.target_explained_variance = target_explained_variance
        self.feature_size = -1

    def standardize(self, X):
        return StandardScaler().fit_transform(X)

    def compute_mean_vector(self, X_std):
            """
            compute mean vector
            :param X_std: transformed data
            :return n X 1 matrix: mean vector
            """
            ### BEGIN SOLUTION
            return np.mean(X_std, axis=0)
            ### END SOLUTION

    def compute_cov(self, X_std, mean_vec):
            """
            Covariance using mean, (don't use any numpy.cov)
            :param X_std:
            :param mean_vec:
            :return n X n matrix:: covariance matrix
            """
            return (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)

    def compute_eigen_vector(self, cov_mat):
            """
            Eigenvector and eigen values using numpy. Uses numpy's eigenvalue function
            :param cov_mat:
            :return: (eigen_values, eigen_vector)
            """
            return np.linalg.eig(cov_mat)

    def compute_explained_variance(self, eigen_vals):
            """
            sort eigen values and compute explained variance.
            explained variance informs the amount of information (variance)
            can be attributed to each of  the principal components.
            :param eigen_vals:
            :return: explained variance.
            """
            ### BEGIN SOLUTION
            tot = sum(eigen_vals)
            return [(i / tot) for i in sorted(eigen_vals, reverse=True)]
            ### END SOLUTION

    def compute_weight_matrix(self, eig_pairs, cum_var_exp):
            """
            compute weight matrix of top principal components conditioned on target
            explained variance.
            (Hint : use cumilative explained variance and target_explained_variance to find
            top components)
            
            :param eig_pairs: list of tuples containing eigenvalues and eigenvectors, 
            sorted by eigenvalues in descending order (the biggest eigenvalue and corresponding eigenvectors first).
            :param cum_var_exp: cumulative expalined variance by features
            :return: weight matrix (the shape of the weight matrix is n X k)
            """
            ### BEGIN SOLUTION
            matrix_w = np.ones((self.feature_size, 1))
            for i in range(len(eig_pairs)):
                if cum_var_exp[i] < self.target_explained_variance:
                    matrix_w = np.hstack((matrix_w,
                                        eig_pairs[i][1].reshape(self.feature_size,
                                                                1)))
            return np.delete(matrix_w, [0], axis=1).tolist()
            ### END SOLUTION

    def fit(self, X):
            """    
            entry point to the transform data to k dimensions
            standardize and compute weight matrix to transform data.
            The fit functioin returns the transformed features. k is the number of features which cumulative 
            explained variance ratio meets the target_explained_variance.
            :param   m X n dimension: train samples
            :return  m X k dimension: subspace data. 
            """
        
            self.feature_size = X.shape[1]
            
            ### BEGIN SOLUTION
            # 16 pts
            X_std = self.standardize(X) # partial: 2 pts
            #---- partial 2 pts
            mean_vec = self.compute_mean_vector(X_std)
            cov_mat = self.compute_cov(X_std, mean_vec) 
            #-------
            eig_vals, eig_vecs = self.compute_eigen_vector(cov_mat) #partial 2pts
            #----- partial 4 pts
            eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in
                        range(len(eig_vals))]
            eig_pairs.sort()
            eig_pairs.reverse()
            #-------
            var_exp = self.compute_explained_variance(eig_vals) # partial 2 pts
            cum_var_exp = self.cumulative_sum(var_exp) #partial 2pts
            matrix_w = self.compute_weight_matrix(eig_pairs=eig_pairs,cum_var_exp=cum_var_exp) #partial 2 pts
            ### END SOLUTION
            print(len(matrix_w),len(matrix_w[0]))
            return self.transform_data(X_std=X_std, matrix_w=matrix_w)