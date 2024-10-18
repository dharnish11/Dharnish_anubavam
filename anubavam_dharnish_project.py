class matrix:
    def __init__(self, filename=None):
        self.array_2d = None
        if filename:
            self.load_from_csv(filename)

    def load_from_csv(self, filename):
        # Read the CSV file and load data into array_2d
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)
        self.array_2d = np.array(data, dtype=float)

    def standardise(self):
        # Standardise the array_2d
        mean = np.mean(self.array_2d, axis=0)
        min_vals = np.min(self.array_2d, axis=0)
        max_vals = np.max(self.array_2d, axis=0)
        self.array_2d = (self.array_2d - mean) / (max_vals - min_vals)

    def get_distance(self, other_matrix, row_i):
        # Euclidean distance between row_i of this matrix and all rows in other_matrix
        row = self.array_2d[row_i]
        distances = np.linalg.norm(other_matrix.array_2d - row, axis=1)
        return matrix(np.expand_dims(distances, axis=1))

    def get_weighted_distance(self, other_matrix, weights, row_i):
        # Weighted Euclidean distance between row_i and other_matrix using weights
        row = self.array_2d[row_i]
        weighted_diff = (weights.array_2d[0] * (other_matrix.array_2d - row) ** 2)
        distances = np.sqrt(np.sum(weighted_diff, axis=1))
        return matrix(np.expand_dims(distances, axis=1))

    def get_count_frequency(self):
        if self.array_2d.shape[1] != 1:
            return 0
        # Count the frequency of each element in the column
        unique, counts = np.unique(self.array_2d, return_counts=True)
        return dict(zip(unique, counts))

def get_initial_weights(m):
    # Return a matrix of random weights summing to 1
    weights = np.random.rand(1, m)
    weights /= np.sum(weights)
    return matrix(weights)

def get_centroids(data, S, K):
    # Calculate centroids as the mean of data points assigned to each cluster
    centroids = []
    for k in range(1, K + 1):
        cluster_data = data.array_2d[S.array_2d.flatten() == k]
        if len(cluster_data) > 0:
            centroids.append(np.mean(cluster_data, axis=0))
    return matrix(np.array(centroids))

def get_separation_within(data, centroids, S, K):
    # Calculate the separation within clusters
    m = data.array_2d.shape[1]
    separation_within = np.zeros((1, m))
    for j in range(m):
        for k in range(1, K + 1):
            cluster_points = data.array_2d[S.array_2d.flatten() == k]
            if len(cluster_points) > 0:
                centroid = centroids.array_2d[k - 1, j]
                distances = np.abs(cluster_points[:, j] - centroid)
                separation_within[0, j] += np.sum(distances)
    return matrix(separation_within)

def get_separation_within(data, centroids, S, K):
    # Calculate the separation within clusters
    m = data.array_2d.shape[1]
    separation_within = np.zeros((1, m))
    for j in range(m):
        for k in range(1, K + 1):
            cluster_points = data.array_2d[S.array_2d.flatten() == k]
            if len(cluster_points) > 0:
                centroid = centroids.array_2d[k - 1, j]
                distances = np.abs(cluster_points[:, j] - centroid)
                separation_within[0, j] += np.sum(distances)
    return matrix(separation_within)
