## K-Means Clustering: A Begginer Guide (Self-Study)

### Introduction
K-Means clustering is one of the simplest and most popular unsupervised learning algorithms. The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K centroids.

### Terminology and Concepts

#### 1. Centroid
A centroid is the central point of a cluster. In K-Means clustering, each cluster is represented by its centroid, which is the mean of all points in the cluster.

#### 2. Cluster
A cluster is a collection of data points grouped together because of certain similarities. The K-Means algorithm aims to partition the data into K clusters.

#### 3. Inertia
Inertia, also known as the within-cluster sum of squares (WCSS), measures how tightly the data points are packed within each cluster. It is calculated as the sum of squared distances between each point and its assigned centroid.

$$
\text{Inertia} = \sum_{i=1}^{m} \sum_{j=1}^{K} \left\| x^{(i)} - \mu_j \right\|^2
$$

where:
- $m$ is the total number of data points.
- $K$ is the number of clusters.
- $x^{(i)}$ is the $i$-th data point.
- $\mu_j$ is the centroid of the $j$-th cluster.

#### 4. Distance Metric
The distance metric commonly used in K-Means is the Euclidean distance. The Euclidean distance between two points $x$ and $y$ in n-dimensional space is given by:

$$
d(x, y) = \sqrt{\sum_{k=1}^{n} (x_k - y_k)^2}
$$

### Steps of the K-Means Algorithm

1. **Initialization**: Randomly choose K initial centroids from the data points.

2. **Assignment**: Assign each data point to the nearest centroid based on the Euclidean distance. This step partitions the data into K clusters.

3. **Update**: Recalculate the centroids by computing the mean of all data points assigned to each centroid.

4. **Repeat**: Repeat the assignment and update steps until the centroids do not change significantly, or a predefined number of iterations is reached.

### Detailed Explanation

#### 1. Initialization
In the initialization step, we select K random points from the dataset as the initial centroids. These points can be chosen randomly or using methods like K-Means++ to improve convergence.

#### 2. Assignment
In the assignment step, each data point is assigned to the nearest centroid. The Euclidean distance between each data point and all centroids is computed, and the data point is assigned to the cluster whose centroid is closest.

For each data point $x^{(i)}$:

$$
c^{(i)} = \arg\min_{j} \left\| x^{(i)} - \mu_j \right\|
$$

where $c^{(i)}$ is the index of the centroid that is closest to $x^{(i)}$.

#### 3. Update
After the assignment step, the centroids are updated to be the mean of all data points assigned to each centroid. For each centroid $\mu_j$:

$$
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x^{(i)}
$$

where $C_j$ is the set of points assigned to centroid $j$ and $|C_j|$ is the number of points in $C_j$.

#### 4. Convergence
The algorithm repeats the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached. The convergence criterion ensures that the algorithm stops when the clusters are stable.

### Choosing the Number of Clusters

Selecting the appropriate number of clusters (K) is crucial for the success of the K-Means algorithm. Common methods to determine the optimal number of clusters include:

#### 1. Elbow Method
The Elbow Method involves plotting the inertia against the number of clusters. The point where the inertia starts to decrease more slowly (forming an "elbow") indicates the optimal number of clusters.

#### 2. Silhouette Score
The Silhouette Score measures how similar a point is to its own cluster compared to other clusters. It ranges from -1 to 1, with higher values indicating better-defined clusters.

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

where:
- $a(i)$ is the average distance between $i$ and all other points in the same cluster.
- $b(i)$ is the lowest average distance between $i$ and all points in the nearest cluster.

#### 3. Gap Statistic
The Gap Statistic compares the total within-cluster variation for different numbers of clusters with their expected values under null reference distribution of the data.

### Advantages and Disadvantages of K-Means

#### Advantages
- **Simplicity**: K-Means is easy to implement and understand.
- **Scalability**: K-Means can handle large datasets efficiently.
- **Speed**: The algorithm is computationally efficient, making it suitable for real-time applications.

#### Disadvantages
- **Choice of K**: The number of clusters (K) must be specified in advance, which can be challenging.
- **Sensitivity to Initialization**: The final results can be affected by the initial placement of centroids.
- **Sensitivity to Outliers**: K-Means can be sensitive to outliers and noisy data.

### Conclusion

K-Means clustering is a powerful and widely used algorithm for partitioning data into meaningful groups. By understanding the theory and steps involved, you can effectively apply K-Means to a variety of clustering problems. In the next section, we will go through a manual example to solidify our understanding of the workflow.

---
### Example of K-Means Clustering

Let's consider a simple dataset with 4 two-dimensional points. We will cluster these points into 2 clusters (K=2).

### Dataset
- Point A: (1, 1)
- Point B: (1.5, 2)
- Point C: (3, 4)
- Point D: (5, 7)

### Initial Setup
Let's randomly choose the initial centroids. For simplicity, we will use the following points as initial centroids:
- Centroid 1: (1, 1) (same as Point A)
- Centroid 2: (5, 7) (same as Point D)

### Step-by-Step Process

#### Step 1: Assignment Step
Assign each data point to the nearest centroid based on Euclidean distance.

**Calculating Distances:**

1. Distance between Point A (1, 1) and Centroid 1 (1, 1):

$$
\sqrt{(1-1)^2 + (1-1)^2} = 0
$$

2. Distance between Point A (1, 1) and Centroid 2 (5, 7):

$$
\sqrt{(1-5)^2 + (1-7)^2} = \sqrt{16 + 36} = \sqrt{52} \approx 7.21
$$

Point A is assigned to Centroid 1.

3. Distance between Point B (1.5, 2) and Centroid 1 (1, 1):

$$
\sqrt{(1.5-1)^2 + (2-1)^2} = \sqrt{0.25 + 1} = \sqrt{1.25} \approx 1.12
$$

4. Distance between Point B (1.5, 2) and Centroid 2 (5, 7):

$$
\sqrt{(1.5-5)^2 + (2-7)^2} = \sqrt{12.25 + 25} = \sqrt{37.25} \approx 6.10
$$

Point B is assigned to Centroid 1.

5. Distance between Point C (3, 4) and Centroid 1 (1, 1):

$$
\sqrt{(3-1)^2 + (4-1)^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.61
$$

6. Distance between Point C (3, 4) and Centroid 2 (5, 7):

$$
\sqrt{(3-5)^2 + (4-7)^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.61
$$

Point C is equally distant to both centroids. For this iteration, let's assign it to Centroid 1.

7. Distance between Point D (5, 7) and Centroid 1 (1, 1):

$$
\sqrt{(5-1)^2 + (7-1)^2} = \sqrt{16 + 36} = \sqrt{52} \approx 7.21
$$

8. Distance between Point D (5, 7) and Centroid 2 (5, 7):

$$
\sqrt{(5-5)^2 + (7-7)^2} = \sqrt{0 + 0} = 0
$$

Point D is assigned to Centroid 2.

**Cluster Assignments after Step 1:**
- Cluster 1: Points A, B, C
- Cluster 2: Point D

#### Step 2: Update Step
Recalculate the centroids as the mean of the points assigned to each cluster.

**Calculating New Centroids:**

1. New Centroid 1 (mean of Points A, B, C):

$$
\left( \frac{1+1.5+3}{3}, \frac{1+2+4}{3} \right) = (1.83, 2.33)
$$

2. New Centroid 2 (mean of Point D):

$$
(5, 7)
$$

**New Centroids after Step 2:**
- Centroid 1: (1.83, 2.33)
- Centroid 2: (5, 7)

#### Step 3: Repeat Steps 1 and 2 until convergence

**Iteration 2: Assignment Step**

1. Distance between Point A (1, 1) and new Centroid 1 (1.83, 2.33):

$$
\sqrt{(1-1.83)^2 + (1-2.33)^2} = \sqrt{0.6889 + 1.7689} \approx 1.71
$$

2. Distance between Point A (1, 1) and Centroid 2 (5, 7):

$$
\sqrt{(1-5)^2 + (1-7)^2} = \sqrt{16 + 36} = \sqrt{52} \approx 7.21
$$

Point A is assigned to Centroid 1.

3. Distance between Point B (1.5, 2) and new Centroid 1 (1.83, 2.33):

$$
\sqrt{(1.5-1.83)^2 + (2-2.33)^2} = \sqrt{0.1089 + 0.1089} \approx 0.47
$$

4. Distance between Point B (1.5, 2) and Centroid 2 (5, 7):

$$
\sqrt{(1.5-5)^2 + (2-7)^2} = \sqrt{12.25 + 25} = \sqrt{37.25} \approx 6.10
$$

Point B is assigned to Centroid 1.

5. Distance between Point C (3, 4) and new Centroid 1 (1.83, 2.33):

$$
\sqrt{(3-1.83)^2 + (4-2.33)^2} = \sqrt{1.3761 + 2.7889} \approx 2.18
$$

6. Distance between Point C (3, 4) and Centroid 2 (5, 7):

$$
\sqrt{(3-5)^2 + (4-7)^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.61
$$

Point C is assigned to Centroid 1.

7. Distance between Point D (5, 7) and new Centroid 1 (1.83, 2.33):

$$
\sqrt{(5-1.83)^2 + (7-2.33)^2} = \sqrt{10.1089 + 21.8889} \approx 5.64
$$

8. Distance between Point D (5, 7) and Centroid 2 (5, 7):

$$
\sqrt{(5-5)^2 + (7-7)^2} = \sqrt{0 + 0} = 0
$$

Point D is assigned to Centroid 2.

**Cluster Assignments after Iteration 2:**
- Cluster 1: Points A, B, C
- Cluster 2: Point D

**Iteration 2: Update Step**

1. New Centroid 1 (mean of Points A, B, C):

$$
\left( \frac{1+1.5+3}{3}, \frac{1+2+4}{3} \right) = (1.83, 2.33)
$$

2. New Centroid 2 (mean of Point D):

$$
(5, 7)
$$

The centroids remain unchanged. Since the centroids did not change, the algorithm has converged.

### Final Clusters and Centroids

**Cluster Assignments:**
- Cluster 1: Points A (1, 1), B (1.5, 2), C (3, 4)
- Cluster 2: Point D (5, 7)

**Centroids:**
- Centroid 1: (1.83, 2.33)
- Centroid 2: (5, 7)

This manual example illustrates how the K-Means algorithm iteratively assigns points to clusters and updates centroids until convergence. Next, we will implement this process programmatically with visualization to better understand the workflow.

---

### K-Means Algorithm Implementation with Visualizations

We'll use the following steps to implement the K-Means algorithm:
1. Initialize the centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update the centroids to be the mean of the points assigned to them.
4. Repeat the assignment and update steps until convergence.
5. Visualize the steps to understand the algorithm's workflow.

#### Step 1: Import Libraries

First, we need to import the necessary libraries.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)
```

#### Step 2: Generate Sample Data

We'll create a sample dataset with 2D points for visualization purposes.

```python
# Generate sample data
X = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])
```

#### Step 3: Initialize Centroids

We'll initialize the centroids randomly from the dataset.

```python
def initialize_centroids(X, K):
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]

# Number of clusters
K = 2

# Initialize centroids
centroids = initialize_centroids(X, K)
print("Initial centroids:\n", centroids)
```

#### Step 4: Assign Points to the Nearest Centroid

We'll compute the Euclidean distance between each point and each centroid and assign the point to the nearest centroid.

```python
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m, dtype=int)
    
    for i in range(m):
        distances = np.sum((X[i] - centroids)**2, axis=1)
        idx[i] = np.argmin(distances)
    
    return idx

# Assign points to the nearest centroid
idx = find_closest_centroids(X, centroids)
print("Initial assignments:", idx)
```

#### Step 5: Update Centroids

We'll update the centroids to be the mean of the points assigned to them.

```python
def compute_centroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0) if len(points) > 0 else np.zeros(n)
    
    return centroids

# Update centroids
centroids = compute_centroids(X, idx, K)
print("Updated centroids:\n", centroids)
```

#### Step 6: Repeat Until Convergence

We'll repeat the assignment and update steps until the centroids no longer change significantly.

```python
def k_means(X, K, max_iters=10):
    centroids = initialize_centroids(X, K)
    previous_centroids = centroids
    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
        if np.all(centroids == previous_centroids):
            break
        previous_centroids = centroids
    return centroids, idx

# Run K-Means
centroids, idx = k_means(X, K)
print("Final centroids:\n", centroids)
print("Final assignments:", idx)
```

#### Step 7: Visualization

We'll visualize the steps of the K-Means algorithm.

```python
def plot_k_means(X, centroids, idx, K, title):
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', marker='o', edgecolor='k', s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot initial state
plot_k_means(X, centroids, idx, K, "Final Clustering")

![k-means-plot](/img/image.png)

```

### Full Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Generate sample data
X = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])

def initialize_centroids(X, K):
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m, dtype=int)
    
    for i in range(m):
        distances = np.sum((X[i] - centroids)**2, axis=1)
        idx[i] = np.argmin(distances)
    
    return idx

def compute_centroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0) if len(points) > 0 else np.zeros(n)
    
    return centroids

def k_means(X, K, max_iters=10):
    centroids = initialize_centroids(X, K)
    previous_centroids = centroids
    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
        if np.all(centroids == previous_centroids):
            break
        previous_centroids = centroids
    return centroids, idx

def plot_k_means(X, centroids, idx, K, title):
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', marker='o', edgecolor='k', s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Run K-Means
K = 2
centroids, idx = k_means(X, K)
plot_k_means(X, centroids, idx, K, "Final Clustering")
```

### Explanation of Visualization

1. **Initial State**: Shows the initial random centroids and the initial assignment of points.
2. **Iteration Steps**: Visualizes how points are reassigned and centroids are updated in each iteration.
3. **Final State**: Displays the final clusters and centroids after the algorithm converges.

