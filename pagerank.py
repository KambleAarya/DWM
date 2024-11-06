import numpy as np

def page_rank(M, num_iterations=100, damping_factor=0.85):
    # Number of nodes (pages)
    N = M.shape[1]

    # Initialize the PageRank vector with equal values
    rank = np.ones(N) / N

    # PageRank calculation
    for _ in range(num_iterations):
        rank = (1 - damping_factor) / N + damping_factor * M @ rank

    return rank

# Example usage:
# Graph represented by an adjacency matrix where each column represents outgoing links
# Each column should sum to 1 for stochasticity, ensuring all probabilities add up to 1
M = np.array([
    [0, 0, 1, 0],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0, 1, 0, 0]
])

# Run PageRank
ranks = page_rank(M)
print("PageRank values:", ranks)