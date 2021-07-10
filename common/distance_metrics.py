import numpy as np

from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
    
def compare_distributions(A, B, support, bootstrap=None):
    
    S,W,J= find_distance(A, B, support)
    
    if bootstrap is not None:
        J_bootA = []
        J_bootB = []
        
        N = np.min([len(A), len(B), 1000])
        
        for i in range(1000):
            selectionA = np.random.permutation(N)
            selectionB = np.random.permutation(N)
            
            S_A,W_A,J_A = find_distance(A[selectionA[:N//2]],
                                          A[selectionA[N//2:]],
                                          support)
            
            S_B,W_B,J_B = find_distance(B[selectionB[:N//2]],
                                          B[selectionB[N//2:]],
                                          support)
            
            J_bootA.append(J_A)
            J_bootB.append(J_B)
            
        return S, W, J, J_bootA, J_bootB
    else:
        
        return S, W, J
    
def find_distance(A, B, support):
    h1, b = np.histogram(A, bins=support, density=True)
    h2, b = np.histogram(B, bins=support, density=True)
    
    if np.min(h1) == 0 or np.min(h2) == 0:
        offset = np.max(np.concatenate((h1,h2))) * 0.001
        h1 += offset
        h2 += offset
    S = entropy(h1, h2)
    W = wasserstein_distance(A, B)
    J = jensenshannon(h1, h2)
    
    return S, W, J
