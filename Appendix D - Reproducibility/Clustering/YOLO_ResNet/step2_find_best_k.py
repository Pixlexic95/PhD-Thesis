# This script can be used for YOLO and ResNet

import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def find_optimal_clusters():
    print("Finding optimal cluster count")

    input_file = 'features_test.npy' 
    
    if not os.path.exists(input_file):
        print(f" {input_file} not found")
        return
        
    data = np.load(input_file)
    print(f"Loaded {len(data)} images from {input_file}")

    print("Compressing data for speed...") # PCA for speed
    pca = PCA(n_components=0.95, random_state=42)
    data_pca = pca.fit_transform(data)
    print(f"PCA Compression: Reduced from {data.shape[1]} to {data_pca.shape[1]} dimensions.")
    print(f"Variance Retained: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print("\nTesting different cluster counts...")
    print(f"{'Clusters (K)':<15} | {'Score':<10} | {'Status'}")

    results = []
    k_values = range(10, 351, 10) #modifiable 

    best_score = -1
    best_k = 0

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3) # Low n_init for speed
        labels = kmeans.fit_predict(data_pca)
        
        score = silhouette_score(data_pca, labels)
        results.append(score)
        
        marker = ""
        if score > best_score:
            best_score = score
            best_k = k
            marker = "New Best:" # this keeps printing so look for the newest
            
        print(f"{k:<15} | {score:.4f}     | {marker}")

    print(f"Use N_CLUSTERS = {best_k}")
    print(f"(It achieved the highest separation score of {best_score:.4f})")

if __name__ == '__main__':
    find_optimal_clusters()
