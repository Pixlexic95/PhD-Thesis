
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

input_file = "image_features.pkl" #pickle output file from step 1

def find_optimal_clusters():
    print("Finding the optimal number of clusters...")

    if not os.path.exists(input_file):
        print(f"{input_file} not found.") #Check the pickle file
        return
        
    print("Loading features...") #loading the pickle file
    with open(input_file, 'rb') as f:
        features_dict = pickle.load(f)

    data = np.vstack(list(features_dict.values())) #converts the numpy features
    print(f"Loaded {len(data)} images.")

    print("Compressing data (PCA)...") # PCA: Reduces the dimensionality and speeds up calculation.
    pca = PCA(n_components=0.95, random_state=42)
    data_pca = pca.fit_transform(data)
    print(f"Dimensions reduced from {data.shape[1]} to {data_pca.shape[1]}")
    print(f"Variance Retained: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print("\nTesting different cluster counts...") #keeps communication line open while it runs
    print(f"{'Clusters (K)':<15} | {'Score':<10} | {'Status'}")

    results = []
    k_values = range(10, 101, 10) # tests from 10 to 100 in steps of 10 - this can be adjusted

    best_score = -1
    best_k = 0

    for k in k_values: # running k-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)# n_init=3 makes it faster for testing
        labels = kmeans.fit_predict(data_pca)
        
        score = silhouette_score(data_pca, labels) #calculate silhouette score. -1 is bad and + 1 is good.
        results.append(score)
        
        marker = ""
        if score > best_score:
            best_score = score
            best_k = k
            marker = "New Best" #just adds a marker to the best score
            
        print(f"{k:<15} | {score:.4f}     | {marker}")

    print(f"\n Set num_clusters = {best_k}")
    print(f"(Highest separation score of {best_score:.4f})")

if __name__ == '__main__':
    find_optimal_clusters()
