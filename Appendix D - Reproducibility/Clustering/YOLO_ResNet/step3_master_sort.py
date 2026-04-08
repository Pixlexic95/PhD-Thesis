#Used for both YOLO and ResNet
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist

def master_clustering_workflow():
    N_CLUSTERS = 140 # optimum number from step 2
    TOP_N_OUTLIERS = 50
    
    features_file = 'features_mix.npy'
    filenames_file = 'filenames_mix.npy'  # use outputs
    dest_folder = 'Sorted_Results_mix2' # renamed as appropriate
    report_file = 'final_report_mix2.html' # same as above

    print(" Sorting starting")

    print(f"Loading features and filenames")
    if not os.path.exists(features_file) or not os.path.exists(filenames_file):
        print(f"Missing .npy files")
        return

    data = np.load(features_file)
    filenames = np.load(filenames_file)
    print(f"Loaded {len(filenames)} images")

    print("Compressing data (PCA)") #pca to 95% variance retention
    pca = PCA(n_components=0.95, random_state=42)
    data_pca = pca.fit_transform(data)

    print(f"Clustering into {N_CLUSTERS} groups")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_pca)
    
    print("Calculating Score...")
    global_score = silhouette_score(data_pca, labels)
    sample_scores = silhouette_samples(data_pca, labels)
    print(f"Global Silhouette Score: {global_score:.4f}")

    print("Identifying outliers...")
    distances = cdist(data_pca, kmeans.cluster_centers_, 'euclidean')
    dist_to_center = np.min(distances, axis=1)

    print(f"6. Moving files to '{dest_folder}'...")
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    
    cluster_data = {} 

    for i in range(len(filenames)):
        label = labels[i]
        src = filenames[i]
        
        if not os.path.exists(src):
            continue

        cluster_dir = os.path.join(dest_folder, f"Cluster_{label:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        dst = os.path.join(cluster_dir, os.path.basename(src))
        shutil.copy2(src, dst)

        if label not in cluster_data: cluster_data[label] = []
        cluster_data[label].append({
            'path': dst,
            'distance': dist_to_center[i],
            'silhouette': sample_scores[i]
        })

    print(f"7. Generating {report_file}...")
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; background: #f0f0f0; padding: 20px; }}
            .cluster-box {{ background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .cluster-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 10px; }}
            .score-bad {{ color: #d9534f; font-weight: bold; }}
            .score-good {{ color: #5cb85c; font-weight: bold; }}
            .images {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .img-card {{ width: 150px; border: 1px solid #ddd; padding: 5px; text-align: center; font-size: 10px; background: #fff; overflow: hidden; }}
            img {{ width: 100%; height: 100px; object-fit: cover; }}
        </style>
    </head>
    <body>
        <h1>Run Summary</h1>
        <p>Global Consistency Score: <strong>{global_score:.4f}</strong></p>
        <p><i>Showing top {TOP_N_OUTLIERS} most confusing images per cluster.</i></p>
    """

    sorted_cluster_ids = sorted(cluster_data.keys())
    for cid in sorted_cluster_ids:
        items = cluster_data[cid]
        avg_sil = np.mean([x['silhouette'] for x in items])
        score_class = "score-good" if avg_sil > 0.1 else "score-bad"
        
        items.sort(key=lambda x: x['distance'], reverse=True) # sort by distance to center (most outlying first)
        top_outliers = items[:TOP_N_OUTLIERS]

        html_content += f"""
        <div class="cluster-box">
            <div class="cluster-header">
                <h2>Cluster {cid:02d}</h2>
                <div>Score: <span class="{score_class}">{avg_sil:.4f}</span> | Count: {len(items)}</div>
            </div>
            <div class="images">
        """

        for item in top_outliers:
            rel_path = os.path.relpath(item['path'], os.path.dirname(os.path.abspath(report_file)))
            html_content += f"""
                <div class="img-card">
                    <img src="{rel_path}" loading="lazy">
                    <br>Dist: {item['distance']:.2f}
                </div>
            """
        html_content += "</div></div>"
    html_content += "</body></html>"
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(html_content)

    print(f"All done. HTML: '{report_file}'")

if __name__ == '__main__':
    master_clustering_workflow()