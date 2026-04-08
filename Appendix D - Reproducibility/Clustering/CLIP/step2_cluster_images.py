
import numpy as np
import os
import shutil
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist

def master_clustering_workflow():
    N_CLUSTERS = 100
    TOP_N_OUTLIERS = 20
    
    input_pkl = 'image_features.pkl' #pickle file from step 1
    source_img_dir = r"/path/directory"
    dest_folder = 'Sorted_Results_clip' #rename as needed, recommend you add the optimal k
    report_file = 'final_report_clip.html' # rename as needed

    print("Starting the sorting process")

    print(f"Loading features from {input_pkl}...")
    if not os.path.exists(input_pkl):
        print(f"{input_pkl} not found.")
        return

    with open(input_pkl, 'rb') as f:
        features_dict = pickle.load(f)

    filenames = list(features_dict.keys())
    data = np.vstack(list(features_dict.values()))
    print(f"   Loaded {len(filenames)} images.")

    print("Compressing data (PCA)") #PCA compression
    pca = PCA(n_components=0.95, random_state=42)
    data_pca = pca.fit_transform(data)

    print(f"Clustering into {N_CLUSTERS} groups...") #clustering into k means
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_pca)
    
    print("Calculating Score") # calculates the silhouette score
    global_score = silhouette_score(data_pca, labels)
    sample_scores = silhouette_samples(data_pca, labels)
    print(f"Global Silhouette Score: {global_score:.4f}")

    print("Identifying outliers") #larger distances to cluster centers
    distances = cdist(data_pca, kmeans.cluster_centers_, 'euclidean')
    dist_to_center = []
    for i, label in enumerate(labels):
        dist_to_center.append(distances[i, label])
    dist_to_center = np.array(dist_to_center)

    print(f"Moving files to '{dest_folder}'") # starts the organisation of the files
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    
    cluster_data = {} 

    for i in range(len(filenames)):
        label = labels[i]
        filename = filenames[i]
        src_path = os.path.join(source_img_dir, filename)
        
        if not os.path.exists(src_path):
            continue

        cluster_dir = os.path.join(dest_folder, f"Cluster_{label:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        dst_path = os.path.join(cluster_dir, filename)
        shutil.copy2(src_path, dst_path) # Exact copy, no modification

        if label not in cluster_data: cluster_data[label] = []
        cluster_data[label].append({
            'path': dst_path,
            'distance': dist_to_center[i],
            'silhouette': sample_scores[i]
        })

    print(f"Generating HTML {report_file}...") # generates the html report that includes the outliers and sil scores
    
    html_content = f"""
    <html>
    <head>
        <title>Clustering Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #eef2f5; padding: 20px; color: #333; }}
            h1 {{ color: #2c3e50; }}
            .cluster-box {{ background: white; padding: 25px; margin-bottom: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            .cluster-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f1f1; padding-bottom: 15px; margin-bottom: 15px; }}
            
            .images {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); 
                gap: 15px; 
            }}
            
            .img-card {{ 
                border: 1px solid #eee; 
                background: #fff; 
                border-radius: 8px;
                overflow: hidden;
                transition: transform 0.2s;
            }}
            
            /* --- FIXED IMAGE STYLING --- */
            .img-container {{
                width: 100%;
                height: 180px; /* Fixed container height */
                background-color: #f8f9fa; /* Light gray background to see transparency */
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}
            
            img {{ 
                max-width: 100%; 
                max-height: 100%; 
                width: auto; 
                height: auto;
                object-fit: contain; /* CRITICAL: Ensures no cropping */
            }}
            
            .meta {{
                padding: 8px;
                font-size: 11px;
                color: #666;
                background: #fff;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <h1>Sort Summary</h1>
        <p>Global Score: <strong>{global_score:.4f}</strong> | Clusters: {N_CLUSTERS}</p>
        <p><i>Images are displayed in their original aspect ratio (not cropped).</i></p>
    """
# customizable html and css if needed.
    sorted_cluster_ids = sorted(cluster_data.keys())
    for cid in sorted_cluster_ids:
        items = cluster_data[cid]
        avg_sil = np.mean([x['silhouette'] for x in items])
        
        items.sort(key=lambda x: x['distance'], reverse=True)# Sort by distance (Outliers first)
        display_items = items[:TOP_N_OUTLIERS]

        html_content += f"""
        <div class="cluster-box">
            <div class="cluster-header">
                <h2>Cluster {cid:02d}</h2>
                <div>Avg Silhouette: <strong>{avg_sil:.4f}</strong> | Total Images: {len(items)}</div>
            </div>
            <div class="images">
        """

        for item in display_items:
            rel_path = os.path.relpath(item['path'], os.path.dirname(os.path.abspath(report_file)))
            
            html_content += f"""
                <div class="img-card">
                    <div class="img-container">
                        <a href="{rel_path}" target="_blank">
                            <img src="{rel_path}" loading="lazy">
                        </a>
                    </div>
                    <div class="meta">
                        Dist: {item['distance']:.2f}<br>
                        Sil: {item['silhouette']:.2f}
                    </div>
                </div>
            """
        html_content += "</div></div>"
    html_content += "</body></html>"
    
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(html_content)

    print(f"Open html '{report_file}' to see raw images.")

if __name__ == '__main__':
    master_clustering_workflow()
