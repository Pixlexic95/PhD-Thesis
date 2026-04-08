import os  

def count_instances(label_dir):  
    total_instances = 0  

    # Loop through all .txt files in the labels directory  
    for label_file in os.listdir(label_dir):  
        if label_file.endswith('.txt'):  
            with open(os.path.join(label_dir, label_file), 'r') as f:  
                # Count lines in each file (1 line = 1 instance)  
                total_instances += len(f.readlines())  

    return total_instances  

# Path to train or val labels folder  
label_dir = 'C:/'  # or '/path/to/val'  
total_instances = count_instances(label_dir)  
print(f"Total instances in the dataset: {total_instances}")  