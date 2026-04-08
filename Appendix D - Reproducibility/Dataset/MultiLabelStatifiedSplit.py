
import os
import glob
import shutil
import random
import argparse
import csv
from collections import defaultdict, Counter

def get_class_counts_from_label(label_path):
    #Returns class_id: count 
    counts = Counter()
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            class_id = int(line.strip().split()[0])
            counts[class_id] += 1
    return counts

def find_image_for_base(base, img_dir):
    #Finds matching image 
    candidates = [base, f"image_{base}", f"frame_{base}", f"img_{base}"]
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for name in candidates:
        for ext in exts:
            cand = os.path.join(img_dir, name + ext)
            if os.path.isfile(cand):
                return cand
    return None

def stratified_split(img_dir, label_dir, out_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42): # can be adapted, this is currently at 70:20:10
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Splits must sum to 1"
    random.seed(seed)

    print("Step 1: Scanning files and pairing images with labels...")
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    
    images_with_labels = {}
    global_class_counts = Counter()
    
    for lbl_path in label_files:
        base = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = find_image_for_base(base, img_dir)
        
        if img_path:
            class_counts = get_class_counts_from_label(lbl_path)
            if sum(class_counts.values()) > 0: # Only include images that actually have objects
                images_with_labels[base] = {
                    'img_path': img_path,
                    'lbl_path': lbl_path,
                    'classes': class_counts
                }
                for c, count in class_counts.items():
                    global_class_counts[c] += count

    print(f"Valid image-label pairs found: {len(images_with_labels)}")
    print(f"Total unique classes found: {len(global_class_counts)}")

    #  Calculate target instances for each split
    targets = {
        'train': {c: count * train_ratio for c, count in global_class_counts.items()},
        'val': {c: count * val_ratio for c, count in global_class_counts.items()},
        'test': {c: count * test_ratio for c, count in global_class_counts.items()}
    }
    
    current_counts = {
        'train': Counter(),
        'val': Counter(),
        'test': Counter()
    }

    # Sort images by rarity (process images with rare classes first). The score of an image is the sum of (1 / total_class_count) for its classes
    def get_image_rarity_score(base):
        classes = images_with_labels[base]['classes']
        return sum(count * (1.0 / global_class_counts[c]) for c, count in classes.items())

    sorted_bases = sorted(list(images_with_labels.keys()), key=get_image_rarity_score, reverse=True)

    # Multi-label Stratified Assignment
    print("Step 2: Assigning splits based on multi-label deficit scoring...")
    split_assignments = {}
    
    for base in sorted_bases:
        img_classes = images_with_labels[base]['classes']
        
        best_split = None
        best_score = -float('inf')
        
        # Calculate which split needs the image most
        for split in ['train', 'val', 'test']:
            score = 0
            for c, count in img_classes.items():
                target = targets[split][c]
                current = current_counts[split][c]
                # prevents dividing by zero
                if target > 0:
                    deficit = (target - current) / target
                    score += deficit * count
            
            if score > best_score:
                best_score = score
                best_split = split
                
        # Assign to the best split
        split_assignments[base] = best_split
        for c, count in img_classes.items():
            current_counts[best_split][c] += count

    #  Copy files to output dir
    print("Step 3: Copying files to respective directories...")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)

    for base, data in images_with_labels.items():
        split = split_assignments[base]
        shutil.copy(data['img_path'], os.path.join(out_dir, split, "images", os.path.basename(data['img_path'])))
        shutil.copy(data['lbl_path'], os.path.join(out_dir, split, "labels", os.path.basename(data['lbl_path'])))

    #  Generate CSV Report
    report_path = os.path.join(out_dir, "split_distribution_report.csv")
    with open(report_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Class_ID", "Total", "Train", "Val", "Test", "Train_%", "Val_%", "Test_%"])
        
        for c in sorted(global_class_counts.keys()):
            total = global_class_counts[c]
            tr = current_counts['train'][c]
            va = current_counts['val'][c]
            te = current_counts['test'][c]
            writer.writerow([
                c, total, tr, va, te,
                f"{(tr/total)*100:.1f}%", f"{(va/total)*100:.1f}%", f"{(te/total)*100:.1f}%"
            ])
            
    print(f"\nDataset balanced")
    print(f"Class distribution report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", required=True)
    parser.add_argument("-l", "--labels", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()
    
    stratified_split(args.images, args.labels, args.output, args.train, args.val, args.test)

if __name__ == "__main__":
    main()
