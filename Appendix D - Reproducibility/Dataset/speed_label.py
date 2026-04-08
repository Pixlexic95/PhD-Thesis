#adapted final script for amphv2 quality assurance
import cv2
import os
import csv
import numpy as np

image_dir = r"C:\path\"
old_labels_dir = r"C:\path\old_labels"
output_dir = r"C:\path\new_labels"

labels_dir = os.path.join(output_dir, "labels")
visuals_dir = os.path.join(output_dir, "visuals")

CLASSES = ['Porcelain_Object',
        'Concrete',
       'Metal_Bullets',
       'Metal_Large_Container',
       'Metal_Object',
       'Metal_Pole',
       'Metal_Propellor',
       'Metal_Rusted',
       'Metal_Wheel',
       'Metal_Wire',
       'Cans',
       'Large_Box',
       'Plastic_Bag',
       'Plastic_Bottle',
       'Plastic_Object',
       'Plastic_Straw',
       'Plastic_Toy',
       'Plastic_Tube',
       'Food_Wrapper',
       'Rubber_Glove',
       'Rubber_Tire',
       'Elastic_Band',
       'Tangled_Rope',
       'Rope',
       'Net',
       'Fishing_Hook',
       'Fishing_Line',
       'Fabric_Bag',
       'Fabric_Mesh',
       'Glass_Bottle',
       'Glass_Container',
       'Holocephalli',
       'Batamorphii',
       'Selachii',
       'Chondostrei',
       'Coelanths',
       'Eels',
       'Large_Fish',
       'Beloniforms',
       'Tetraodontiformes',
       'Syngnathiformes',
       'Pleuronectiformes',
       'Zoarcoidei',
       'Scorpaeniformes', 
       'Blennies', 
       'Reef_Fish_Oval', 
       'Reef_Fish_Triangle', 
       'Parrotfish', 
       'Echinoidea', 
       'Ophiurodea', 
       'Asteroidea', 
       'Holothuroidea', 
       'Crinoidea',
       'Bivalvia',
       'Nudibranchia',
       'Cephalopoda',
       'Gastropoda',
       'Brachyura',
       'Homarus',
       'Majoidea',
       'Pleocyemata',
       'Ctenophora',
       'Actiniaria',
       'Scyphozoa',
       'Scleractinia',
       'Alcyonacea',
       'Antipatharia',
       'Porifera',
       'Annelida',
       'Macro_Algae',
       'Sea_Moss',
       'Seagrass',
       'Turtle',
       'Dolphin',
       'Seal',
       'Shell',
       'Rock',
       'Wood',
       'Unidentified'
]
# any class names can be changed in the yaml as long as they have the same number
class Annotator:
    def __init__(self, window_name):
        self.window_name = window_name
        self.boxes = []      # List of [x1, y1, x2, y2]
        self.labels = []     # List of strings
        self.drawing = False
        self.drag_box_idx = -1
        self.drag_corner_idx = -1 # 0: TL, 1: TR, 2: BR, 3: BL
        self.img = None
        self.display_img = None
        self.handle_size = 8
        self.last_label = CLASSES[0]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(self.boxes):
                corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
                for j, (cx, cy) in enumerate(corners):
                    if abs(x - cx) < self.handle_size and abs(y - cy) < self.handle_size:
                        self.drag_box_idx, self.drag_corner_idx = i, j
                        return
            self.drawing = True
            self.boxes.append([x, y, x, y])
            self.labels.append("Pending...")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.boxes[-1][2], self.boxes[-1][3] = x, y
            elif self.drag_box_idx != -1:
                idx = self.drag_box_idx
                if self.drag_corner_idx == 0: self.boxes[idx][0], self.boxes[idx][1] = x, y
                elif self.drag_corner_idx == 1: self.boxes[idx][2], self.boxes[idx][1] = x, y
                elif self.drag_corner_idx == 2: self.boxes[idx][2], self.boxes[idx][3] = x, y
                elif self.drag_corner_idx == 3: self.boxes[idx][0], self.boxes[idx][3] = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                b = self.boxes[-1]
                self.boxes[-1] = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
                self.request_label(len(self.boxes)-1)
            self.drag_box_idx = -1
            self.drag_corner_idx = -1

    def request_label(self, idx):
        print("\n" + "="*20)
        for i, name in enumerate(CLASSES): print(f"[{i}] {name}")
        choice = input(f"Label for Box #{idx+1} (Default '{self.last_label}'): ").strip().lower()
        if choice.isdigit() and 0 <= int(choice) < len(CLASSES):
            label = CLASSES[int(choice)]
        elif choice == "" or choice == " ":
            label = self.last_label
        else:
            label = choice
        self.labels[idx] = label
        self.last_label = label

    def render(self):
        self.display_img = self.img.copy()
        for i, (box, label) in enumerate(zip(self.boxes, self.labels)):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(self.display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.display_img, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                cv2.rectangle(self.display_img, (cx-4, cy-4), (cx+4, cy+4), (0, 255, 255), -1)
        cv2.imshow(self.window_name, self.display_img)

def save_yolo_format(base_name, boxes, labels, img_shape):
    h, w = img_shape[:2]
    txt_name = os.path.splitext(base_name)[0] + ".txt"
    txt_path = os.path.join(labels_dir, txt_name)
    
    with open(txt_path, 'w') as f:
        for box, label in zip(boxes, labels):
            # Get class ID
            class_id = CLASSES.index(label) if label in CLASSES else 0
            # YOLO format: x_center, y_center, width, height (normalized 0-1)
            x_center = ((box[0] + box[2]) / 2) / w
            y_center = ((box[1] + box[3]) / 2) / h
            width = abs(box[2] - box[0]) / w
            height = abs(box[3] - box[1]) / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    # Setup directories
    for d in [labels_dir, visuals_dir]:
        if not os.path.exists(d): os.makedirs(d)
    
    # Save classes.txt for reference
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for c in CLASSES: f.write(c + "\n")

    exts = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith(exts)]
    
    annotator = Annotator("Annotator")
    cv2.namedWindow(annotator.window_name)
    cv2.setMouseCallback(annotator.window_name, annotator.mouse_callback)

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        annotator.img = cv2.imread(img_path)
        if annotator.img is None: continue
        annotator.boxes, annotator.labels = [], [] 

        while True:
            annotator.render()
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n'): # NEXT
                if annotator.boxes:
                    # Save txt
                    save_yolo_format(base_name, annotator.boxes, annotator.labels, annotator.img.shape)
                    # Save ground truth
                    vis_path = os.path.join(visuals_dir, "gt_" + base_name)
                    cv2.imwrite(vis_path, annotator.display_img)
                    print(f"Saved YOLO labels and Ground Truth image for: {base_name}")
                break
            elif key == ord('d'): # Ddelete key
                if annotator.boxes: annotator.boxes.pop(); annotator.labels.pop()
            elif key == ord('q'): # Quit
                cv2.destroyAllWindows(); return

    cv2.destroyAllWindows()
    print(f"Files saved in {output_dir}")

if __name__ == "__main__":
    main()
