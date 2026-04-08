
import os
import cv2 # OpenCV for opening and processing images
import matplotlib.pyplot as plt # visualizing the image

V1_CLASSES = {
    0: "Fish",
    1: "Crustacean",
    2: "Jellyfish",
    3: "Coral",
    4: "Wood",
    5: "Seaweed",
    6: "Shell",
    7: "Plastic Wrapper",
    8: "Plastic Container",
    9: "Metal Container",
    10: "Net",
    11: "Plastic Object",
    12: "Metal Object",
    13: "Cloth",
    14: "Glass",
    15: "Rubber"
}

V2_MENU = {
    "Clay": ["Porcelain_Object", "Concrete"],
    "Metal": ["Metal_Bullets", "Metal_Large_Container", "Metal_Object", "Metal_Pole", 
              "Metal_Propellor", "Metal_Rusted", "Metal_Wheel", "Metal_Wire", "Cans", "Large_Box"],
    "Plastic": ["Plastic_Bag", "Plastic_Bottle", "Plastic_Object", "Plastic_Straw", 
                "Plastic_Toy", "Plastic_Tube", "Food_Wrapper"],
    "Rubber": ["Rubber_Glove", "Rubber_Tire", "Elastic_Band"],
    "Fishing Gear": ["Tangled_Rope", "Rope", "Net", "Fishing_Hook", "Fishing_Line"],
    "Cloth": ["Fabric_Bag", "Fabric_Mesh"],
    "Glass": ["Glass_Bottle", "Glass_Container"],
    "Chondrichthyes": ["Holocephalli", "Batamorphii", "Selachii"],
    "Osteichthyes": ["Chondostrei", "Coelanths"],
    "Teleostei": ["Eels", "Large_Fish", "Beloniforms"],
    "Actinopterygii": ["Tetraodontiformes", "Syngnathiformes", "Pleuronectiformes", 
                       "Zoarcoidei", "Scorpaeniformes", "Blennies", "Reef_Fish_Oval", 
                       "Reef_Fish_Triangle", "Parrotfish"],
    "Echinodermata": ["Echinoidea", "Ophiurodea", "Asteroidea", "Holothuroidea", "Crinoidea"],
    "Mollusca": ["Bivalvia", "Nudibranchia", "Cephalopoda", "Gastropoda"],
    "Arthropoda": ["Brachyura", "Homarus", "Majoidea", "Pleocyemata"],
    "Cnidaria": ["Ctenophora", "Actiniaria", "Scyphozoa"],
    "Anthozoa": ["Scleractinia", "Alcyonacea", "Antipatharia"],
    "Porifera": ["Porifera"],
    "Annelida": ["Annelida"],
    "Plants": ["Macro_Algae", "Sea_Moss", "Seagrass"],
    "Reptile": ["Turtle"],
    "Mammal": ["Dolphin", "Seal"],
    "Misc": ["Shell", "Rock", "Wood", "Unidentified"]
}

# Here we "flatten" the hierarchical dictionary into a numbered list, so that YOLO can understand (0, 1, 2, 3...)
FLAT_MENU = {}
current_id = 0
for parent, subclasses in V2_MENU.items():
    for subclass in subclasses:
        FLAT_MENU[current_id] = {"parent": parent, "subclass": subclass}
        current_id += 1

IMAGES_DIR = r"path/to/images/" #r is for backslashes
OLD_LABELS_DIR = r"/path/to/labels/"
NEW_LABELS_DIR = r"/new/labels/path/"

def yolo_to_pixels(yolo_bbox, img_width, img_height):
    x_center, y_center, w, h = map(float, yolo_bbox)
    box_width = w * img_width
    box_height = h * img_height
    x_min = int((x_center * img_width) - (box_width / 2))
    y_min = int((y_center * img_height) - (box_height / 2))
    x_max = int((x_center * img_width) + (box_width / 2))
    y_max = int((y_center * img_height) + (box_height / 2))
    return x_min, y_min, x_max, y_max

def main():
    print("Starting Interactive Relabeller")
    os.makedirs(NEW_LABELS_DIR, exist_ok=True)
    
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_filename in image_files:
        base_name = os.path.splitext(img_filename)[0]

        old_label_path = os.path.join(OLD_LABELS_DIR, base_name + ".txt") # keeps the original name
        new_label_path = os.path.join(NEW_LABELS_DIR, base_name + ".txt")
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        if os.path.exists(new_label_path):
            continue 

        if not os.path.exists(old_label_path): #remove this for labelling images without old paths
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Problem reading image {img_filename}. Skipping.")
            continue
            
        img_height, img_width, _ = img.shape
        
        with open(old_label_path, 'r') as file:
            lines = file.readlines()
            
        new_lines = []
        
        for line in lines:
            parts = line.strip().split()
            old_class_id = int(parts[0])
            yolo_coords = parts[1:]
            
            display_img = img.copy()
            x1, y1, x2, y2 = yolo_to_pixels(yolo_coords, img_width, img_height)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(display_img_rgb)
            plt.title(f"Old Label: {V1_CLASSES.get(old_class_id, 'Unknown')}")
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            
            print(f"\nImage: {img_filename} | Old Class: {V1_CLASSES.get(old_class_id, 'Unknown')}")
            print("-" * 30)
            
            current_parent = "" #print the menu each time
            for new_id, data in FLAT_MENU.items():
                if data['parent'] != current_parent:
                    print(f"\n--- {data['parent']} ---")
                    current_parent = data['parent']
                print(f"[{new_id}] {data['subclass']}")
                
            print("\n[s] Skip this box (delete it)")
            print("[q] Quit program")
            
            valid_input = False
            while not valid_input:
                choice = input("\nEnter the new class ID: ").strip().lower()
                
                if choice == 'q':
                    plt.close('all')
                    print("Progress saved. Quitting.")
                    return 
                elif choice == 's':
                    valid_input = True
                    print("Skipping box.")
                elif choice.isdigit() and int(choice) in FLAT_MENU:
                    valid_input = True
                    new_class_id = int(choice)
                    selected_data = FLAT_MENU[new_class_id]
                    print(f"Assigned to: {selected_data['parent']} -> {selected_data['subclass']}")
                    
                    new_line = f"{new_class_id} " + " ".join(yolo_coords) + "\n"
                    new_lines.append(new_line)
                else:
                    print("Invalid choice.")
            
            plt.close('all')
            
        with open(new_label_path, 'w') as file:
            file.writelines(new_lines)

    print("\nAll images have been relabelled")

    # generates the new classes.txt for YOLO - can comment out if not necessary
    with open(os.path.join(NEW_LABELS_DIR, "classes.txt"), 'w') as file:
        for new_id, data in FLAT_MENU.items():
            file.write(f"{data['subclass']}\n")

if __name__ == "__main__":
    main()
