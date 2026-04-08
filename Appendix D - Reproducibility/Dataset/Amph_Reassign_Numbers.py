import os

labels_dir = r"/path/dir/" # update this to labels

OLD_CLASSES = [ #original list
    'Porcelain_Object', 'Concrete', 'Metal_Bullets', 'Metal_Large_Container', 'Metal_Object',
    'Metal_Pole', 'Metal_Propellor', 'Metal_Rusted', 'Metal_Wheel', 'Metal_Wire', 'Cans',
    'Large_Box', 'Plastic_Bag', 'Plastic_Bottle', 'Plastic_Object', 'Plastic_Straw',
    'Plastic_Toy', 'Plastic_Tube', 'Food_Wrapper', 'Rubber_Glove', 'Rubber_Tire',
    'Elastic_Band', 'Tangled_Rope', 'Rope', 'Net', 'Fishing_Hook', 'Fishing_Line',
    'Fabric_Bag', 'Fabric_Mesh', 'Glass_Bottle', 'Glass_Container', 'Holocephalli',
    'Batamorphii', 'Selachii', 'Chondostrei', 'Coelanths', 'Eels', 'Large_Fish',
    'Beloniforms', 'Tetraodontiformes', 'Syngnathiformes', 'Pleuronectiformes',
    'Zoarcoidei', 'Scorpaeniformes', 'Blennies', 'Reef_Fish_Oval', 'Reef_Fish_Triangle',
    'Parrotfish', 'Echinoidea', 'Ophiurodea', 'Asteroidea', 'Holothuroidea', 'Crinoidea', 
    'Bivalvia', 'Nudibranchia', 'Cephalopoda', 'Gastropoda', 'Brachyura', 'Homarus', 
    'Majoidea', 'Pleocyemata', 'Ctenophora', 'Actiniaria', 'Scyphozoa', 'Scleractinia', 
    'Alcyonacea', 'Antipatharia', 'Porifera', 'Annelida', 'Macro_Algae', 'Sea_Moss', 
    'Seagrass', 'Turtle', 'Dolphin', 'Seal', 'Shell', 'Rock', 'Wood', 'Unidentified'
]

NEW_CLASSES = [ # new list
    'Porcelain_Object', 'Concrete', 'Metal_Bullets', 'Metal_Large_Container', 'Metal_Object',
    'Metal_Pole', 'Metal_Propellor', 'Metal_Rusted', 'Metal_Wheel', 'Metal_Wire', 'Cans',
    'Large_Box', 'Plastic_Bag', 'Plastic_Bottle', 'Plastic_Object', 'Plastic_Straw',
    'Plastic_Toy', 'Plastic_Tube', 'Food_Wrapper', 'Rubber_Glove', 'Rubber_Tire',
    'Elastic_Band', 'Tangled_Rope', 'Rope', 'Net', 'Fishing_Hook', 'Fishing_Line',
    'Fabric_Bag', 'Fabric_Mesh', 'Glass_Bottle', 'Glass_Container', 'Holocephalli',
    'Batamorphii', 'Selachii', 'Chondostrei', 'Coelanths', 'Eels', 'Large_Fish',
    'Beloniforms', 'Tetraodontiformes', 'Syngnathiformes', 'Pleuronectiformes',
    'Zoarcoidei', 'Scorpaeniformes', 'Blennies', 'Reef_Fish_Oval', 'Reef_Fish_Triangle',
    'Reef_Fish_Streamline', # <-- this was the new class added that changed the order
    'Parrotfish', 'Echinoidea', 'Ophiurodea', 'Asteroidea', 'Holothuroidea', 'Crinoidea', 
    'Bivalvia', 'Nudibranchia', 'Cephalopoda', 'Gastropoda', 'Brachyura', 'Homarus', 
    'Majoidea', 'Pleocyemata', 'Ctenophora', 'Actiniaria', 'Scyphozoa', 'Scleractinia', 
    'Alcyonacea', 'Antipatharia', 'Porifera', 'Annelida', 'Macro_Algae', 'Sea_Moss', 
    'Seagrass', 'Turtle', 'Dolphin', 'Seal', 'Shell', 'Rock', 'Wood', 'Unidentified'
]

def main():
    mapping = {}
    for old_idx, class_name in enumerate(OLD_CLASSES):
        if class_name in NEW_CLASSES:
            new_idx = NEW_CLASSES.index(class_name)
            mapping[old_idx] = new_idx
        else:
            print(f"{class_name} not found in new list")

    for old_id, new_id in mapping.items():
        if old_id != new_id:
            print(f"{OLD_CLASSES[old_id]} : Shifted {old_id} -> {new_id}")

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"\nFound {len(txt_files)} label files to process")
    
    confirm = input("Type 'yes' to proceed and overwrite these files: ") # checks that youve saved a backup
    if confirm.lower() != 'yes':
        print("Cancelled")
        return

    files_changed = 0
    for txt_file in txt_files:
        file_path = os.path.join(labels_dir, txt_file)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        needs_rewrite = False
        
        for line in lines:
            parts = line.strip().split()
            
            if len(parts) < 5: # Skip empty lines or malformed lines (YOLO should have at least 5 parts and files are based on YOLO format)
                continue
                
            if not parts[0].isdigit(): # make sure the first part is a number
                continue
                
            old_class_id = int(parts[0])
            
            if old_class_id in mapping:
                new_class_id = mapping[old_class_id]
                if old_class_id != new_class_id:
                    needs_rewrite = True
                parts[0] = str(new_class_id)
            
            new_lines.append(" ".join(parts) + "\n")
            
        if needs_rewrite:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            files_changed += 1

    print(f"Updated classes in {files_changed} files")

if __name__ == "__main__":
    main()