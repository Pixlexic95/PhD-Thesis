import os
import glob
import argparse
from xml.dom.minidom import Document
from PIL import Image

def convert_to_abs(size, yolo_box):
    w_img, h_img = size
    x_c, y_c, w, h = yolo_box
    x_c *= w_img
    y_c *= h_img
    w *= w_img
    h *= h_img

    xmin = int(round(x_c - w / 2))
    ymin = int(round(y_c - h / 2))
    xmax = int(round(x_c + w / 2))
    ymax = int(round(y_c + h / 2))
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w_img - 1, xmax)
    ymax = min(h_img - 1, ymax)
    return xmin, ymin, xmax, ymax

def txt_to_xml(lut, input_dir, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for txt in glob.glob(os.path.join(input_dir, "*.txt")):
        filename = os.path.splitext(os.path.basename(txt))[0]
        # Find corresponding image file (try common extensions)
        img_path = None
        for ext in [".jpg", ".jpeg", ".JPG"]:
            candidate = os.path.join(img_dir, filename + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"WARNING: No image found for {filename}, skipping.")
            continue

        # Get image size
        with Image.open(img_path) as img:
            width, height = img.size

        xml_path = os.path.join(output_dir, filename + '.xml')
        doc = Document()

        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)

        # filename
        fname_node = doc.createElement('filename')
        fname_node.appendChild(doc.createTextNode(os.path.basename(img_path)))
        annotation.appendChild(fname_node)

        # size
        size_node = doc.createElement('size')
        width_node = doc.createElement('width')
        width_node.appendChild(doc.createTextNode(str(width)))
        size_node.appendChild(width_node)
        height_node = doc.createElement('height')
        height_node.appendChild(doc.createTextNode(str(height)))
        size_node.appendChild(height_node)
        depth_node = doc.createElement('depth')
        depth_node.appendChild(doc.createTextNode("3"))
        size_node.appendChild(depth_node)
        annotation.appendChild(size_node)

        # Parse objects
        with open(txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                classid = int(parts[0])
                coords = list(map(float, parts[1:5]))
                xmin, ymin, xmax, ymax = convert_to_abs((width, height), coords)
                name = lut.get(classid, "Unknown")

                obj = doc.createElement('object')
                name_node = doc.createElement('name')
                name_node.appendChild(doc.createTextNode(name))
                obj.appendChild(name_node)

                bndbox = doc.createElement('bndbox')
                for tag, val in zip(
                    ['xmin', 'ymin', 'xmax', 'ymax'],
                    [str(xmin), str(ymin), str(xmax), str(ymax)]
                ):
                    n = doc.createElement(tag)
                    n.appendChild(doc.createTextNode(val))
                    bndbox.appendChild(n)
                obj.appendChild(bndbox)
                annotation.appendChild(obj)

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(doc.toprettyxml(indent='  '))
        print(f"Wrote {xml_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True) # follow instructions on xml to txt, if needed
    ap.add_argument("-d", "--images", required=True)
    ap.add_argument("-o", "--output", required=True)")
    args = ap.parse_args()
    lut = {
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
        15: "Rubber",
    }
    txt_to_xml(lut, args.input, args.images, args.output)