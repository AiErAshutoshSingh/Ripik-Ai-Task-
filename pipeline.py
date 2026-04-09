import os
import json
import cv2
import argparse
from ultralytics import YOLO

def run_inference(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO("runs/detect/train/weights/best.pt")

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        results = model(img_path)[0]
        image = cv2.imread(img_path)

        detections = {}

        if results.boxes is not None:
            for i, box in enumerate(results.boxes.xyxy):
                x_min, y_min, x_max, y_max = map(int, box.tolist())

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                detections[str(i+1)] = {
                    "bbox_coordinates": [x_min, y_min, x_max, y_max]
                }

        cv2.imwrite(os.path.join(output_dir, img_name), image)

        json_path = os.path.join(output_dir, img_name.rsplit(".",1)[0] + ".json")
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_inference(args.image_dir, args.output_dir)