from ultralytics import YOLO
import cv2
import os

model = YOLO("best.pt")
input_folder = r"C:\train diseases"
subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

for folder in subfolders:
    image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        results = model(image_path, save=False)
        frame = cv2.imread(image_path)

        for i, r in enumerate(results):
            for box in r.boxes:
                b = box.xyxy[0]
                x_min, y_min, x_max, y_max = map(int, b)
                original_area = 299 * 299
                cropped_area = (x_max - x_min) * (y_max - y_min)
                area_ratio = cropped_area / original_area

                adjustment_factor = 0.25
                expansion = 28
                if area_ratio > 0.8:
                    expansion -= int(expansion * adjustment_factor)
                elif area_ratio < 0.5:
                    expansion += int(expansion * adjustment_factor)
                expansion = max(5, min(expansion, 50))

                x_min_exp = max(0, x_min - expansion)
                y_min_exp = max(0, y_min - expansion)
                x_max_exp = min(frame.shape[1], x_max + expansion)
                y_max_exp = min(frame.shape[0], y_max + expansion)

                crop_width = x_max_exp - x_min_exp
                crop_height = y_max_exp - y_min_exp
                center_x = (x_max_exp + x_min_exp) // 2
                center_y = (y_max_exp + y_min_exp) // 2

                x_min_crop = max(0, center_x - (crop_width // 2))
                y_min_crop = max(0, center_y - (crop_height // 2))
                x_max_crop = min(frame.shape[1], center_x + (crop_width // 2))
                y_max_crop = min(frame.shape[0], center_y + (crop_height // 2))

                cropped_image = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

                output_folder = os.path.join(
                    r"C:\Users\kalai\Pictures\refined\train",
                    os.path.basename(folder)
                )
                os.makedirs(output_folder, exist_ok=True)

                name, _ = os.path.splitext(image_file)
                output_path = os.path.join(output_folder, f"{name}_{i}.jpg")
                cv2.imwrite(output_path, cropped_image)
