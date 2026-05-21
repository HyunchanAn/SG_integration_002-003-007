import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from mobile_sam import sam_model_registry, SamPredictor
from vsams.models.classifier import SurfaceClassifier
from torchvision import transforms


def index_dataset(
    dataset_root="dataset/260414", output_path="vsams/data/visual_library.pth"
):
    print("Initializing SAM and Classifier...")
    # 1. Setup Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MobileSAM (vit_t)
    sam_checkpoint = "checkpoints/mobile_sam.pt"
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # V-SAMS Classifier
    classifier = SurfaceClassifier()
    checkpoint_path = "checkpoints/v_sams_model.pth"
    if os.path.exists(checkpoint_path):
        classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # 2. Walk through dataset
    library = []

    # Subdirectories correspond to product names
    if not os.path.exists(dataset_root):
        print(f"Dataset root {dataset_root} does not exist!")
        return

    product_folders = sorted(
        [
            f
            for f in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, f))
        ]
    )
    print(f"Found {len(product_folders)} product folders.")

    for folder in tqdm(product_folders, desc="Processing Products"):
        folder_path = os.path.join(dataset_root, folder)
        product_name = (
            folder.split("_", 2)[-1] if len(folder.split("_")) >= 3 else folder
        )

        # Collect surface images (_01.jpg, _02.jpg, ...)
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        surface_images = []
        for f in all_files:
            name_part = os.path.splitext(f)[0]
            parts = name_part.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                surface_images.append(f)

        print(f"Folder {folder}: Found {len(surface_images)} surface images.")

        # ID images (for visual reference display)
        id_images = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith(".jpg") and f not in surface_images
        ]
        ref_image_path = os.path.join(folder_path, id_images[0]) if id_images else None

        features_list = []

        for img_name in surface_images:
            img_path = os.path.join(folder_path, img_name)
            try:
                # Load image with support for Korean paths
                img_array = np.fromfile(img_path, np.uint8)
                cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if cv_img is None:
                    print(f"Warning: Failed to decode {img_path}")
                    continue

                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, _ = cv_img.shape

                # SAM Masking
                predictor.set_image(cv_img)
                # Prompt: Center point
                input_point = np.array([[w // 2, h // 2]])
                input_label = np.array([1])

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
                mask = masks[0]

                # Apply Mask: Black out background
                masked_img = cv_img.copy()
                masked_img[~mask] = 0

                # Crop to mask bounding box
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    y1, y2, x1, x2 = (
                        y_indices.min(),
                        y_indices.max(),
                        x_indices.min(),
                        x_indices.max(),
                    )
                    masked_img = masked_img[y1:y2, x1:x2]

                # Convert to PIL for Classifier
                pil_img = Image.fromarray(masked_img)
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

                # Extract Features
                with torch.no_grad():
                    features = classifier.extract_features(input_tensor)
                    features_list.append(features.cpu().numpy().flatten())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if features_list:
            library.append(
                {
                    "product_name": product_name,
                    "ref_image": ref_image_path,
                    "features": np.array(features_list),
                }
            )

    # 3. Save Library
    torch.save(library, output_path)
    print(f"Library saved to {output_path} with {len(library)} products.")


if __name__ == "__main__":
    index_dataset()
