import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_color_constancy(image, gamma=2.2, p=6):
    I = image.astype(np.float32)
    J = 255.0 * np.power(I / 255.0, 1.0 / gamma)
    
    rgb_vector = np.power(np.mean(np.power(J, p), axis=(0, 1)), 1.0 / p)
    rgb_norm = rgb_vector / np.sqrt(np.sum(np.power(rgb_vector, 2)) + 1e-6)
    norm_z = 1.0 / (rgb_norm * np.sqrt(3.0))
    
    K = J * norm_z
    L = np.clip(K, 0, 255).astype(np.uint8)
    
    return L


def apply_clahe(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a, b))
    image_clahe_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    return image_clahe_rgb



def process_directory(input_dir, output_dir, use_clahe=True, use_hair_removal=False):
    """Processes a directory of images with selected enhancement pipelines."""
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    for filename in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(input_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        processed_rgb = apply_color_constancy(img_rgb, gamma=2.2, p=6)
        
        if use_clahe:
            processed_rgb = apply_clahe(processed_rgb, clip_limit=2.0)

        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, processed_bgr)


if __name__ == "__main__":
    DIRECTORY_INPUT = "raw_images"  
    DIRECTORY_OUTPUT = "images"

    process_directory(
        input_dir=DIRECTORY_INPUT, 
        output_dir=DIRECTORY_OUTPUT, 
        use_clahe=True
    )