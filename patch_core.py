from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter, zoom
import pickle


def infer_patch_grid(num_patches: int) -> tuple[int, int]:
    side = int(np.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patches to square segmentation map.")
    return side, side


def sample_features(
    features: np.ndarray,
    rng: np.random.Generator,
    sample_ratio: float,
    max_patches_per_image: int,
) -> np.ndarray:
    n = len(features)
    k = max(1, int(n * sample_ratio))
    k = min(k, max_patches_per_image, n)
    idx = rng.choice(n, size=k, replace=False)
    return features[idx]

def add_spatial_coordinates(features: np.ndarray, weight: float = 10.0) -> np.ndarray:
    """
    Dokleja znormalizowane współrzędne 2D do wektorów cech.
    Oryginalny kształt: (num_patches, channels)
    Nowy kształt: (num_patches, channels + 2)
    """
    num_patches, channels = features.shape
    grid_h, grid_w = infer_patch_grid(num_patches)
    
    # Tworzymy znormalizowaną siatkę współrzędnych od -1.0 do 1.0
    y_coords = np.linspace(-1.0, 1.0, grid_h)
    x_coords = np.linspace(-1.0, 1.0, grid_w)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    # Spłaszczamy siatkę i łączymy w pary (Y, X)
    # Kolejność odpowiada spłaszczaniu tensora w Twoim ekstraktorze cech
    spatial_features = np.stack([yv.flatten(), xv.flatten()], axis=1).astype(np.float32)
    
    # Mnożymy przez wagę - to NAJWAŻNIEJSZY hiperparametr!
    spatial_features *= weight
    
    # Doklejamy współrzędne do cech wizualnych
    return np.concatenate([features, spatial_features], axis=1)

def preprocess_image(
    image_path: Path, size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def create_knn_memory_bank(
    session,
    train_dir,
    coreset_ratio=0.03,
    max_patches=12000,
    max_patches_per_image=256,
    index_path: Path | str = "knn_cable.pkl",
):
    seed = 42
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    train_images = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])

    if not train_images:
        raise RuntimeError("No training images found.")

    # Create compact memory bank from good images.
    rng = np.random.default_rng(seed)
    memory_chunks = []
    for image_path in train_images:
        arr = preprocess_image(image_path=image_path)
        features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
        features = add_spatial_coordinates(features)
        sampled = sample_features(features, rng, sample_ratio=1.0, max_patches_per_image=max_patches_per_image)
        memory_chunks.append(sampled)
    memory_bank = np.concatenate(memory_chunks, axis=0).astype(np.float32)

    # Global coreset sampling.
    k = max(1, int(len(memory_bank) * coreset_ratio))
    k = min(k, max_patches, len(memory_bank))
    selected_indices = rng.choice(len(memory_bank), size=k, replace=False)
    memory_bank = memory_bank[selected_indices]

    nn_index = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    nn_index.fit(memory_bank)

    with open(index_path, "wb") as f:
        pickle.dump(nn_index, f)    
    print(f"Train images: {len(train_images)}")
    print(f"Memory bank shape: {memory_bank.shape}")
    return nn_index

def compute_best_threshold(session, train_dir, nn_index, visualize=False):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    train_images = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])

    # Compute anomaly scores for good images and plot histogram
    y_score_image = []
    all_patch_scores = []
    for image_path in train_images:
        arr = preprocess_image(image_path=image_path)
        features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
        features = add_spatial_coordinates(features)

        distances, _ = nn_index.kneighbors(features, n_neighbors=1, return_distance=True)
        dist_score = distances[:, 0]

        image_score = float(np.max(dist_score))
        y_score_image.append(image_score)

        all_patch_scores.extend(dist_score.tolist())

    image_threshold = float(np.mean(y_score_image) + 2.0 * np.std(y_score_image))
    
    # --- OBLICZANIE PROGU PIKSELA (Segmentacja maski) ---
    all_patch_scores = np.array(all_patch_scores)
    # Wyznaczamy 99.5 percentyl ze wszystkich patchy.
    # Oznacza to, że tylko 0.5% najbardziej nietypowych patchy z dobrych zdjęć uznamy za szum.
    pixel_threshold = float(np.percentile(all_patch_scores, 99.5))

    print(f"Image threshold (OK/NOK) = {image_threshold:.6f}")
    print(f"Pixel threshold (Maska)  = {pixel_threshold:.6f}")

    if visualize:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_score_image, bins=30, color='blue', alpha=0.7)
        plt.axvline(x=image_threshold, color="r", linestyle='--', label=f'Threshold: {image_threshold:.2f}')
        plt.title("Histogram - Max Image Scores")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(all_patch_scores, bins=50, color='green', alpha=0.7)
        plt.axvline(x=pixel_threshold, color="r", linestyle='--', label=f'Threshold: {pixel_threshold:.2f}')
        plt.title("Histogram - All Pixel/Patch Scores")
        plt.legend()
        
        plt.show()

    return image_threshold, pixel_threshold

def load_knn_memory_bank(index_path: Path | str = "knn_cable.pkl"):
    with open(index_path, "rb") as f:
        nn_index = pickle.load(f)
    return nn_index

def visualize_best_threshold_test(session, test_root, nn_index, best_threshold):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    if not test_root.exists():
        raise FileNotFoundError(f"Missing test directory: {test_root}")

    y_score_nok = []
    y_true_nok = []

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "good":
            continue

        class_images = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
        for image_path in class_images:
            arr = preprocess_image(image_path=image_path)
            features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
            features = add_spatial_coordinates(features)
            distances, _ = nn_index.kneighbors(features, n_neighbors=1, return_distance=True)
            dist_score = distances[:, 0]
            image_score = float(np.max(dist_score))

            y_score_nok.append(image_score)
            y_true_nok.append(1)

    if not y_score_nok:
        raise RuntimeError("No NOK images found in dataset/test.")

    y_score_nok = np.array(y_score_nok, dtype=np.float32)
    nok_detected = int(np.sum(y_score_nok >= best_threshold))
    nok_total = int(len(y_score_nok))
    print(f"NOK images: {nok_total}")
    print(f"NOK detected above threshold: {nok_detected}/{nok_total}")

    plt.figure(figsize=(8, 4))
    plt.hist(y_score_nok, bins=50)
    plt.vlines(x=best_threshold, ymin=0, ymax=max(1, nok_total // 3), color="r")
    plt.title("Histogram of scores for NOK images")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.show()


def predict(session, nn_index, threshold_image, threshold_pixel, image_path):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    arr = preprocess_image(image_path=image_path)
    fault_type = image_path.parent.name

    features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
    features = add_spatial_coordinates(features)
    distances, _ = nn_index.kneighbors(features, n_neighbors=1, return_distance=True)
    dist_score = distances[:, 0]

    image_score = float(np.max(dist_score))
    y_pred_image = (image_score >= threshold_image)
    class_label = ['OK','NOK']
    
    grid_h, grid_w = infer_patch_grid(len(dist_score))
    segm_map = dist_score.reshape((grid_h, grid_w))
    
    zoom_factor = (224 / grid_h, 224 / grid_w)
    segm_map_resized = zoom(segm_map, zoom_factor, order=1)
    # sigma=4.0 lub 8.0 sprawdza się świetnie dla mapy 224x224
    heat_map = gaussian_filter(segm_map_resized, sigma=8.0)
    binary_mask = heat_map > threshold_pixel

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_rgb = np.transpose(arr[0], (1, 2, 0)) * std + mean
    img_rgb = np.clip(img_rgb, 0.0, 1.0)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(img_rgb)
    plt.title(f'{fault_type}')
    plt.axis('off')

    # Wykres 2: Heatmapa nałożona na obraz (Overlay)
    plt.subplot(1, 3, 2)
    plt.imshow(img_rgb) # Podkład
    # Fix: usunięto błędne vmin=vmax, dodano przezroczystość (alpha)
    plt.imshow(heat_map, cmap='jet', alpha=0.5) 
    plt.title(f'Score: {image_score:.2f} (Próg: {best_threshold:.2f}) \nPredykcja: {class_label[y_pred_image]}')
    plt.axis('off')

    # Wykres 3: Czysta maska segmentacji (binarna)
    plt.subplot(1, 3, 3)
    # Maska to piksele, które przekroczyły próg
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Maska segmentacji')
    plt.axis('off')
        
    plt.show()
    


if __name__ == "__main__":
    # Load ONNX model
    model_path = Path("resnet50_extractor.onnx")
    train_dir = Path("/home/igorsiata/studia/algorytmy_wizyjne/cable_defect_detection/dataset/train/good")
    test_dir = Path("/home/igorsiata/studia/algorytmy_wizyjne/cable_defect_detection/dataset/test")
    coreset_ratio = 0.1
    max_patches = 12000
    max_patches_per_image = 512
    index_path = Path("knn_cable.pkl")
    load_knn = False
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    best_threshold = 20.935613

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not (0.0 < coreset_ratio <= 1.0):
        raise ValueError("coreset_ratio must be in (0, 1].")
    if max_patches <= 0:
        raise ValueError("max_patches must be > 0.")
    if max_patches_per_image <= 0:
        raise ValueError("max_patches_per_image must be > 0.")
    
    if load_knn:
        nn_index = load_knn_memory_bank(index_path)
    else:
        nn_index = create_knn_memory_bank(
            session,
            train_dir,
            coreset_ratio=coreset_ratio,
            max_patches=max_patches,
            max_patches_per_image=max_patches_per_image,
            index_path=index_path,
        )

    threshold_image, threshold_pixel = compute_best_threshold(session, train_dir, nn_index, True)
    # visualize_best_threshold_test(session, test_dir, nn_index, best_threshold)

    for class_dir in sorted(test_dir.iterdir()):
        class_images = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
        for image_path in class_images[3:4]:
            predict(session, nn_index, threshold_image, threshold_pixel, image_path)
    
