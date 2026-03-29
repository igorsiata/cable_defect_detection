from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import pickle


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

def crate_knn_memory_bank(session, train_dir, coreset_ratio=0.1):
    seed = 42
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    train_images = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])

    if not train_images:
        raise RuntimeError("No training images found.")

    # Create memory bank from good images
    memory_chunks = []
    for image_path in train_images:
        arr = preprocess_image(image_path=image_path)
        features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
        memory_chunks.append(features)
    memory_bank = np.concatenate(memory_chunks, axis=0).astype(np.float32)

    # keep 10% of memory bank
    rng = np.random.default_rng(seed)
    k = max(1, int(len(memory_bank) * coreset_ratio))
    selected_indices = rng.choice(len(memory_bank), size=k, replace=False)
    memory_bank = memory_bank[selected_indices]

    nn_index = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    nn_index.fit(memory_bank)

    with open("knn_index.pkl", "wb") as f:
        pickle.dump(nn_index, f)    
    print(f"Model: {model_path}")
    print(f"Train images: {len(train_images)}")
    print(f"Memory bank shape: {memory_bank.shape}")
    return nn_index

def compute_best_threshold(session, train_dir, nn_index, visualize=False):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    train_images = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])

    # Compute anomaly scores for good images and plot histogram
    y_score = []
    for image_path in train_images:
        arr = preprocess_image(image_path=image_path)
        features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
        distances, _ = nn_index.kneighbors(features, n_neighbors=1, return_distance=True)
        dist_score = distances[:, 0]
        image_score = float(np.max(dist_score))
        y_score.append(image_score)

    best_threshold = float(np.mean(y_score) + 2.0 * np.std(y_score))
    print(f"best_threshold = {best_threshold:.6f}")

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.hist(y_score, bins=50)
        plt.vlines(x=best_threshold, ymin=0, ymax=max(1, len(y_score) // 3), color="r")
        plt.title("Histogram of scores for good images")
        plt.xlabel("Anomaly score")
        plt.ylabel("Count")
        plt.show()

    return best_threshold

def load_knn_memory_bank():
    with open("knn_index.pkl", "rb") as f:
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


def predict(session, nn_index, best_threshold, image_path):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    arr = preprocess_image(image_path=image_path)
    fault_type = image_path.parent.name

    features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
    distances, _ = nn_index.kneighbors(features, n_neighbors=1, return_distance=True)
    dist_score = distances[:, 0]
    image_score = float(np.max(dist_score))

    segm_map = dist_score.reshape((28, 28))
    y_pred_image = 1*(image_score >= best_threshold)
    class_label = ['OK','NOK']

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_rgb = np.transpose(arr[0], (1, 2, 0)) * std + mean
    img_rgb = np.clip(img_rgb, 0.0, 1.0)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(img_rgb)
    plt.title(f'{fault_type}')

    plt.subplot(1,3,2)
    heat_map = segm_map
    plt.imshow(heat_map, cmap='jet',vmin=best_threshold, vmax=best_threshold*1) 
    plt.title(f'Anomaly score: {image_score / best_threshold:0.4f} || {class_label[y_pred_image]}')

    plt.subplot(1,3,3)
    plt.imshow((heat_map > best_threshold*0.95), cmap='gray')
    plt.title(f'segmentation map')
        
    plt.show()
    


if __name__ == "__main__":
    # Load ONNX model
    model_path = Path("resnet50_extractor.onnx")
    train_dir = Path("dataset/train/good")
    test_dir = Path("dataset/test")
    coreset_ratio = 0.1  # keep 10% of patches
    load_knn = True
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    best_threshold = 20.347673

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not (0.0 < coreset_ratio <= 1.0):
        raise ValueError("coreset_ratio must be in (0, 1].")
    
    if load_knn:
        nn_index = load_knn_memory_bank()
    else:
        nn_index = crate_knn_memory_bank(session, train_dir, 0.1)

    # best_threshold = compute_best_threshold(session, train_dir, nn_index, True)
    # visualize_best_threshold_test(session, test_dir, nn_index, best_threshold)

    for class_dir in sorted(test_dir.iterdir()):
        class_images = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
        for image_path in class_images[3:4]:
            predict(session, nn_index, best_threshold, image_path)
    
