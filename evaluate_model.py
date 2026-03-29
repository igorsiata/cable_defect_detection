import numpy as np
import cv2
import os
import model
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from model import predict


@dataclass
class ImageEval:
    img_path: str
    gt_path: str
    defect_type: str


def load_images(directory) -> List[ImageEval]:
    images = []
    defect_types = set()
    images_dir = os.path.join(directory, "test")
    gt_dir = os.path.join(directory, "ground_truth")
    for defect_type in os.listdir(images_dir):
        defect_types.add(defect_type)
        defect_path_ref = os.path.join(images_dir, defect_type)
        defect_path_gt = os.path.join(gt_dir, defect_type)

        if os.path.isdir(defect_path_ref):
            for img_file in os.listdir(defect_path_ref):
                if defect_type == "good":
                    gt_img_path = None
                else:
                    gt_img_path = os.path.join(
                        defect_path_gt, img_file[:-4] + "_mask.png"
                    )
                img = ImageEval(
                    os.path.join(defect_path_ref, img_file), gt_img_path, defect_type
                )
                images.append(img)
    return images, list(defect_types)


def evaluate_model(images: List[ImageEval], defect_types: List[str], pred_fcn):
    metrics = {
        df: {"precision": [], "recall": [], "f1": [], "iou": []}
        for df in defect_types
    }
    image_iou_scores = []
    for image_eval in images:
        img = cv2.imread(image_eval.img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if image_eval.defect_type == "good":
            gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            gt = cv2.imread(image_eval.gt_path, cv2.IMREAD_GRAYSCALE)
            _, gt = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

        pred = pred_fcn(img_rgb)
        pred = np.asarray(pred)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]

        pred_bool = pred > 0
        gt_bool = gt > 0

        TP = np.sum(pred_bool & gt_bool)  # wykryto obiekt i faktycznie jest obiekt
        FP = np.sum(pred_bool & ~gt_bool)  # wykryto obiekt, ale w GT go nie ma
        FN = np.sum(~pred_bool & gt_bool)  # nie wykryto, a obiekt byl
        TN = np.sum(~pred_bool & ~gt_bool)  # poprawnie tlo

        union = np.sum(pred_bool | gt_bool)
        # For good images with empty GT and empty prediction, treat IoU as perfect.
        iou_image = float(TP / union) if union > 0 else 1.0
        image_iou_scores.append(iou_image)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0
            else 0.0
        )
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        metrics[image_eval.defect_type]["precision"].append(precision)
        metrics[image_eval.defect_type]["recall"].append(recall)
        metrics[image_eval.defect_type]["f1"].append(f1)
        metrics[image_eval.defect_type]["iou"].append(iou)

    return metrics, image_iou_scores


def show_examples_per_class(images: List[ImageEval], defect_types: List[str], pred_fcn):
    # Pick one representative sample per class (first occurrence in dataset order).
    sample_by_class = {}
    for image_eval in images:
        if image_eval.defect_type not in sample_by_class:
            sample_by_class[image_eval.defect_type] = image_eval

    ordered_classes = sorted(defect_types)
    cols = len(ordered_classes)
    fig, axes = plt.subplots(3, cols, figsize=(3 * cols, 10))

    if cols == 1:
        axes = np.array(axes).reshape(3, 1)

    row_labels = ["image", "pred_mask", "ground_truth_mask"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, rotation=90, size=10)

    for col, defect_type in enumerate(ordered_classes):
        image_eval = sample_by_class.get(defect_type)
        axes[0, col].set_title(defect_type)
        if image_eval is None:
            for row in range(3):
                axes[row, col].axis("off")
            continue

        img = cv2.imread(image_eval.img_path)
        if img is None:
            for row in range(3):
                axes[row, col].axis("off")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = pred_fcn(img_rgb)
        pred = (pred > 0).astype(np.uint8)

        if image_eval.defect_type == "good":
            gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            gt = cv2.imread(image_eval.gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                _, gt = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

        axes[0, col].imshow(img_rgb)
        axes[1, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(gt, cmap="gray", vmin=0, vmax=1)

        for row in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def run_eval(pred_fcn, show_examples=False):
    images, defect_types = load_images("./dataset")
    metrics, image_iou_scores = evaluate_model(images, defect_types, pred_fcn)

    print("\n Competition IoU (mean over all test images, good + defective):")
    print("\tIoU:", np.mean(image_iou_scores) if image_iou_scores else 0.0)

    if show_examples:
        show_examples_per_class(images, defect_types, pred_fcn)


if __name__ == "__main__":
    run_eval(predict, show_examples=True)
