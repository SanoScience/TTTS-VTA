import torch
from torchmetrics.classification import JaccardIndex, Dice, Precision, Recall
from torchmetrics import Accuracy
import cv2 as cv


def evaluate_segmentation_metrics(pred_mask, true_mask):
    """
    Function to calculate IoU, Dice Coefficient, Pixel Accuracy, Precision, and Recall for segmentation masks.

    Parameters:
    - pred_mask (Tensor): Predicted segmentation mask.
    - true_mask (Tensor): Ground truth segmentation mask.
    - num_classes (int): Number of classes. Use 2 for binary segmentation or more for multi-class.

    Returns:
    - A dictionary containing the metrics.
    """

    # Binary case
    iou_metric = JaccardIndex(task="binary")
    dice_metric = Dice()
    accuracy_metric = Accuracy(task="binary")
    precision_metric = Precision(task="binary")
    recall_metric = Recall(task="binary")


    # Calculate metrics
    iou = iou_metric(pred_mask, true_mask)
    dice = dice_metric(pred_mask, true_mask)
    accuracy = accuracy_metric(pred_mask, true_mask)
    precision = precision_metric(pred_mask, true_mask)
    recall = recall_metric(pred_mask, true_mask)

    # Return all metrics in a dictionary
    return {
        "IoU": iou.item(),
        "Dice Coefficient": dice.item(),
        "Pixel Accuracy": accuracy.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
    }


# Example usage with binary masks (1 for object, 0 for background)
pred_mask = cv.imread("2.png")  # Predicted mask
pred_mask = torch.tensor(pred_mask, dtype=torch.int64)

true_mask = cv.imread("3.png")  # Ground truth mask
true_mask = torch.tensor(true_mask, dtype=torch.int64)

# Binary segmentation case
metrics = evaluate_segmentation_metrics(pred_mask, true_mask)

# Print the metrics
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
