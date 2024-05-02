import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as p

def intersection_over_union(boxes_preds, boxes_labels, box_format = "midpoint"):
    # boxes shape (N, 4) where N is the number of boxes
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for when they don't intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # calculate union
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)

def non_max_suppression(bboxes, prob_threshold, iou_threshold, box_format = "midpoint"):
    # bboxes = [[1, 0.9, x1, y1, x2, y2], [], [], ...]
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse= True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format= box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold= 0.5, box_format= "midpoint", num_classes= 20):
    # pred_boxes (list): [[train_idx, class_pred, prob, x1, y1, x2, y2], ...]
    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        # amount_bboxes = {0: tensor([0, 0, 0]), 1: tensor([0,0,0,0,0])}

        detections.sort(key= lambda x: x[2], reverse= True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_imgs = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth_imgs):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format= box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1

                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # [1,1,0,1,0] -> [1,2,2,3,3]
        TP_cumsum = torch.cumsum(TP, dim= 0)            
        FP_cumsum = torch.cumsum(FP, dim= 0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        average_precision.append(torch.trapz(precisions, recalls))

    return sum(average_precision) / num_classes

def plot_image_with_boxes(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    h, w, c = im.shape
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    for box in boxes:
        box = box[2:]
        assert len(box) == 4
        x = box[0] - box[2] / 2
        y = box[1] - box[3] / 2
        rect = p.Rectangle(
            (x * w, y * h),
            box[2] * w,
            box[3] * h,
            linewidth = 1,
            edgecolor = 'r'
        )
        ax.add_patch(rect)

    plt.show()

def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., -9:-5]
    bboxes2 = predictions[..., -4:]
    scores = torch.cat((predictions[..., -10].unsqueeze(0), predictions[..., -5].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :-10].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., -10], predictions[..., -5]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def get_bboxes(loader, model, iou_threshold, threshold, device, box_format="midpoint"):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    train_idx = 0

    for _, item in enumerate(loader):
        X, y = item["image"].to(device), item["label_matrix"].to(device)

        predictions = model(X)

        batch_size = X.shape[0]
        true_bboxes = cellboxes_to_boxes(y)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    return all_pred_boxes, all_true_boxes