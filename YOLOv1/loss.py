import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOv1Loss(nn.Module):
    def __init__(self, config):
        super(YOLOv1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction= "sum")

        self.S = config["split_size"]
        self.B = config["number_boxes"]
        self.C = config["number_classes"]

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions are size (BATCH_SIZE, S*S*(C + B*5)) -> change to (BATCH_SIZE, S, S, C + B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # Calculate IoU for 2 predicted bboxes and targets
        # print(predictions[..., -9:-5].shape)
        iou_b1 = intersection_over_union(predictions[..., -9:-5], targets[..., -9:-5])
        iou_b2 = intersection_over_union(predictions[..., -4:], targets[..., -9:-5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim= 0)

        # Take the box with highest IoU out of the two prediction
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = targets[..., -5].unsqueeze(3)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., -4:]
                + (1 - bestbox) * predictions[..., -9:-5]
            )
        )

        box_targets = exists_box * targets[..., -9:-5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., -5:-4] + (1 - bestbox) * predictions[..., -10:-9]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., -10:-9]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., -10:-9], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., -10:-9], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., -5:-4], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., -10:-9], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :-10], end_dim=-2,),
            torch.flatten(exists_box * targets[..., :-10], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
