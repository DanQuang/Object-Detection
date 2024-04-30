import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model import YOLOv1
from dataset import Load_Data
from utils import mean_average_precision, get_bboxes
from loss import YOLOv1Loss

class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # init model
        S = config["split_size"]
        B = config["number_boxes"]
        C = config["number_classes"]
        self.model = YOLOv1(split_size = S, num_boxes = B, num_classes = C).to(self.device)

        self.dataloader = Load_Data(config)
        self.loss = YOLOv1Loss(config)
        self.optim = optim.Adam(self.model.parameters(), lr= self.learning_rate, weight_decay= 0)

    def train(self):
        train, dev = self.dataloader.load_train_dev()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        last_model = "YOLOv1_last_model.pth"
        best_model = "YOLOv1_best_model.pth"

        if os.path.exists(os.path.join(self.save_path, last_model)):
                checkpoint = torch.load(os.path.join(self.save_path, last_model))
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optim.load_state_dict(checkpoint["optim_state_dict"])
                print("Load the last model")
                initial_epoch = checkpoint["epoch"] + 1
                print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, best_model)):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0

        self.model.train()
        for epoch in range(initial_epoch, initial_epoch + self.num_epochs):
            mean_loss = []

            for _, item in tqdm(enumerate(train)):
                self.optim.zero_grad()
                X, y = item["image"].to(self.device), item["label_matrix"].to(self.device)

                out = self.model(X)
                loss = self.loss(out, y)
                mean_loss.append(loss.item())

                loss.backward()
                self.optim.step()

            print(f"Epoch {epoch}: Train loss: {sum(mean_loss) / len(mean_loss)}")

            self.model.eval()
            with torch.inference_mode():
                pred_boxes, target_boxes = get_bboxes(dev, self.model, 0.5, 0.5, self.device)

                mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold= 0.5)

                print(f"mAP in Dev set: {mean_avg_prec}")

                # save last model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'score': mean_avg_prec
                }, os.path.join(self.save_path, last_model))

                if mean_avg_prec < best_score and epoch > 0:
                    threshold += 1
                else:
                    threshold = 0

                if mean_avg_prec >= best_score:
                    best_score = mean_avg_prec
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'score':mean_avg_prec
                    }, os.path.join(self.save_path, best_model))
                    print(f"Saved the best model with mAP of {mean_avg_prec:.4f}")

                # early stopping
                if threshold >= self.patience:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break