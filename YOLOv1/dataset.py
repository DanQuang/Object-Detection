import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms

class VOCDataset(Dataset):
    def __init__(self, config):
        super(VOCDataset, self).__init__()

        self.annotations = pd.read_csv(config["data"]["csv_file"])
        self.img_dir = config["data"]["img_dir"]
        self.label_dir = config["data"]["label_dir"]
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        self.S = config["split_size"]
        self.B = config["number_boxes"]
        self.C = config["number_classes"]

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, w, h])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        image = self.transform(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S

            if label_matrix[i, j, -10] == 0:
                label_matrix[i, j, -10] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                label_matrix[i, j, -9:-5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return {
            "image": image,
            "label_matrix": label_matrix
        }
    
class Load_Data:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.dataset = VOCDataset(config)

        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(self.dataset, [0.7, 0.1, 0.2])

    def load_train_dev(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size= self.train_batch,
                                      shuffle= True)
        
        dev_dataloader = DataLoader(self.dev_dataset,
                                    batch_size= self.dev_batch,
                                    shuffle= False)
        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataloader = DataLoader(self.test_dataset,
                                    batch_size= self.test_batch,
                                    shuffle= False)
        return test_dataloader