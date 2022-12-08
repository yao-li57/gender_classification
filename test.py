import os.path
import time
import torch
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from model import Resnet_50
from create_dataset import read_split_data, MyDataset

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("using device: ", device)

    test_imgs_path = "./BitmojiDataset_Sample/testimages"
    test_labels_output_path = "./BitmojiDataset_Sample/sample_submission.csv"
    df = pd.read_csv(test_labels_output_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    weight_filepath = "./model_save/epoch_20_98.81%.pth"
    model = Resnet_50()
    model.load_state_dict(torch.load(weight_filepath))
    model.to(device)

    cnt = 0
    model.eval()
    with torch.no_grad():
        for img_name in os.listdir(test_imgs_path):
            img_path = os.path.join(test_imgs_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img = img.to(device).unsqueeze(0)
            pred = int(model(img).argmax(1))
            df.iloc[cnt, 1] = -1 if pred == 0 else 1
            cnt += 1
    df.to_csv("./output.csv")
