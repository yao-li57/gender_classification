import os.path
import time
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Resnet_50
from create_dataset import read_split_data, MyDataset


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))

    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print("using device: ", device)

    # 读取数据，构建dataset, dataloader
    train_imgs_path = "./BitmojiDataset_Sample/trainimages"
    train_labels_path = "./BitmojiDataset_Sample/train.csv"
    model_save_path = "./model_save"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    train_imgs, train_labels, val_imgs, val_labels = read_split_data(train_imgs_path, train_labels_path, val_ratio=0.3, plot=True)
    train_dataset = MyDataset(train_imgs, train_labels, transform)
    val_dataset = MyDataset(val_imgs, val_labels, transform)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)

    # 创建模型
    model = Resnet_50()
    model.to(device)
    # model.Init_weight()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    lr = 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 每3个epoch变为0.1倍

    epochs = 20
    num_iters = 0

    train_losses = []
    train_acc = []
    val_acc = []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        # train
        train_pred_true_num = 0
        train_loss = []
        # tqdm 1
        loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        for _, (x, target) in loop:
            x = x.to(device)
            target = target.to(device)

            predict = model(x)
            loss = loss_fn(predict, target)
            train_pred_true_num += (predict.argmax(1) == target).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iters += 1
            train_losses.append(loss.item())
            train_loss.append(loss.item())

            # tqdm 2 更新信息
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss=loss.item(), acc="{:.2f}%".format(train_pred_true_num / len(train_dataset)*100),
                             lr=f"{optimizer.param_groups[0]['lr']}",
                             running_time='{:.2f}s'.format(time.perf_counter()-start_time))
        scheduler.step()
        acc = train_pred_true_num / len(train_dataset)
        train_acc.append(acc)

        torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_{}_{:.2f}%.pth".format(epoch+1, acc*100)))

        print("\nepoch{}: average_train_loss {}".format(epoch + 1, np.mean(train_loss)))
        print("epoch{}: train_acc {:.2f}%".format(epoch + 1, acc * 100))

        # test
        model.eval()
        val_pred_true_num = 0
        val_loss = []
        with torch.no_grad():
            for data in val_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                val_loss.append(loss.item())
                val_pred_true_num += (outputs.argmax(1) == targets).sum()
        acc = val_pred_true_num / len(val_dataset)
        val_acc.append(acc)

        print("\nepoch{}: average_test_loss:{}".format(epoch + 1, np.mean(val_loss)))
        print("epoch{}: val_acc:{:.2f}%".format(epoch + 1, acc * 100))
