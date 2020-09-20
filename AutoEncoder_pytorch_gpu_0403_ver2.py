# change part
# change the last dimention of feauture 2 to 10
# change epochs and val_nums
# change out_dir

# coding: utf-8

# In[ ]:

import time
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F
import sys, os
from PIL import Image
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader

start_time = time.time()

# 点描用関数
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# データ成形用関数
def to_img(x):
#     x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# ネットワーク定義
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 392),
            nn.ReLU(True),
            nn.Linear(392, 196),
            nn.ReLU(True),
            nn.Linear(196, 98),
            nn.ReLU(True),
            nn.Linear(98, 49),
            nn.ReLU(True),
            nn.Linear(49, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 49),
            nn.ReLU(True),
            nn.Linear(49, 98),
            nn.ReLU(True),
            nn.Linear(98, 196),
            nn.ReLU(True),
            nn.Linear(196, 392),
            nn.ReLU(True),
            nn.Linear(392, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available!')

else:
    print('cuda is not available')


# 保存用ディレクトリ作成
out_dir = './autoencoder_0403_gpu_ver2'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epoch数の定義
num_epochs = 10000

# crossvalidation の回数
validation_number = 5
vest_validation_num = 0
vest_validation_score = 1

for val_num in range(validation_number):
    val_start_time = time.time()
    print('validation : {}'.format(val_num + 1))

    img_transform = transforms.Compose([
    # torchvision.transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0,1] => [-1,1]
    ])

    # meke a dataset train and validation 7 :3
    trainval_dataset = datasets.MNIST('./', train=True, download=True, transform=img_transform)

    n_samples = len(trainval_dataset) # n_samples is 60000
    train_size = int(len(trainval_dataset) * 0.7) # train_size is 42000
    val_size = n_samples - train_size # val_size is 42000

    # shuffleしてから分割
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)

    validation_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # ネットワークのインスタンスを生成
    model = Autoencoder().to(device)

    # lossを保存するリストを初期化
    train_loss_list = []
    validation_loss_list = []
    val_sum = 0
    for epoch in range(num_epochs):

        if (epoch + 1) % 10 == 0:
            print("now is {} epoch".format(epoch + 1))

        # 損失関数を定義
        loss_function = nn.MSELoss()

        # 最適化関数の定義
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        for img, labels in train_loader:
            x = img.view(img.size(0), -1)

            # 今回はGPUを使用
            if cuda:
                x = Variable(x).cuda()
            else:
                x = Variable(x)


            x_modeled_train  = model(x)

            loss = loss_function(x_modeled_train, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.data)

        print("epoch[{}/{}], train_loss: {}".format(epoch + 1, num_epochs, loss.data))
        for img, labels in validation_loader:
            x = img.view(img.size(0), -1)
            # 今回はGPUを使用
            if cuda:
                x = Variable(x).cuda()
            else:
                x = Variable(x)

            x_modeled_val = model(x)
            loss = loss_function(x_modeled_val, x)
            validation_loss_list.append(loss.data)

    for val_loss in validation_loss_list:
        val_sum += val_loss
    val_score = val_sum / len(validation_loss_list)
    print("validation_loss_{}: {}".format(val_num + 1, val_score))

    pic_origin = to_img(x.cpu())
    pic_changed = to_img(x_modeled_val.cpu().data)
    save_image(pic_changed, './{}/cha_image_validation_{}.png'.format(out_dir, val_num + 1))
    save_image(pic_origin, './{}/ori_image_validation_{}.png'.format(out_dir, val_num + 1))
    torch.save(model.state_dict(), './{}/validation_{}_state_dict.pth'.format(out_dir, val_num + 1))
    if val_score < vest_validation_score:
        vest_validation_score = val_score
        vest_validation_num = val_num + 1
        print("validation{} is best loss score: {}".format(val_num + 1, vest_validation_score))

    val_finish_time = time.time()
    print("validation_cost_time: {}s".format(val_finish_time - val_start_time))

print("best validation: {}".format(val_num + 1))
print("/n")
print("best loss score: {}".format(vest_validation_score))
finish_time = time.time()
print("total_costtime: {}s".format(finish_time - start_time))
