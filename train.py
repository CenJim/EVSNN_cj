import argparse
import copy

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import torch.nn as nn
import lpips
from model.snn_network import EVSNN_LIF_final, PAEVSNN_LIF_AMPLIF_final
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image


class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 包含所有sequence文件夹的根目录
        transform: torchvision.transforms 对象，用于对输出图像进行处理
        """
        self.root_dir = root_dir
        self.crop_transform = RandomCropTransform(size=(128, 128))
        self.transform = transform
        self.pairs = []

        # search for all sequence dirs
        sequences = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir)) if
                     os.path.isdir(os.path.join(root_dir, d))]

        for seq in sequences:
            input_folder = os.path.join(seq, 'VoxelGrid-betweenframes-5')
            output_folder = os.path.join(seq, 'frames')

            # 获取所有输入文件和输出文件
            input_files = [f for f in sorted(os.listdir(input_folder)) if os.path.splitext(f)[1] == '.npy']
            output_files = [f for f in sorted(os.listdir(output_folder)) if
                            (os.path.splitext(f)[1] == '.png' and f != 'frame_0000000000.png')]

            # 确保输入和输出数量相同
            for input_file, output_file in zip(input_files, output_files):
                self.pairs.append((os.path.join(input_folder, input_file), os.path.join(output_folder, output_file)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get the index of the layer of the voxel grid
        # layer_index = int(idx % 5)
        # file_index = int(np.floor(idx / 5))

        # get the file path of the voxel grid
        pair = self.pairs[idx]

        # 读取数据
        input_data = np.load(pair[0])
        output_image = Image.open(pair[1])

        # # 转换输入数据为torch.Tensor
        transform_img_tensor = transforms.ToTensor()
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = transform_img_tensor(output_image)

        # 如果有传入转换器，则应用转换
        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        input_tensor, output_tensor = self.crop_transform(input_tensor, output_tensor)

        return input_tensor, output_tensor


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        lpips_value = self.lpips_loss(pred, target)
        l1_value = self.l1_loss(pred, target)
        return torch.mean(lpips_value + l1_value)


class RandomCropTransform:
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, img, target):
        # 获取随机裁剪的参数
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.size)

        # 应用同样的裁剪到图像和目标
        img_cropped = TF.crop(img, i, j, h, w)
        target_cropped = TF.crop(target, i, j, h, w)

        return img_cropped, target_cropped


def main(model_name: str, pretrain_models: str, root_files: str, save_path: str, height: int, width: int,
         num_events_per_pixel: float):
    # Parameters
    epochs = 100
    batch_size = 8
    learning_rate = 0.002
    crop_size = 128
    time_steps = 5  # Calculate loss every 5 timesteps
    kwargs = {'activation_type': 'lif', 'v_threshold': 1., 'v_reset': 0., 'tau': 2.}

    # Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EVSNN_LIF_final(kwargs).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fun = CombinedLoss()

    # Data loading and transformations

    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])  # Normalize event tensors
    ])
    dataset = SequenceDataset(root_files, transform=transform)  # Placeholder for your dataset class
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    # mem_states
    output = torch.zeros(8, 1, 128, 128)
    for epoch in range(epochs):
        for i, (events, targets) in enumerate(dataloader):
            events, targets = events.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(events, output)  # model might need modification to handle states
            output = output.detach()

            if (i + 1) % time_steps == 0:
                loss = loss_fun(output, targets)
                loss.backward()
                optimizer.step()
                # model.reset_states()  # Reset states if your model has this functionality

            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss:  ")
    # 保存训练好的模型
    model_path = 'pretrained_models/testEVSNN.pth'
    torch.save(model.state_dict(), model_path)
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-network', type=str, default='EVSNN_LIF_final')  # EVSNN_LIF_final or PAEVSNN_LIF_AMPLIF_final
    parser.add_argument('-path_to_pretrain_models', type=str, default='./pretrained_models/EVSNN.pth')  #
    parser.add_argument('-path_to_root_files', type=str, default='./data/poster_6dof_cut.txt')
    parser.add_argument('-save_path', type=str, default='./results')
    parser.add_argument('-height', type=int, default=180)
    parser.add_argument('-width', type=int, default=240)
    parser.add_argument('-num_events_per_pixel', type=float, default=0.5)
    args = parser.parse_args()
    model_name = args.network
    pretrain_models = args.path_to_pretrain_models
    root_files = args.path_to_root_files
    save_path = args.save_path
    height = args.height
    width = args.width
    num_events_per_pixel = args.num_events_per_pixel
    main(model_name, pretrain_models, root_files, save_path, height, width, num_events_per_pixel)
