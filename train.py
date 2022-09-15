from torch import optim
import time
from utils import *
from torch.utils.data import Dataset
from dataset import *
import matplotlib.pyplot as plt
from arc.resnet import *
# from arc.dncnn import *
# from mix_loss import MIX_LOSS
from ssim1 import ssim1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Ours()
net.to(device=device)

epochs = 50
lr = 0.005
batch_size = 128

optimizer = optim.Adam(net.parameters(), lr)
criterion = nn.MSELoss()

train_path_x = "D:\\Seismic\\seis_data\\patchs\\feature\\"
train_path_y = "D:\\Seismic\\seis_data\\patchs\\label\\"
# train_path_x = "data/feature/"
# train_path_y = "data//label/"

dataset = MyDataset(train_path_x, train_path_y)
print('Dataset size:', len(dataset))
valida_size = int(len(dataset) * 0.1)
train_size = len(dataset) - valida_size * 2
# 划分数据集
train_dataset, test_dataset, valida_dataset = torch.utils.data.random_split(dataset,
                                                                            [train_size, valida_size, valida_size])
# 加载并且乱序训练数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 加载并且乱序验证数据集
valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=batch_size, shuffle=True)
# 加载测试数据集,不做乱序处理
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

temp_sets1 = []  # 用于记录训练，验证集的loss,每一个epoch都做一次训练，验证
temp_sets2 = []  # 用于记录测试集的SNR
temp_sets3 = []
temp_sets4 = []

start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 开始时间
for epoch in range(epochs):
    # 训练集训练网络
    train_loss = 0.0
    net.train()  # 开启训练模式
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0):
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        out1 = net(batch_x)
        loss1 = criterion(out1, batch_y)
        train_loss += loss1.item()  # 累加计算本循环的loss
        optimizer.zero_grad()  # 梯度归零
        loss1.backward()  # 反向传播
        optimizer.step()
    train_loss = train_loss / (batch_idx1 + 1)  # 本次epoch的平均训练loss
    # 验证集验证网络
    net.eval()  # 开启评估/测试模式
    val_loss = 0.0
    for batch_idx2, (val_x, val_y) in enumerate(valida_loader, 0):
        # 加载数据至GPU
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            out2 = net(val_x)  # 使用网络参数，输出预测结果
            # 计算loss
            loss2 = criterion(out2, val_y)
            val_loss += loss2.item()  # 累加计算本循环的loss
    val_loss = val_loss / (batch_idx2 + 1)  # 本次epoch的平均验证loss
    # 训练，验证，测试的loss保存至loss_sets中
    loss_set = [train_loss, val_loss]
    temp_sets1.append(loss_set)

    snr_set1 = 0.0
    snr_set2 = 0.0
    psnr_set1 = 0.0
    psnr_set2 = 0.0
    ssim_set1 = 0.0
    ssim_set2 = 0.0
    for batch_idx3, (test_x, test_y) in enumerate(test_loader, 0):
        # 加载数据至GPU
        test_x = test_x.to(device=device, dtype=torch.float32)
        test_y = test_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            out3 = net(test_x)  # 使用网络参数，输出预测结果(训练的是噪声)

            # 计算网络去噪后的数据和干净数据的信噪比(此处是计算了所有的数据，除以了batch_size求均值)
            SNR1 = compare_SNR(test_x, test_y)  # 去噪前的信噪比
            SNR2 = compare_SNR(out3, test_y)  # 去噪后的信噪比
            Ssim1 = ssim1(test_x, test_y)
            Ssim2 = ssim1(out3, test_y)
            psnr1 = calculate_psnr(test_x, test_y)
            psnr2 = calculate_psnr(out3, test_y)
            snr_set1 += SNR1
            snr_set2 += SNR2
            psnr_set1 += psnr1
            psnr_set2 += psnr2
            ssim_set1 += Ssim1
            ssim_set2 += Ssim2
        # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的count值
    snr_set1 = snr_set1 / (batch_idx3 + 1)
    snr_set2 = snr_set2 / (batch_idx3 + 1)
    psnr_x = psnr_set1 / (batch_idx3 + 1)
    psnr_y = psnr_set2 / (batch_idx3 + 1)
    ssir_set1 = ssim_set1 / (batch_idx3 + 1)
    ssir_set2 = ssim_set2 / (batch_idx3 + 1)

    # 训练，验证，测试的loss保存至loss_sets中
    snr_set = [snr_set1, snr_set2]
    ssim_set = [ssir_set1, ssir_set2]
    psnr_set = [psnr_x, psnr_y]

    temp_sets2.append(snr_set)
    temp_sets3.append(snr_set)
    temp_sets4.append(snr_set)

    print("epoch={}  t_loss：{:.4f}，v_loss：{:.4f}".format(epoch + 1, train_loss, val_loss))
    print("   >>>    ssim1：{:.4f} dB，ssim2：{:.4f} dB".format(ssir_set1, ssir_set2))
    print("   >>>    psnr1：{:.4f}，psnr2：{:.4f}".format(psnr_x, psnr_y))
    print("   >>>    SNR1：{:.4f}，SNR2：{:.4f}".format(snr_set1, snr_set2))

    # 保存网络模型
    model_name = f'model_epoch{epoch + 1}'  # 模型命名
    torch.save(net, os.path.join('./save_dir', model_name + '.pth'))  # 保存整个神经网络的模型结构以及参数

end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 结束时间

# 保存训练花费的时间到当前文件夹下
with open('save_dir/训练时间.txt', 'w', encoding='utf-8') as f:
    f.write(start_time)
    f.write(end_time)
    f.close()
print("训练开始时间{}>>>>>>>>>>>>>>>>训练结束时间{}".format(start_time, end_time))  # 打印所用时间

