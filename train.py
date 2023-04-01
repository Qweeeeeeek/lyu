from torch import optim
import time
from utils import *
from torch.utils.data import Dataset
from dataset import *
import matplotlib.pyplot as plt
from ssim1 import ssim1
from unet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = unet()
net.to(device=device)

epochs = 50
lr = 0.001
batch_size = 64

optimizer = optim.Adam(net.parameters(), lr)
criterion = nn.MSELoss()

train_path_x = "D:\\Seismic\\A1\\feature\\"
train_path_y = "D:\\Seismic\\A1\\label\\"

dataset = MyDataset(train_path_x, train_path_y)
print('Dataset size:', len(dataset))
valida_size = int(len(dataset) * 0.1)
train_size = len(dataset) - valida_size * 2
train_dataset, test_dataset, valida_dataset = torch.utils.data.random_split(dataset,
                                                                            [train_size, valida_size, valida_size])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

temp_sets1 = []
temp_sets2 = []
temp_sets3 = []
temp_sets4 = []

start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())
for epoch in range(epochs):
    train_loss = 0.0
    net.train()  # 开启训练模式
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0):
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        out1 = net(batch_x)
        loss1 = criterion(out1, batch_y)
        train_loss += loss1.item()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
    train_loss = 10*train_loss / (batch_idx1 + 1)

    net.eval()
    val_loss = 0.0
    for batch_idx2, (val_x, val_y) in enumerate(valida_loader, 0):
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out2 = net(val_x)
            # 计算loss
            loss2 = criterion(out2, val_y)
            val_loss += loss2.item()
    val_loss = 10*val_loss / (batch_idx2 + 1)  # 本次epoch的平均验证loss
    loss_set = [train_loss, val_loss]
    temp_sets1.append(loss_set)

    snr_set1 = 0.0
    snr_set2 = 0.0
    psnr_set1 = 0.0
    psnr_set2 = 0.0
    ssim_set1 = 0.0
    ssim_set2 = 0.0
    for batch_idx3, (test_x, test_y) in enumerate(test_loader, 0):
        test_x = test_x.to(device=device, dtype=torch.float32)
        test_y = test_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out3 = net(test_x)

            SNR1 = compare_SNR(test_x, test_y)
            SNR2 = compare_SNR(out3, test_y)
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

    snr_set1 = snr_set1 / (batch_idx3 + 1)
    snr_set2 = snr_set2 / (batch_idx3 + 1)
    psnr_x = psnr_set1 / (batch_idx3 + 1)
    psnr_y = psnr_set2 / (batch_idx3 + 1)
    ssir_set1 = ssim_set1 / (batch_idx3 + 1)
    ssir_set2 = ssim_set2 / (batch_idx3 + 1)

    snr_set = [snr_set1, snr_set2]
    ssim_set = [ssir_set1, ssir_set2]
    psnr_set = [psnr_x, psnr_y]

    temp_sets2.append(snr_set)
    temp_sets3.append(ssim_set)
    temp_sets4.append(psnr_set)

    print("epoch={}  t_loss：{:.4f}，v_loss：{:.4f}, ssim：{:.4f} dB, psnr：{:.4f}, SNR：{:.4f}".format(epoch + 1, train_loss, val_loss, ssir_set2, psnr_y, snr_set2))

    model_name = f'model_epoch{epoch + 1}'
    torch.save(net, os.path.join('./save_dir', model_name + '.pth'))
print(">>>ssim1：{:.4f} dB--psnr1：{:.4f}--SNR1：{:.4f}".format(ssir_set1, psnr_x, snr_set1))
end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 结束时间

# 保存训练花费的时间到当前文件夹下
with open('save_dir/训练时间.txt', 'w', encoding='utf-8') as f:
    f.write(start_time)
    f.write(end_time)
    f.close()
# print("训练开始时间{}>>>>>>>>>>>>>>>>训练结束时间{}".format(start_time, end_time))  # 打印所用时间

loss_sets = []
for sets in temp_sets1:
    for i in range(2):
        loss_sets.append(sets[i])
loss_sets = np.array(loss_sets).reshape(-1, 2)  # 重塑形状10*2，-1表示自动推导
np.savetxt('save_dir/loss_sets.txt', loss_sets, fmt='%.4f')

snr_sets = []
for sets in temp_sets2:
    for i in range(2):
        snr_sets.append(sets[i])
snr_sets = np.array(snr_sets).reshape(-1, 2)
np.savetxt('save_dir/snr_sets.txt', snr_sets, fmt='%.4f')

