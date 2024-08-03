import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print('GPU已准备好')
else:
    print('找不到GPU，请确保你的系统配置正确')

# 打印GPU设备信息
print("GPU设备数量: {}".format(torch.cuda.device_count()))
print("当前GPU设备: {}".format(torch.cuda.current_device()))
print("GPU设备名: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
print("PyTorch版本: {}".format(torch.__version__))


# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化神经网络和优化器
net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)
net.to(device)

# 准备随机训练数据
inputs = torch.randn(32, 100, device=device)
labels = torch.randint(0, 10, (32,), device=device)

# 开始训练
for epoch in range(5):  # 迭代5次
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('Epoch %d, Loss: %.3f' % (epoch+1, loss.item()))

print('训练完成')
