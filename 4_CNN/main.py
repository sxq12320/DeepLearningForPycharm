import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets , transforms
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义注意力机制模块 (SE Block)
# ==========================================
class SE_Block(nn.Module):
    def __init__(self , channel , reduction = 16):
        super(SE_Block , self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel , out_features=channel // reduction , bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // reduction , out_features=channel , bias=False),
            nn.Sigmoid()
        )

    def forward(self , x):
        b , c , _ , _ = x.size()
        y = self.avg_pool(x).view(b , c)
        y = self.fc(y).view(b , c , 1 , 1)
        return x * y.expand_as(x)

# ==========================================
# 2. 定义 ResNet 基本模块 (BasicBlock)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self , in_channel , out_channel , stride = 1 , downsample = None):
        super(BasicBlock , self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel , out_channels=out_channel , kernel_size=3 , stride=stride , padding=1 , bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=out_channel , out_channels=out_channel , kernel_size=3 , stride=1 , padding=1 , bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        
        self.se_block = SE_Block(channel=out_channel)
        self.downsample = downsample

    def forward(self , x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se_block(out)

        out += identity
        out = self.relu(out)

        return out

# ==========================================
# 3. 定义主网络 ResNet-18 (带注意力)
# ==========================================
class ResNet_Attention(nn.Module):
    def __init__(self , block , layers , num_classes = 1000):
        super(ResNet_Attention , self).__init__()
        self.in_channel = 64
        
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=7 , stride=2 , padding=3 , bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)

        self.layer1 = self._make_layer(block , 64 , layers[0])
        self.layer2 = self._make_layer(block , 128 , layers[1] , stride=2)
        self.layer3 = self._make_layer(block , 256 , layers[2] , stride=2)
        self.layer4 = self._make_layer(block , 512 , layers[3] , stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1 , 1))
        self.fc = nn.Linear(in_features=512 * block.expansion , out_features=num_classes)

    def _make_layer(self , block , channel , blocks , stride = 1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel , out_channels=channel * block.expansion , kernel_size=1 , stride=stride , bias=False),
                nn.BatchNorm2d(num_features=channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel , channel , stride , downsample))
        self.in_channel = channel * block.expansion
        for _ in range(1 , blocks):
            layers.append(block(self.in_channel , channel))

        return nn.Sequential(*layers)

    def forward(self , x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out , 1)
        out = self.fc(out)

        return out

# ==========================================
# 4. 训练配置与流程
# ==========================================
def main():
    # 路径配置
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(current_path, 'data', 'RSCD', 'train')
    val_dir = os.path.join(current_path, 'data', 'RSCD', 'test')

    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((224 , 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406] , [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(train_dir):
        print(f"错误：找不到路径 {train_dir}")
        return

    # 数据集加载
    full_train_dataset = datasets.ImageFolder(train_dir , transform=transform)
    val_data = datasets.ImageFolder(val_dir , transform=transform)
    num_classes = len(full_train_dataset.classes)
    
    # 10% 随机采样 (为了快速训练)
    total_len = len(full_train_dataset)
    train_len = int(total_len * 0.1) 
    discard_len = total_len - train_len
    train_subset, _ = random_split(full_train_dataset, [train_len, discard_len])
    
    print(f"--------------------------------------------------")
    print(f"⚠️  极速模式：使用 {train_len} 张图片训练")
    print(f"--------------------------------------------------")

    train_loader = DataLoader(train_subset , batch_size=BATCH_SIZE , shuffle=True , num_workers=2)
    val_loader = DataLoader(val_data , batch_size=BATCH_SIZE , shuffle=False , num_workers=2)

    model = ResNet_Attention(BasicBlock , [2 , 2 , 2 , 2] , num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters() , lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # [新增] 4个列表用于记录数据
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{EPOCHS}]')
        
        for inputs , labels in loop:
            inputs , labels = inputs.to(device) , labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs , labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            # [新增] 计算训练集准确率
            _ , predicted = torch.max(outputs.data , 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # --- 测试/验证阶段 ---
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs , labels in val_loader:
                inputs , labels = inputs.to(device) , labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # [新增] 计算测试集 Loss
                
                running_test_loss += loss.item()
                
                _ , predicted = torch.max(outputs.data , 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        epoch_test_loss = running_test_loss / len(val_loader)
        epoch_test_acc = 100 * correct_test / total_test
        
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Test  Loss: {epoch_test_loss:.4f} | Test  Acc: {epoch_test_acc:.2f}%")

    print(f"训练耗时: {(time.time() - start_time)/60:.1f} 分钟")
    torch.save(model.state_dict() , 'rscd_resnet_attn_lite.pth')

    # ==========================================
    # [升级] 绘制双曲线对比图
    # ==========================================
    plt.figure(figsize=(12, 5))

    # 图1：Loss 对比
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', color='red')
    plt.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss', color='blue', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve (Train vs Test)')
    plt.legend()
    plt.grid(True)

    # 图2：Accuracy 对比
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accs, label='Train Acc', color='green')
    plt.plot(range(1, EPOCHS + 1), test_accs, label='Test Acc', color='orange', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve (Train vs Test)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_result_full.png')
    print("图像已保存为: training_result_full.png")

if __name__ == '__main__':
    main()