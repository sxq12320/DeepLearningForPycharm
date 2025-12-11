import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
# from model.cnn import SimpleCNN
from model.resnet import ResNet18

# ===================== 1. 硬件加速配置（保留提速，移除内存消耗） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")

# GPU加速优化（保留，不占内存）
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# ===================== 2. 数据变换优化（轻量化） =====================
# 简化变换，减少内存计算
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 直接Resize，减少CenterCrop的内存开销
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ===================== 3. 数据集加载（移除缓存，解决内存溢出） =====================
RSCD_ROOT = r"E:\mastercode\DeepLearningForPycharm\4_CNN\data\RSCD"

# 恢复原生ImageFolder，移除内存缓存
trainset = datasets.ImageFolder(
    root=os.path.join(RSCD_ROOT, "train"),
    transform=train_transform
)
testset = datasets.ImageFolder(
    root=os.path.join(RSCD_ROOT, "test"),
    transform=test_transform
)

num_classes = len(trainset.classes)
print(f"自动识别到训练集类别数：{num_classes}")
print(f"类别列表：{trainset.classes[:5]}...")

# ===================== 4. DataLoader优化（低内存版） =====================
# 降低batch_size，减少单次内存占用；关闭pin_memory（CPU内存不足时）
BATCH_SIZE = 2  # 从4→2，进一步降低内存消耗
GRAD_ACCUM_STEPS = 16  # 累积8步→等效batch_size=16，保速度

train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    num_workers=0,  # 强制0，避免多进程占用内存
    shuffle=True,
    pin_memory=False,  # 关闭内存固定，节省CPU内存
    drop_last=True,
    persistent_workers=False
)

test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE * 2,  # 测试集batch_size=4，平衡速度和内存
    num_workers=0,
    shuffle=False,
    pin_memory=False
)

# ===================== 5. 训练函数（内存优化版） =====================
def train(model, train_loader, criterion, optimizer, epochs, save_path):
    best_acc = 0.0
    model.train()
    train_dataset_len = len(train_loader.dataset)
    
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零，节省内存
        # 优化tqdm：减少进度条更新频率，降低CPU开销
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", ncols=80, mininterval=1.0)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            # 数据传输：CPU→设备，避免中间内存占用
            inputs = inputs.to(DEVICE, non_blocking=False)  # 关闭异步传输，节省内存
            labels = labels.to(DEVICE, non_blocking=False)
            
            # 混合精度前向传播（保留提速，不占额外内存）
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
            
            # 反向传播：分步释放内存
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            running_loss += loss.item() * GRAD_ACCUM_STEPS * inputs.size(0)
            
            # 梯度累积更新，每8步更新一次
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # 每更新一次梯度，清理无用变量，释放内存
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 每20个批次更新一次进度条，减少IO
            if batch_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}"})
        
        # 计算epoch损失，释放中间变量
        epoch_loss = running_loss / train_dataset_len
        print(f"\nEpoch {epoch+1} | Train Loss: {epoch_loss:.4f}")
        del running_loss
        
        # 验证：先清理内存再验证
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accuracy = evaluate(model, test_loader, criterion)
        
        # 保存模型：仅保存最优，减少磁盘IO
        if accuracy > best_acc:
            best_acc = accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存 | 最佳准确率: {best_acc:.2f}%")
        
        pbar.close()
        # 每轮epoch结束，清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ===================== 6. 验证函数（内存优化） =====================
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 完全禁用梯度，节省内存
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE, non_blocking=False)
            labels = labels.to(DEVICE, non_blocking=False)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每批次后释放内存
            del outputs, loss, predicted
    
    avg_loss = test_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    model.train()
    # 验证结束清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accuracy

# ===================== 7. 主函数（内存优化） =====================
if __name__ == "__main__":
    epochs = 20
    learning_rate = 1e-5
    save_path = "model_pth/best_27class_low_mem.pth"
    
    # 模型初始化：先在CPU初始化，再移到设备，减少内存峰值
    
    # model = SimpleCNN(num_class=num_classes)
    
    # model = model.to(DEVICE)

    model = ResNet18(num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(reduction="mean")
    # 优化器：减少参数缓存，节省内存
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False  # 关闭foreach，节省内存（PyTorch 2.0+）
    )
    
    # 开始训练
    train(model, train_loader, criterion, optimizer, epochs, save_path)
    # 最终验证
    evaluate(model, test_loader, criterion)