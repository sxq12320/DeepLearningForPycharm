# 引入相对应的一些外部库函数
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
import gzip


def init_data(target):
    """将MNIST中的数据导入模型中,训练和检验数据的初始化
    Args:
        @ target-导入的数据库相对位置
    Returns:
        @ x_train:训练数据的具体图像
        @ t_train:训练数据的图像标签
        @ x_test:测试数据集的具体图像
        @ t_test:测试数据集的图像标签
    """
    train_images_path = os.path.join(target, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(target, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(target, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(target, "t10k-labels-idx1-ubyte.gz")

    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试访问的训练图像路径: {train_images_path}")
    print(f"训练图像路径是否存在: {os.path.exists(train_images_path)}")

    with gzip.open(train_images_path , 'rb') as f:
        x_train = np.frombuffer(f.read(),dtype=np.uint8 , offset=16)
    x_train = x_train.reshape(-1 ,784)
    x_train = (x_train / 255.0) * 2 - 1 #归一化

    with gzip.open(train_labels_path , 'rb') as f:
        t_train = np.frombuffer(f.read(),dtype=np.uint8 , offset=8)
    t_train = np.eye(10)[t_train] #转换成one-hot编码

    with gzip.open(test_images_path , 'rb') as f:
        x_test = np.frombuffer(f.read(),dtype=np.uint8 , offset=16)
    x_test = x_test.reshape(-1 , 784)
    x_test = (x_test / 255.0) * 2 - 1 #归一化

    with gzip.open(test_labels_path , 'rb') as f:
        t_test = np.frombuffer(f.read(),dtype=np.uint8 , offset=8)
    t_test = np.eye(10)[t_test] #转换成one-hot编码
    return x_train,x_test,t_train,t_test


def init_network(input , hidden1 , hidden2 , output):
    """神经网络初始化,初始化权重、偏置等
    Args:
        @ input-输入层的维度
        @ hidden1-第一个隐藏层的维度
        @ hidden2-第二个隐藏层的维度
        @ ouput-输出层的维度
    Returns:
        @ W1-输入层到第一个隐藏层的权重初始化
        @ W2-第一个隐藏层到第二个隐藏层的权重初始化
        @ W3-第二个隐藏层到输出层的权重初始化
        @ b1-输入层到第一个隐藏层的偏置初始化
        @ b2-第一个隐藏层到第二个隐藏层的偏置初始化
        @ b3-第二个隐藏层到输出层的偏置初始化
    """
    W1 = np.random.randn(input , hidden1)
    W2 = np.random.randn(hidden1 , hidden2)
    W3 = np.random.randn(hidden2 , output)
    b1 = np.zeros((1 , hidden1))
    b2 = np.zeros((1 , hidden2))
    b3 = np.zeros((1 , output))
    return W1,W2,W3,b1,b2,b3


def tanh(x):
    """定义双曲正切函数作为激活函数
    Args:
        @ x-输入自变量
    Returns:
        @ y-输出因变量
    """
    y = np.tanh(x)
    return y


def tanh_dao(x):
    """定义双曲正切函数的导数用在反向传播部分
    Args:
        @ x-输入自变量
    Returns:
        @ y-输出因变量
    """
    y =  1 - np.tanh(x) ** 2
    return y


def softmax(x):
    """softmax激活函数的定义
    Args:
        @ x-输入自变量
    Returns:
        @ y-输出因变量
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y 


def cross_loss(y_pred , y_true):
    """定义交叉熵损失函数
    Args:
        @ y_pred-训练后自己得到的数据
        @ y_pred-应当得到的期望数据
    Returns:
        @ y-交叉熵损失大小
    """
    y = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    return y


def adam_optimizer(params , grads , m , v , t , lr , beta1=0.9 , beta2=0.9 , eps=1e-8):
    """定义Adam优化器
    Args:
        @ params-需要跟新的参数列表即(W1,W2,W3,b1,b2,b3)
        @ grads-各个梯度列表(dwi,dbi)
        @ m-一阶动量矩列表
        @ v-二阶动量矩列表
        @ t-最大的迭代次数
        @ lr-基本学习率
        @ beta1-一阶据估计的指数衰减率
        @ beta2-二阶据估计的指数衰减率
        @ eps-一个无穷小
    Returns:
        @ updated_params-更新后的参数
        @ m-一阶矩
        @ v-二阶矩
    """
    updated_params = []
    for i, (param, grad) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)

        m_corrected = m[i] / (1 - beta1 ** t)
        v_corrected = v[i] / (1 - beta2 ** t)

        param = param - lr * m_corrected / (np.sqrt(v_corrected) + eps)
        updated_params.append(param)

    return updated_params, m, v


def forward(x , W1 , W2 , W3 , b1 , b2 , b3):
    """定义前向传播函数
    Args:
        @ x -输入矩阵
        @ Wi - 权重矩阵
        @ bi - 偏置矩阵
    Returns:
        @ zi - 经过变化的节点中间输出
        @ ai - 经过激活函数后的节点中间输出
        @ y_pred - 前向传播输出
    """
    z1 = np.dot(x , W1 ) + b1
    a1 = tanh(z1)

    z2 = np.dot(a1 , W2) + b2
    a2 = tanh(z2)

    z3 = np.dot(a2 , W3) + b3
    y_pred = softmax(z3)

    return z1 , z2 , z3 , a1 , a2 , y_pred


def backward(x , y , z1 , a1 , z2 , a2 , z3 , y_pred , W1 , W2 , W3):
    """反向传播，求解各个参数的梯度大小
    Args:
        @ x-输入层
        @ y-最大的输入层层数
        @ zi-未经过激活函数的函数
        @ ai-经过激活函数的未知数据
        @ y_pred-经过前向传播后的输出量
        @ Wi-各个权重的大小
    Returns:
        @ dwi-各个权重更新后的变化梯度
        @ dbi-各个偏置更新后的变化梯度
    """
    batch_size = x.shape[0]
    #反向传播梯度的计算
    dz3 = y_pred - y
    dW3 = np.dot(a2.T , dz3)/batch_size
    db3 = np.sum(dz3 , axis=0 , keepdims=True) / batch_size
    
    dz2 = np.dot(dz3 , W3.T)*tanh_dao(z2)
    dW2 = np.dot(a1.T , dz2)/batch_size
    db2 = np.sum(dz2 , axis=0 , keepdims=True) / batch_size

    dz1 = np.dot(dz2 , W2.T)*tanh_dao(z1)
    dW1 = np.dot(x.T , dz1)/batch_size
    db1 = np.sum(dz1 , axis=0 , keepdims=True) / batch_size

    return dW1, db1, dW2, db2, dW3, db3


target = r"E:\mastercode\DeepLearningForPycharm\3_DNN\data\MNIST\raw"
x_train,x_test,t_train,t_test = init_data(target)
print("训练集图像形状:", x_train.shape)  
print("训练集标签形状:", t_train.shape)  
print("测试集图像形状:", x_test.shape)    
print("测试集标签形状:", t_test.shape) 

input = 28*28
hidden1 = 300
hidden2 = 300
output = 10
W1,W2,W3,b1,b2,b3 = init_network(input , hidden1 , hidden2 , output)

m = [np.zeros_like(param) for param in [W1 , W2 , W3 , b1 , b2 , b3]]
v = [np.zeros_like(param) for param in [W1 , W2 , W3 , b1 , b2 , b3]]
t = 0

epochs = 300
batch_size = 200
lr = 0.001

loss_list = []
acc_list = []

for epoch in range(1, epochs + 1):
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i + batch_size]
        t_batch = t_train[i:i + batch_size]

        z1, z2, z3, a1, a2, y_pred = forward(x_batch, W1, W2, W3, b1, b2, b3)

        dW1, db1, dW2, db2, dW3, db3 = backward(x_batch, t_batch, z1, a1, z2, a2, z3, y_pred, W1, W2, W3)

        params = [W1, W2, W3, b1, b2, b3]
        grads  = [dW1, dW2, dW3, db1, db2, db3]
        t += 1
        params, m, v = adam_optimizer(params, grads, m, v, t, lr=lr)
        W1, W2, W3, b1, b2, b3 = params

    _, _, _, _, _, y_train_pred = forward(x_train, W1, W2, W3, b1, b2, b3)
    loss = cross_loss(y_train_pred, t_train)

    _, _, _, _, _, y_test_pred = forward(x_test, W1, W2, W3, b1, b2, b3)
    acc = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(t_test, axis=1))
    loss_list.append(loss)
    acc_list.append(acc)
    print(f"Epoch {epoch}/{epochs}   Loss={loss:.4f}   Test Acc={acc:.4f}")

_,_,_,_,_,y_test_pred=forward(x_test , W1 , W2 , W3 , b1 , b2 , b3)
y_true = np.argmax(t_test,axis=1)#数组最大值所在位置
y_pred_label = np.argmax(y_test_pred , axis=1)#数组最大值所在位置
accuacy = np.mean(y_true==y_pred_label)#求均值
print("\nFinal Test Accuracy: ",accuacy)

cm = np.zeros((10,10), dtype=int)
for i in range(len(y_true)):
    cm[y_true[i]][y_pred_label[i]] += 1

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10,4))

plt.plot(loss_list,linewidth = 3,color = 'red' , label = 'LOSS LINE')
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.plot(acc_list ,linewidth = 3 , color = 'blue' , label = 'Accuracy LINE')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.grid(True)
plt.legend()
plt.show()

cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,6))
plt.imshow(cm_norm, cmap='Blues')
plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

for i in range(10):
    for j in range(10):
        value = cm_norm[i][j]
        plt.text(j, i, f"{value:.2f}",
                 ha='center', va='center',
                 color='white' if value > 0.5 else 'black')

plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.show()