from model.cnn import SimpleCNN
import torch

x = torch.randn(32 , 3 , 224 , 224)
model = SimpleCNN(num_class=4)
output = model(x)
print(output.shape) 