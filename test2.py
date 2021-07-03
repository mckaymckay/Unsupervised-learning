import torch
x=torch.arange(15).view(5,3)
x=x.float()
x_mean=torch.mean(x)
print(x)
print(x_mean)
print(torch.pow(x,2))
print(torch.mul(x,x))