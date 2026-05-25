
# This is redundant now, but it won't be when we introduce
# Datasets...
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sets learning rate - this is "eta" ~ the "n"-like Greek letter
# 超参数：学习率lr
lr = 0.1

torch.manual_seed(42)
# Now we can create a model and send it at once to the device
# 模型
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Defines a SGD optimizer to update the parameters 
# (now retrieved directly from the model)
# 优化器
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a MSE loss function
# 损失函数
loss_fn = nn.MSELoss(reduction='mean')
