
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置学习率
lr = 0.1

torch.manual_seed(42)
# 创建模型类并上传GPU
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# 定义更新器
optimizer = optim.SGD(model.parameters(), lr=lr)

# 定义损失函数
loss_fn = nn.MSELoss(reduction='mean')

# 创建每次训练循环的函数
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
