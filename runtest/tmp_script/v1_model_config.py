
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

optim_sgd = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

# 创建训练循环实例
train_step = make_train_step(model,loss_fn,optim_sgd)
