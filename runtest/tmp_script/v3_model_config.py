
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

optim_sgd = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

# 创建训练循环实例
train_step = make_train_step(model,loss_fn,optim_sgd)
val_step = make_val_step_fn(model,loss_fn)

# 添加writer
writer = SummaryWriter("logs/simple_linear_"+datetime.now().timestamp().__str__())


x_sample,y_sample = next(iter(train_loader))
writer.add_graph(model,x_sample.to(device))
