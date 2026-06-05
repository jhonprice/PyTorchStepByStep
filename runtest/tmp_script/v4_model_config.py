
torch.manual_seed(42)

lr = 0.1

model = nn.Sequential(nn.Linear(1, 1))

optim_sgd = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')
