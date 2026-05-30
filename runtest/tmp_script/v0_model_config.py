
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.1

model = nn.Sequential(nn.Linear(1,1)).to(device)

optim_sgd = optim.SGD(model.parameters(),lr=lr)

loss_fn = nn.MSELoss(reduction='mean')
