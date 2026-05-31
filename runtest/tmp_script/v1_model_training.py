
n_epochs = 1000

# 使用列表存储loss的变化
losses = []

for epoch in range(n_epochs):
    loss = train_step(x_train_tensor,y_train_tensor)
    losses.append(loss)
