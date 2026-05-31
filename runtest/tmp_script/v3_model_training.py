
# 保持训练次数一致
n_epochs = 200

# 使用列表存储loss的变化
losses = []

for epoch in range(n_epochs):

    loss = mini_batch(device,train_loader,train_step)

    losses.append(loss)
