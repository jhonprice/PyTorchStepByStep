
n_epochs = 1000

# 使用列表存储loss的变化
losses = []

for epoch in range(n_epochs):

    mini_batch_loss = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        tmp_loss = train_step(x_batch,y_batch)
        mini_batch_loss.append(tmp_loss)
    loss = np.mean(mini_batch_loss)

    losses.append(loss)
