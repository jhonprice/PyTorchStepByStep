
# 定义epoch循环次数(训练周期)
n_epochs = 1000

losses = []

# For each epoch...
for epoch in range(n_epochs):
    # inner loop
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        # 每次只上传一个minibatch到GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 执行一次训练循环,记录损失
        mini_batch_loss = train_step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    # 对每个minibatch取平均损失
    loss = np.mean(mini_batch_losses)
    
    losses.append(loss)
