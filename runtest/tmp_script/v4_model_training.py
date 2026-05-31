
# 保持训练次数一致
n_epochs = 200

# 使用列表存储loss的变化
losses = []
val_losses = []

for epoch in range(n_epochs):
    # 训练内循环
    loss = mini_batch(device,train_loader,train_step)
    losses.append(loss)
    # 验证部分
    # 注意: 验证不需要梯度更新
    with torch.no_grad():
        val_loss = mini_batch(device,val_loader,val_step)
        val_losses.append(val_loss)
