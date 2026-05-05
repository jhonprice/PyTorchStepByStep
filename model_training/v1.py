
# 定义周期数
n_epochs = 1000

# 记录每次循环的损失
losses = []

# For each epoch...
for epoch in range(n_epochs):
    # 随机梯度下降：全部训练数据，训练N遍
    loss = train_step_fn(x_train_tensor, y_train_tensor)
    losses.append(loss)
