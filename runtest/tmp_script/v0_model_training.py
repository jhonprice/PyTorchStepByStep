
n_epochs = 1000

for epoch in range(n_epochs):
    model.train()

    yhat = model(x_train_tensor)

    loss = loss_fn(yhat,y_train_tensor)

    loss.backward()

    optim_sgd.step()
    optim_sgd.zero_grad()
