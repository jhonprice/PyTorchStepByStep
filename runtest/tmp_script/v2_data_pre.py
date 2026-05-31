
# 数据生成

np.random.seed(42)

true_b = 1
true_w = 2
N = 100

x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))

# Shuffles the indices
#idx = np.arange(N)
#np.random.shuffle(idx)

# Uses first 80 random indices for train
#train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
#val_idx = idx[int(N*.8):]

# Generates train and validation sets
#x_train, y_train = x[train_idx], y[train_idx]
#x_val, y_val = x[val_idx], y[val_idx]


# 数据准备

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

#x_train_tensor = torch.as_tensor(x_train).float()
#y_train_tensor = torch.as_tensor(y_train).float()

#train_data = TensorDataset(x_train_tensor,y_train_tensor)

#train_loader = DataLoader(dataset=train_data,batch_size=16,shuffle=True)

torch.manual_seed(13)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_all_tensor = torch.as_tensor(x).float()
y_all_tensor = torch.as_tensor(y).float()

dataset = TensorDataset(x_all_tensor,y_all_tensor)

## 执行拆分

ratio = .8
n_total = len(dataset)
n_train = int(ratio * n_total)
n_val = n_total - n_train

train_data,val_data = random_split(dataset,[n_train,n_val])

train_loader = DataLoader(dataset=train_data,batch_size=16,shuffle=True)

# 注意: 验证数据无须shuffle
val_loader = DataLoader(dataset=val_data,batch_size=16)
