true_b=1
true_w=2
N=100

np.random.seed(42) #test

x=np.random.randn(N,1)

epsilon= (.1*np.random.randn(N,1))

y=true_b+true_w*x

print(x,y)