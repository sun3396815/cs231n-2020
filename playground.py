import numpy as np

a = np.array(range(10)).reshape(-1, 5)
b = np.array(range(30)).reshape(-1, 5)
for aa in a:
    for bb in b:
        cc = np.sqrt(np.sum(aa * bb))
print(123)

d = np.array(range(100))
index = np.argsort(d)
print(index)

e = np.array(range(1200)).reshape([6, 200])
ee = np.array_split(e, 3)
eee = np.stack(ee)
eeee = np.squeeze(ee, axis=0)
print(123)

f = np.array(range(12)).reshape([3,4])
print(f)
print("--------------------------")
mean0 = np.mean(f, axis=0)
print(mean0)
print("--------------------------")
mean1 = np.mean(f, axis=1)
print(mean1)
print("--------------------------")
