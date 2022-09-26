import torch

model = torch.nn.Linear(1, 1)
optim = torch.optim.SGD(model.parameters(), lr=1.0)


# Variant1
optim.zero_grad()
data1 = torch.rand(1, 1)
data2 = torch.rand(1, 1)

result1 = model(data1)
result2 = model(data2)

loss = result1.sum() + result2.sum()
loss.backward()
print(f'model.weight.grad={model.weight.grad}')


# Variant2
optim.zero_grad()

data12 = torch.concat([data1, data2])
result12 = model(data12)
loss = result12.sum()
loss.backward()
print(f'model.weight.grad={model.weight.grad}')
