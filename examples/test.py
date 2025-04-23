import luxtorch

a = luxtorch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
b = a.sqrt()
print(b)
b.backward()
print(a.grad)