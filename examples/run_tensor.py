import luxtorch

a = luxtorch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
b = luxtorch.tensor([[7, 8], [9, 10], [11, 12]], requires_grad=True)
d = luxtorch.tensor([[7, 8, 9], [10, 11, 12]], requires_grad=True)
x = a @ b
print(x)
grad_output = x.zeros() + luxtorch.tensor([1.0])
x.backward(grad_output)
print(a.grad)