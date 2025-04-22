import luxtorch
import random


class Network(luxtorch.Module):
    def __init__(self):
        super().__init__()
        self.layer = luxtorch.Linear(2, 1)

    def forward(self, x):
        return self.layer.forward(x)

x = luxtorch.rand((5, 2))
y = luxtorch.tensor([[5.0]])
epochs = 200
net = Network()
optimizer = luxtorch.SGD(net.parameters(), lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    out = net(x)
    loss = (out - y) * (out - y)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print loss
    if epoch % 10 == 0:
        # Print the loss every 20 epochs
        # Note: mean(0) is used to reduce the loss tensor to a scalar
        # This is a placeholder for the actual loss reduction logic
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.mean(0).item():.4f}")

test = luxtorch.rand((1, 2))
print(net(test))