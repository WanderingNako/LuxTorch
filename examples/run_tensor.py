import luxtorch
import random

class Network(luxtorch.Module):
    def __init__(self):
        super().__init__()
        self.layer = luxtorch.Linear(2, 1)

    def forward(self, x):
        return self.layer.forward(x)

x, y = luxtorch.synthetic_data([[2], [-3.4]], 4.2, 100)
epochs = 200
net = Network()
optimizer = luxtorch.SGD(net.parameters(), lr=0.1)

for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    out = net(x)
    loss = (out - y) * (out - y)
    loss = loss.mean(0)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print loss
    if epoch % 10 == 0:
        # Print the loss every 20 epochs
        # Note: mean(0) is used to reduce the loss tensor to a scalar
        # This is a placeholder for the actual loss reduction logic
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print(net.named_parameters())