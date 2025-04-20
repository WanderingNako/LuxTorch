import luxtorch
import random


class Network(luxtorch.Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(2, 1)

    def forward(self, x):
        return self.layer.forward(x)[0]


class Linear(luxtorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", luxtorch.Scalar(0)
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", luxtorch.Scalar(0)
                )
            )

    def forward(self, inputs):
        # Batch size is 1
        # inputs.shape = [1, 2]
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y

x = [luxtorch.Scalar(1.0, name="x_1"), luxtorch.Scalar(2.0, name="x_2")]
y = luxtorch.Scalar(5.0, name="y")
epochs = 200
net = Network()
optimizer = luxtorch.SGD(net.parameters(), lr=0.1)

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
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data:.4f}")

print(net(x))