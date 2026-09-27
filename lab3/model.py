import torch #type: ignore
import torch.nn as nn #type: ignore

X = torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.05
)

for epoch in range(5000):

    preds = model(X)

    loss = criterion(preds, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} "
            f"Loss: {loss.item():.6f}"
        )

print("\nPredictions:\n")

with torch.no_grad():

    preds = model(X)

    for x, p, target in zip(X, preds, y):
        print(
            f"Input={x.tolist()} "
            f"Pred={p.item():.4f} "
            f"Target={target.item()}"
        )

def activation_test(activation, name):

    model = nn.Sequential(
        nn.Linear(2, 2),
        activation,
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05
    )

    grad_norm = None

    for epoch in range(5000):

        preds = model(X)

        loss = criterion(preds, y)

        optimizer.zero_grad()

        loss.backward()

        if epoch == 0:
            grad_norm = model[0].weight.grad.norm().item()

        optimizer.step()

    with torch.no_grad():

        preds = model(X)

        correct = (
            ((preds > 0.5).float() == y)
            .all()
            .item()
        )

    print(
        f"{name}: "
        f"Loss={loss.item():.6f}, "
        f"Correct={correct}, "
        f"GradNorm={grad_norm:.6f}"
    )


print("\nACTIVATION COMPARISON\n")

activation_test(nn.Sigmoid(), "Sigmoid")
activation_test(nn.Tanh(), "Tanh")
activation_test(nn.ReLU(), "ReLU")

print("\nSYMMETRY EXPERIMENT\n")

sym_model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

for param in sym_model.parameters():
    nn.init.constant_(param, 0.5)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    sym_model.parameters(),
    lr=0.05
)

for epoch in range(3000):

    preds = sym_model(X)

    loss = criterion(preds, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

print("Hidden layer weights:")
print(sym_model[0].weight)

print("\nHidden layer biases:")
print(sym_model[0].bias)

print("\nTHREE CLASS EXTENSION\n")

X_multi = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

y_multi = torch.tensor([
    0,
    1,
    1,
    2
])

multi_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 3)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    multi_model.parameters(),
    lr=0.05
)

for epoch in range(5000):

    logits = multi_model(X_multi)

    loss = criterion(
        logits,
        y_multi
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

with torch.no_grad():

    logits = multi_model(X_multi)

    probs = torch.softmax(
        logits,
        dim=1
    )

    print("\nProbabilities:\n")
    print(probs)

    print(
        "\nSum of first probability vector:"
    )

    print(
        probs[0].sum().item()
    )