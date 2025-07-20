import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        return self.fc2(x)
    

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

transform = transforms.ToTensor()
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

index = 3

image, label = test_data[index]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True Label: {label}")

plt.axis("off")
plt.show(block=True)
input("Press Enter to continue...")

image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)

    _, predicted = torch.max(output, 1)

print(f"Predicted Label: {predicted.item()}")