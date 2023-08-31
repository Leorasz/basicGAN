import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
import torch.optim as optim

mndata = fetch_openml('mnist_784', version=1)
images, labels = mndata["data"].values.tolist(), mndata["target"].values.tolist()

numberInQuestion = 1
epochs = 1

toRemove = []

for index, i in enumerate(labels):
    if i != str(numberInQuestion):
        toRemove.append(index)
for i in toRemove[::-1]:
    images.pop(i)
images = converted_list = [[1 if x > 100 else 0 for x in inner_list] for inner_list in images]
def printNumber(number):
    stri = ""
    for index, i in enumerate(number):
        stri += str(round(i))
        if len(stri) == 28:
            print(stri)
            stri = ""

discriminator = nn.Sequential(
    nn.Linear(784, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

generator = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Sigmoid()
)

epochs = 10  # Number of epochs
batch_size = 1  # Batch size

# Loss and optimizers
criterion = nn.BCELoss()  # Using BCELoss because the output layer of discriminator has a Sigmoid
optimizer_g = optim.Adam(generator.parameters(), lr=3e-4)
optimizer_d = optim.Adam(discriminator.parameters(), lr=3e-4)

print("Data cleaning finished, training")
# Training loop
if len(images) % batch_size != 0:
    images = images[:-(len(images)%batch_size)]
images = torch.tensor(images, dtype=torch.float32).view(-1, 784)
for epoch in range(epochs):
    for i, real_images in enumerate(images):  # Assuming 'images' is your data loader or list

        # Discriminator update
        optimizer_d.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size, 1)
        output_real = discriminator(real_images.view(batch_size, -1))
        loss_real = criterion(output_real, real_labels)

        # Fake images
        noise = torch.rand(batch_size, 5)  # 5-dimensional noise
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)

        # Combine losses and update discriminator
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Generator update
        optimizer_g.zero_grad()
        
        # Generate fake images
        noise = torch.rand(batch_size, 5)  # 5-dimensional noise
        fake_images = generator(noise)
        
        # We want discriminator to mistake fake images as real
        output = discriminator(fake_images)
        loss_g = criterion(output, real_labels)  # target is real labels
        
        # Update generator
        loss_g.backward()
        optimizer_g.step()
    print(f"Epoch [{epoch+1}/{epochs}]")

printNumber(generator(torch.rand(batch_size, 5))[0].tolist())