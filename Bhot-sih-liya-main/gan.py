import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, frame1, frame2):
        x = torch.cat((frame1, frame2), 1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)  # Ensure output is [batch_size, 1]

    

# Define the dataset class
class FrameInterpolationDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.transform = transform

        # Debug: Print the number of images and the first few image files
        print(f"Total images found: {len(self.image_files)}")
        if len(self.image_files) > 0:
            print(f"First few images: {self.image_files[:5]}")

    def __len__(self):
        return max(0, len(self.image_files) - 2)

    def __getitem__(self, idx):
        frame1_path = os.path.join(self.image_folder, self.image_files[idx])
        frame2_path = os.path.join(self.image_folder, self.image_files[idx + 2])  # Interpolated in between
        interpolated_path = os.path.join(self.image_folder, self.image_files[idx + 1])

        frame1 = Image.open(frame1_path).convert("RGB")
        frame2 = Image.open(frame2_path).convert("RGB")
        interpolated = Image.open(interpolated_path).convert("RGB")

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            interpolated = self.transform(interpolated)

        return frame1, frame2, interpolated

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize models, loss function, and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, start_epoch, epochs):
    adversarial_loss = nn.BCELoss()

    for epoch in range(start_epoch, epochs):
        for i, (frame1, frame2, interpolated) in enumerate(dataloader):
            # Move data to GPU if available
            frame1, frame2, interpolated = frame1.to(device), frame2.to(device), interpolated.to(device)

            # Adversarial ground truths
            valid = torch.ones(frame1.size(0), 1, requires_grad=False).to(device)
            fake = torch.zeros(frame1.size(0), 1, requires_grad=False).to(device)

            # ---------------------
            #  Train Generator
            # ---------------------

            optimizer_G.zero_grad()

            # Generate interpolated frame using the generator
            generated_interpolated = generator(frame1, frame2)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(torch.cat((frame1, generated_interpolated, frame2), 1)), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            real_loss = adversarial_loss(discriminator(torch.cat((frame1, interpolated, frame2), 1)), valid)

            # Fake loss (on generated images)
            fake_loss = adversarial_loss(discriminator(torch.cat((frame1, generated_interpolated.detach(), frame2), 1)), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save model checkpoints after each epoch
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")


# Initialize dataset and dataloader
image_folder = "images/images"
dataset = FrameInterpolationDataset(image_folder, transform=transform)

# Debug: Print dataset length
print(f"Dataset length: {len(dataset)}")

# Ensure dataset length is positive
if len(dataset) > 0:
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
else:
    raise ValueError("Dataset length is zero. Ensure there are enough images in the folder.")

# Start training
start_epoch = 0
epochs = 200
train(generator, discriminator, dataloader, optimizer_G, optimizer_D, start_epoch, epochs)
