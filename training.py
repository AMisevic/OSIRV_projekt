import torch
import os
import torch.optim as optim 
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

from data import train_loader
from unet import UNet
from diffusion import Diffusion

timestep = 1000 
epochs = 5
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
save_file = "./models/"

model = UNet(n_channels = 3, n_classes = 3).to(device)
diffusion = Diffusion(timestep, device) 
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc = f"Epoch {epoch + 1}/{epochs}")

    for compressed, original in progress_bar:
        compressed = compressed.to(device)
        original = original.to(device)

        t = torch.randint(0, timestep, (compressed.size(0),), device = device)

        noised, noise = diffusion.forward_diff(original, t)

        noise_pred = model(noised)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss = loss.item())

    print(f"Epoch {epoch + 1} complete. Average loss: {epoch_loss / len(train_loader)}")

model_path = os.path.join(save_file, f"model_epoch_{epoch + 1}.pth")
torch.save(model.state_dict(), model_path)

# Evaluacija modela

model.eval()
with torch.no_grad():
    for i, (compressed, _) in enumerate(train_loader):
        compressed = compressed.to(device)
        t = timestep - 1

        restored = compressed
        for step in range(timestep):
            noise_pred = model(restored)
            restored = diffusion.reverse_diff(restored, t - step, noise_pred)

        save_image(restored, f"restored_image_{i}.png")
        if i == 5: 
            break

