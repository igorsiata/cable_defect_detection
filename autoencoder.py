import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import evaluate_model
import cv2
import matplotlib.pyplot as plt
from trainer import Trainer


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        in_chan = 3
        hidden_1 = 32
        hidden_2 = 2 * hidden_1
        latent_dim = hidden_2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, hidden_1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_1),
            nn.ReLU(),
            nn.Conv2d(hidden_1, hidden_1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_1),
            nn.ReLU(),
            nn.Conv2d(hidden_1, hidden_2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_2),
            nn.ReLU(),
            nn.Conv2d(hidden_2, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                hidden_2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_2,
                hidden_2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_2, hidden_1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(hidden_1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_1, in_chan, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_data():
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    data_dir = "./dataset/train/"
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
        ]
    )
    image_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    return train_loader


def run_evaluation(model):
    threshold = 0.12
    model.eval()
    model_device = next(model.parameters()).device

    def predict(image_1024):
        img_224 = cv2.resize(image_1024, (224, 224), interpolation=cv2.INTER_AREA)
        img_224 = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_224).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(model_device)
        with torch.no_grad():
            reconstruction = model(img_tensor)
            # plt.figure(figsize=(6, 6))
            # plt.imshow(reconstruction.squeeze(0).permute(1, 2, 0).numpy())
            # plt.title("Reconstruction")
            # plt.axis("off")
            # plt.show()
            diff = torch.abs(img_tensor - reconstruction)
            diff_gray = diff.mean(dim=1)
            diff_np = diff_gray.squeeze().cpu().numpy()

            diff_blurred = cv2.GaussianBlur(diff_np, (5, 5), 0)

            # 2. Progowanie (Thresholding) na rozmytym błędzie
            mask_224 = (diff_blurred > threshold).astype(np.uint8) * 255

            # 3. Operacja Otwarcia (Morphological Opening)
            # Usuwa z maski małe, pojedyncze kropki i cienkie linie, które mogły przetrwać.
            kernel = np.ones((3, 3), np.uint8)
            mask_224 = cv2.morphologyEx(mask_224, cv2.MORPH_OPEN, kernel)
            # mask_224 = (diff_np > threshold).astype(np.uint8) * 255

        mask_1024 = cv2.resize(mask_224, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        return mask_1024

    evaluate_model.run_eval(predict, show_examples=True)


def visualize_input_output(model, device):
    IMAGE_SIZE = (224, 224)
    data_dir = "./dataset/test/"

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    image_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        image_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    model.eval()
    inputs, _ = next(iter(dataloader))
    inputs = inputs[:8].to(device)
    with torch.no_grad():
        outputs = model(inputs)

    inputs = inputs.cpu()
    outputs = outputs.cpu()

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        inp_top = inputs[i].permute(1, 2, 0).numpy()
        out_top = outputs[i].permute(1, 2, 0).numpy()
        inp_bottom = inputs[i + 4].permute(1, 2, 0).numpy()
        out_bottom = outputs[i + 4].permute(1, 2, 0).numpy()

        # Pierwsza czwórka (zdjęcia 1-4)
        axes[0, i].imshow(inp_top)
        axes[0, i].set_title(f"Oryginał {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(out_top)
        axes[1, i].set_title(f"Rekonstrukcja {i+1}")
        axes[1, i].axis("off")

        # Druga czwórka (zdjęcia 5-8)
        axes[2, i].imshow(inp_bottom)
        axes[2, i].set_title(f"Oryginał {i+5}")
        axes[2, i].axis("off")

        axes[3, i].imshow(out_bottom)
        axes[3, i].set_title(f"Rekonstrukcja {i+5}")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = load_data()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        run_name="autoencoder_32",
    )
    trainer.train()
    # model.load_state_dict(
    #     torch.load("./best_model_autonecoder_1.pth", map_location=device)
    # )
    visualize_input_output(model, device)

    # train_model(model, device)

    #
    # model = model.to(device)
    # visualize_input_output(model, device)
    # run_evaluation(model)
