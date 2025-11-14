import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Audio, display

class TrainAudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, categories, max_frames=512, fraction=1.0):
        self.root_dir = root_dir
        self.categories = categories
        self.max_frames = max_frames
        self.file_list = []
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}
        for cat_name in self.categories:
            cat_dir = os.path.join(root_dir, cat_name)
            files_in_cat = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.wav')]
            num_to_sample = int(len(files_in_cat) * fraction)
            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128
        )(wav)
        log_spec = torch.log1p(mel_spec)
        _, _, n_frames = log_spec.shape
        if n_frames < self.max_frames:
            pad = self.max_frames - n_frames
            log_spec = F.pad(log_spec, (0, pad))
        else:
            log_spec = log_spec[:, :, :self.max_frames]
        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return log_spec, label_vec

class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        self.fc = nn.Linear(latent_dim + num_categories, 256 * 8 * 32)
        self.unflatten_shape = (256, 8, 32)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h).view(-1, *self.unflatten_shape)
        fake_spec = self.net(h)
        return fake_spec

class CGAN_Discriminator(nn.Module):
    def __init__(self, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape
        self.label_embedding = nn.Linear(num_categories, H * W)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=(8, 32), stride=1, padding=0)
        )

    def forward(self, spec, y):
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape)
        h = torch.cat([spec, label_map], dim=1)
        logit = self.net(h)
        return logit.view(-1, 1)

def generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050):
    generator.eval()
    y = F.one_hot(torch.tensor([category_idx] * num_samples), num_classes=generator.num_categories).float().to(device)
    z = torch.randn(num_samples, generator.latent_dim, device=device)
    with torch.no_grad():
        log_spec_gen = generator(z, y)
    spec_gen = torch.expm1(log_spec_gen).squeeze(1)
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1, n_mels=128, sample_rate=sample_rate
    ).to(device)
    linear_spec = inverse_mel(spec_gen)
    griffin = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, n_iter=32).to(device)
    waveform = griffin(linear_spec)
    return waveform.cpu()

def save_and_play(wav, sample_rate, filename):
    if wav.dim() > 2:
        wav = wav.squeeze(0)
    torchaudio.save(filename, wav, sample_rate=sample_rate)
    display(Audio(data=wav.numpy(), rate=sample_rate))

def train_gan(generator, discriminator, dataloader, device, categories, epochs, lr, latent_dim):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs('gan_generated_audio', exist_ok=True)
    os.makedirs('gan_spectrogram_plots', exist_ok=True)
    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}', leave=True)
        for real_specs, labels in loop:
            real_specs, labels = real_specs.to(device), labels.to(device)
            batch_size = real_specs.size(0)
            real_labels_tensor = torch.ones(batch_size, 1, device=device)
            fake_labels_tensor = torch.zeros(batch_size, 1, device=device)
            optimizer_D.zero_grad()
            loss_D_real = criterion(discriminator(real_specs, labels), real_labels_tensor)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_specs = generator(z, labels)
            loss_D_fake = criterion(discriminator(fake_specs.detach(), labels), fake_labels_tensor)
            (loss_D_real + loss_D_fake).backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            loss_G = criterion(discriminator(fake_specs, labels), real_labels_tensor)
            loss_G.backward()
            optimizer_G.step()
            loop.set_postfix(loss_D=(loss_D_real + loss_D_fake).item(), loss_G=loss_G.item())
        generator.eval()
        fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
        if len(categories) == 1:
            axes = [axes]
        for cat_idx, cat_name in enumerate(categories):
            y_cond = F.one_hot(torch.tensor([cat_idx]), num_classes=generator.num_categories).float().to(device)
            z_sample = torch.randn(1, generator.latent_dim).to(device)
            with torch.no_grad():
                spec_gen_log = generator(z_sample, y_cond)
            axes[cat_idx].imshow(spec_gen_log.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            axes[cat_idx].set_title(f'{cat_name} (Epoch {epoch})')
            axes[cat_idx].axis('off')
            wav = generate_audio_gan(generator, cat_idx, 1, device)
            save_and_play(wav, 22050, f'gan_generated_audio/{cat_name}_ep{epoch}.wav')
        plt.tight_layout()
        plt.savefig(f'gan_spectrogram_plots/epoch_{epoch:03d}.png')
        plt.close(fig)
        generator.train()

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LATENT_DIM = 100
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    BASE_PATH = '/content/train/train/'
    categories = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])
    train_dataset = TrainAudioSpectrogramDataset(BASE_PATH, categories)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle
