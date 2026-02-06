import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# 0) Data Loader (CHANNELS 변수에 따른 명시적 처리)
# =========================================================
def load_npz_patches_to_0_1(file_list, img_rows=512, img_cols=512, target_channels=2):
    arrays = []
    for path in file_list:
        if not os.path.exists(path): 
            print(f"Warning: 파일을 찾을 수 없음 -> {path}")
            continue
        data = np.load(path)
        key = next((k for k in ["train_x", "patches", "x", "arr_0"] if k in data), list(data.keys())[0])
        X_raw = data[key]
        
        if X_raw.ndim == 3: X_raw = X_raw[..., None]
        elif X_raw.ndim == 4 and X_raw.shape[1] < 10: 
            X_raw = np.transpose(X_raw, (0, 2, 3, 1))
        arrays.append(X_raw)
    
    if not arrays:
        raise ValueError("데이터 로드 실패. 경로를 확인하세요.")

    X = np.concatenate(arrays, axis=0).astype(np.float32)
    if X.max() > 1.0: X /= 255.0

    current_channels = X.shape[-1]
    
    if target_channels == 2 and current_channels == 1:
        # One-Hot Encoding: [Pore, Grain]
        X_pore = 1.0 - X
        X_grain = X
        X = np.concatenate([X_pore, X_grain], axis=-1)
    elif target_channels == 1 and current_channels == 2:
        # Grain 채널만 추출
        X = X[..., 1:2]
            
    return X

# =========================================================
# 1) Generator (AttributeError 해결 및 가변 활성화 함수)
# =========================================================
class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=2, img_size=512):
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * 16 * 512),
            nn.ReLU(True)
        )
        
        n_up = int(np.log2(img_size // 16))
        chs = [512, 256, 128, 64, 32, 16, 8]
        blocks = []
        in_ch = chs[0]
        for i in range(n_up):
            out_ch = chs[i + 1]
            blocks += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_ch = out_ch
            
        # [해결] nn.Sequential로 묶어 self.deconv 속성으로 등록
        self.deconv = nn.Sequential(*blocks)
        
        final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)
        if out_channels == 2:
            self.out_conv = nn.Sequential(final_conv, nn.Softmax(dim=1))
        else:
            self.out_conv = nn.Sequential(final_conv, nn.Sigmoid())

    def forward(self, z):
        x = self.fc(z).reshape(-1, 512, 16, 16)
        return self.out_conv(self.deconv(x))

# =========================================================
# 2) Critic Block
# =========================================================
class CriticBlock(nn.Module):
    def __init__(self, in_channels, img_size):
        super().__init__()
        def sn_conv(in_c, out_c): return spectral_norm(nn.Conv2d(in_c, out_c, 3, 2, 1))
        
        self.net = nn.Sequential(
            sn_conv(in_channels, 64), nn.LeakyReLU(0.2, True),
            sn_conv(64, 128), nn.LeakyReLU(0.2, True),
            sn_conv(128, 256), nn.LeakyReLU(0.2, True),
            sn_conv(256, 512), nn.LeakyReLU(0.2, True),
        )
        self.fc = spectral_norm(nn.Linear(512 * (img_size // 16)**2, 1))
        
    def forward(self, x):
        return self.fc(self.net(x).reshape(x.size(0), -1))

# =========================================================
# 3) WGAN-SN Trainer (Full Dataset Loop)
# =========================================================
class WGAN_SN_Torch:
    def __init__(self, img_rows=512, img_cols=512, channels=2, latent_dim=100, lr_g=2e-5, lr_c=2e-5, 
                 sample_dir="outcomes/snapshots", model_dir="outcomes/models", device=None):
        self.img_rows, self.latent_dim, self.channels = img_rows, latent_dim, channels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = Generator(latent_dim, channels, img_rows).to(self.device)
        self.critic1 = CriticBlock(channels, img_rows).to(self.device)
        self.critic2 = CriticBlock(channels, img_rows // 2).to(self.device)
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
        self.c_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), 
                                            lr=lr_c, betas=(0.0, 0.9))
        
        self.downsample = nn.AvgPool2d(2)
        self.sample_dir, self.model_dir = sample_dir, model_dir
        os.makedirs(sample_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)

    def train(self, X_train, epochs=100, batch_size=32, n_critic=1, save_interval=5, print_every=1):
        if X_train.shape[-1] == self.channels: 
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            
        # DataLoader 설정 (전체 데이터 루프용)
        X_t = torch.from_numpy(X_train).float()
        dataset = TensorDataset(X_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Start Training (Batches: {len(dataloader)})")

        for epoch in range(1, epochs + 1):
            c_loss_accum = 0.0
            g_loss_accum = 0.0
            
            for i, (real_batch,) in enumerate(dataloader):
                real = real_batch.to(self.device)
                
                # --- Critic Update ---
                for _ in range(n_critic):
                    z = torch.randn(real.size(0), self.latent_dim, device=self.device)
                    fake = self.generator(z).detach()
                    
                    self.c_optimizer.zero_grad()
                    loss_c = (self.critic1(fake).mean() - self.critic1(real).mean()) + \
                             (self.critic2(self.downsample(fake)).mean() - self.critic2(self.downsample(real)).mean())
                    loss_c.backward()
                    self.c_optimizer.step()
                    c_loss_accum += loss_c.item()
                
                # --- Generator Update ---
                z = torch.randn(real.size(0), self.latent_dim, device=self.device)
                fake = self.generator(z)
                self.g_optimizer.zero_grad()
                loss_g = -(self.critic1(fake).mean() + self.critic2(self.downsample(fake)).mean())
                loss_g.backward()
                self.g_optimizer.step()
                g_loss_accum += loss_g.item()

            if epoch % print_every == 0:
                avg_c = c_loss_accum / (len(dataloader) * n_critic)
                avg_g = g_loss_accum / len(dataloader)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Epoch {epoch:03d}] C_Loss: {avg_c:.4f} G_Loss: {avg_g:.4f}")
            
            if epoch % save_interval == 0: 
                self.save_samples(epoch, X_t.to(self.device))
                torch.save(self.generator.state_dict(), os.path.join(self.model_dir, f"gen_ep_{epoch:03d}.pt"))

    @torch.no_grad()
    def save_samples(self, epoch, real_tensor):
        self.generator.eval()
        z = torch.randn(4, self.latent_dim, device=self.device)
        fake = self.generator(z).cpu().numpy()
        idx = torch.randint(0, len(real_tensor), (4,))
        real = real_tensor[idx].cpu().numpy()
        
        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
        c_idx = 1 if self.channels == 2 else 0 # 2채널이면 Grain 시각화
        
        for i in range(4):
            axes[i, 0].imshow(fake[i, c_idx], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].imshow(real[i, c_idx], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_xlabel(f"Generated (Epoch {epoch})", fontsize=12)
            axes[i, 1].set_xlabel("Real (Dataset)", fontsize=12)
            for j in range(2):
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(self.sample_dir, f"comp_ep_{epoch:03d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        self.generator.train()