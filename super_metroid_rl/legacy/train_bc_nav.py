#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os

class NavigationPolicy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NavigationPolicy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        # Input: 1 x H // 2 x W // 2? No, extraction was 128x112
        # Let's assume input is 112x128 (H,W)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            self.feature_size = self.features(dummy).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) # Logits for BCE
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.features(x)
        x = self.fc(x)
        return x

class DemoDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.obs = data['obs'] # (N, H, W)
        self.acts = data['acts'] # (N, 12)
        
        # Add channel dim to obs
        self.obs = self.obs[:, np.newaxis, :, :]
        
    def __len__(self):
        return len(self.obs)
        
    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx].astype(np.float32)

def train(demo_path, model_path, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading data from {demo_path}...")
    dataset = DemoDataset(demo_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 112x128 input
    input_shape = (dataset[0][0].shape[1], dataset[0][0].shape[2])
    print(f"Input Data Shape: {input_shape}")
    
    policy = NavigationPolicy(input_shape, 12).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (obs, acts) in enumerate(dataloader):
            obs, acts = obs.to(device), acts.to(device)
            
            optimizer.zero_grad()
            logits = policy(obs)
            loss = criterion(logits, acts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
    print(f"Saving model to {model_path}...")
    torch.save(policy.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_path', help='Path to .npz file')
    parser.add_argument('model_path', help='Path to save .pth model')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train(args.demo_path, args.model_path, args.epochs)
