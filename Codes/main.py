# main.py

import torch
from torch.utils.data import DataLoader
from models_def import TL_LipLanguageModel
from train_models import train_model
from opt_param import params
from loss import get_loss
from lr_schedule import get_scheduler
from MatDataLoad import LipDataset

# ----------------- Step 0: -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------- Step 1: -----------------
source_data = LipDataset('source_user.mat')  
target_data = LipDataset('target_user.mat')  

source_loader = DataLoader(source_data, batch_size=params['batch_size'], shuffle=True)
target_loader = DataLoader(target_data, batch_size=params['batch_size'], shuffle=True)

# ----------------- Step 2:  -----------------
print("\n===== Training Source Model =====")
model_source = TL_LipLanguageModel(
    input_dim=params['input_dim'],
    hidden_dim=params['hidden_dim'],
    num_layers=params['num_layers'],
    num_classes=params['num_classes'],
    freeze_first_lstm=False
)

optimizer_source = torch.optim.Adam(model_source.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
criterion = get_loss()
scheduler_source = get_scheduler(optimizer_source, mode='step', step_size=50, gamma=0.5)

train_model(
    model=model_source,
    train_loader=source_loader,
    val_loader=target_loader,
    criterion=criterion,
    optimizer=optimizer_source,
    device=device,
    num_epochs=params['num_epochs']
)

pretrained_lstm = model_source.lstm

# ----------------- Step 3:  -----------------
print("\n===== Fine-tuning Target Model with Transfer Learning =====")
model_target = TL_LipLanguageModel(
    input_dim=params['input_dim'],
    hidden_dim=params['hidden_dim'],
    num_layers=params['num_layers'],
    num_classes=params['num_classes'],
    freeze_first_lstm=True
)

model_target.lstm.weight_ih_l0.data = pretrained_lstm.weight_ih_l0.data.clone()
model_target.lstm.weight_hh_l0.data = pretrained_lstm.weight_hh_l0.data.clone()
model_target.lstm.bias_ih_l0.data = pretrained_lstm.bias_ih_l0.data.clone()
model_target.lstm.bias_hh_l0.data = pretrained_lstm.bias_hh_l0.data.clone()

optimizer_target = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_target.parameters()),
    lr=params['lr']
)
scheduler_target = get_scheduler(optimizer_target, mode='step', step_size=30, gamma=0.5)

train_model(
    model=model_target,
    train_loader=target_loader,
    val_loader=target_loader,
    criterion=criterion,
    optimizer=optimizer_target,
    device=device,
    num_epochs=150
)
