# Lip Language Interface

A PyTorch-based lip language decoding framework using wearable sensors and transfer learning for fast adaptation across users.

## System Requirements
- Python >= 3.7  
- MATLAB >= R2021b (for `.mat` dataset preparation)

## Project Structure
- `main.py` – Entry point: source & TL training  
- `models_def.py` – LSTM model w/ freeze support  
- `train_models.py` – Training loop  
- `loss.py` – Loss function definition  
- `lr_schedule.py` – Scheduler utility  
- `opt_param.py` – Hyperparameters  
- `MatDataLoad.py` – `.mat` dataset loader  

## Features
- LSTM-based lip decoding  
- Transfer learning from source to new users  
- Efficient adaptation across users  
