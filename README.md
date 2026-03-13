# Quadruped Training

You can check out a detailed write-up about building the project here:

[Training Locomotion Policies for a Quadruped](https://medium.com/@kpbudihal/training-locomotion-policies-for-a-quadruped-9e63e36070f0?postPublishedType=initial)

## Setup

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

Run training with the default settings:

```bash
python train_model.py
```

Example with custom arguments:

```bash
python train_model.py --training-time 5000000 --num-envs 4 --model-name PPO_1.zip
```

Trained models are saved to `saved_models/final_models/`.

## Run A Trained Model

Run inference with the default model:

```bash
python run_model.py
```

Run a specific saved model:

```bash
python run_model.py --model-name PPO_1.zip
```
