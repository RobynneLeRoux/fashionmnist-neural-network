# FashionMNIST Neural Network

A multi-layer perceptron in PyTorch that classifies FashionMNIST clothing images into 10 categories, with a systematic hyperparameter study.

## Results
- **91.1% validation accuracy** with the baseline network (Adam, learning rate 0.001, dropout 0.25)
- 27-configuration grid search: 3 optimisers (SGD with momentum, Adam, RMSprop) × 3 learning rates × 3 dropout rates
- **Key finding:** dropout rate had the biggest effect. Dropout of 0.1 beat 0.25 and 0.5 across every optimiser; 0.5 over-regularised and suppressed learning. Adaptive optimisers (Adam, RMSprop) beat SGD.

## Architecture
`784 inputs → 128 (ReLU) → 64 (ReLU) → 10 outputs`

## What I built
- Custom PyTorch `Dataset` that loads FashionMNIST from CSV and normalises the images
- MLP model, training loop and evaluation in PyTorch
- Grid search, loss curves and confusion matrices to diagnose which classes the model confuses

## Run it
1. `pip install -r requirements.txt` (install PyTorch with GPU support if available: https://pytorch.org/get-started/locally/)
2. Download the CSV version of FashionMNIST (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`, e.g. from Kaggle) into `data/`.
3. Open `fashionmnist_mlp.ipynb` and run all cells.

*Built for CSC2042S Machine Learning, University of Cape Town (2025).*
