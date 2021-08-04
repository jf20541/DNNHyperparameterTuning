# DNNHyperparameterTuning


## Objective
Deep Neural Networks have a variety of hyperparameters such as **learning rate, number of layers, hidden size, dropout, regularization strength, activation functions, etc** that can strongly determine the performance of the model. The optimization model uses metric values through search process probabilistic to converge to the optimal combination of hyperparameters.

**Optuna** has three building blocks:
1. Trial: A single call of the objective function
2. Study: An optimization session, set of trials
3. Parameter: Parameters optimized

## Output (Optimal HyperParameters)
```
- 'optimizer': <class 'torch.optim.adamw.AdamW'>
- 'num_layers': 6, 
- 'hidden_size': 38, 
- 'dropout': 0.11785186946366781, 
- 'learning_rate': 0.00048109264473829103
 ```
```bash
       ....                   ....                  ....
Epoch:10/15, Train ROC-AUC: 0.9539, Eval ROC-AUC: 0.9320
Epoch:11/15, Train ROC-AUC: 0.9564, Eval ROC-AUC: 0.9394
Epoch:12/15, Train ROC-AUC: 0.9578, Eval ROC-AUC: 0.9481
Epoch:13/15, Train ROC-AUC: 0.9684, Eval ROC-AUC: 0.9539
Epoch:14/15, Train ROC-AUC: 0.9688, Eval ROC-AUC: 0.9514
Epoch:15/15, Train ROC-AUC: 0.9701, Eval ROC-AUC: 0.9625
```

## HyperParameters Importance
- `optimizer:` Updates the model in response to the output of the loss function with complex derivatives
- `num_layers:` Number of hidden layers in the model
- `hidden_size:` Number of hidden unit or cells on each hidden layer 
- `dropout:`Randomly drop hidden units (along with their connections) from the neural network during training
- `learning_rate:` The step size at each iteration while moving toward a minimum of a loss function


![](https://github.com/jf20541/DNNHyperparameterTuning/blob/main/plots/HyperparameterImportance.png?raw=true)


### Model-Based Samplers 
- **GridSearch:** Exhaustive search over specified parameter values
- **Tree-structured Parzen Estimator (TPE):** Bayesian optimization based on kernel fitting
- **Gaussian Process:** Bayesian optimization based on Gaussian Process
- **Covariance Matrix Adaptation Evolution Strategy (CMA-ES):** Meta-heuristics model for continuous space. 
- **RandomSearch:** Random parameters to fully explore all of its space equally. 


## Repository File Structure
    ├── src          
    │   ├── train.py             # Training the DNN, objective function to optimize hyper-parameters and evaluate  
    │   ├── model.py             # Neural Networks architecture, inherits nn.Module
    │   ├── engine.py            # Class Engine for Training, Evaluation, and Loss function 
    │   ├── dataset.py           # Custom Dataset that return a paris of [input, label] as tensors
    │   └── config.py            # Define path as global variable
    ├── inputs
    │   ├── train_folds.csv      # Stratified K-Fold Dataset 
    │   └── train.csv            # Cleaned Data and Featured Engineered 
    ├── models
    │   └── model.bin            # Deep Neural Networks parameters saved into model.bin 
    ├── requierments.txt         # Packages used for project
    └── README.md

## Model's Architecture
```
DeepNeuralNetwork(
  (model): Sequential(
    (0): Linear(in_features=31, out_features=38, bias=True)
    (1): Dropout(p=0.11785186946366781, inplace=False)
    (2): ReLU()
    (3): Linear(in_features=38, out_features=38, bias=True)
    (4): Dropout(p=0.11785186946366781, inplace=False)
    (5): ReLU()
    (6): Linear(in_features=38, out_features=38, bias=True)
    (7): Dropout(p=0.11785186946366781, inplace=False)
    (8): ReLU()
    (9): Linear(in_features=38, out_features=38, bias=True)
    (10): Dropout(p=0.11785186946366781, inplace=False)
    (11): ReLU()
    (12): Linear(in_features=38, out_features=38, bias=True)
    (13): Dropout(p=0.11785186946366781, inplace=False)
    (14): ReLU()
    (15): Linear(in_features=38, out_features=38, bias=True)
    (16): Dropout(p=0.11785186946366781, inplace=False)
    (17): ReLU()
    (18): Linear(in_features=38, out_features=1, bias=True)
    (19): Sigmoid()
  )
)
Ep
```  
