# DNNHyperparameterTuning


## Objective
Deep Neural Networks have a variety of hyperparameters such as **learning rate, number of layers, hidden size, dropout, regularization strength, activation functions, etc** that can strongly determine the performance of the model. The optimization model uses metric values through search process probabilistic to converge to the optimal combination of hyperparameter.

**Optuna** has three building blocks:
1. Trial: A single call of the objective function
2. Study: An optimization session, set of trials
3. Parameter: Parameters optimized

## Hyper-Parameters
`optimizer:`\
`num_layers:` \
`hidden_size:`\
`dropout:`\
`learning_rate:` The step size at each iteration while moving toward a minimum of a loss function

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
    
## Output (Optimal Hyper-Parameters)
```
Trial 7 finished with value: 0.6218652781986055 and parameters: 

- 'optimizer': <class 'torch.optim.adamw.AdamW'>
- 'num_layers': 5, 
- 'hidden_size': 113, 
- 'dropout': 0.40159780066818385, 
- 'learning_rate': 6.648176014795157e-05

 ```

## Output
```bash
       ....                   ....                  ....

```

## Model's Architecture
```
DeepNeuralNetwork(

)
```  
