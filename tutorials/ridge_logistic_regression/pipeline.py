"""
Evaluate module.
"""

from copy import deepcopy

from typing import Tuple, List, Union
from collections import OrderedDict

import torch
import wandb
import numpy as np

from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
from sklearn.datasets import load_breast_cancer

from models import LogisticRegression, NormConstrainedLogisticRegression

import cooper


def load_breast_cancer_data() -> TensorDataset:

    """Load the breast cancer dataset and return it as a torch tensor.

    Returns:
        TensorDataset: Breast cancer dataset."""

    breast_cancer = load_breast_cancer()

    X = torch.tensor(breast_cancer["data"], dtype=torch.float32)
    y = torch.tensor(breast_cancer["target"], dtype=torch.float32)

    return TensorDataset(X, y)


def split_data(dataset: torch.Tensor, ratio: float) -> Tuple[Subset, Subset, Subset]:

    """Split the data into training, validation, and test data.

    Args:
        data (torch.Tensor): The data to split.
        ratio (float): The ratio of the data to use for training, validation, and test.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The training, validation, and test data."""

    n_samples = len(dataset)
    train_samples = int(np.ceil(n_samples * ratio))
    val_samples = int(np.ceil((n_samples - train_samples) * 0.5))
    test_samples = n_samples - train_samples - val_samples

    return random_split(dataset, [train_samples, val_samples, test_samples])


def preparation(
    config: wandb.config,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.nn.Module, int]:
    """Prepares data and instances the model for training.
    Args:
        train_ratio (float): The ratio of the data to be used for training.
        batch_size (int): The batch size to use for training.
        n_iters (int): The number of iterations to train for.
    Returns:
        tuple: A tuple of the train loader, the validation loader, the test loader, the model, and
            the number of epochs to train it.
    """

    train_ratio, batch_size, n_iters = (
        config.train_ratio,
        config.batch_size,
        config.n_iters,
    )

    dataset = load_breast_cancer_data()

    train_data, val_data, test_data = split_data(dataset, train_ratio)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = LogisticRegression(dataset[0][0].shape[0], 1)

    num_epochs = int(n_iters / (len(train_data) / batch_size))

    return train_loader, val_loader, test_loader, model, num_epochs


def set_optimizers(
    cmp: cooper.CMPState,
    params: List[torch.nn.Parameter],
    primal_lr: float,
    dual_lr: float,
) -> Tuple[cooper.LagrangianFormulation, cooper.ConstrainedOptimizer]:
    """Set the optimizers for the model.
    Args:
        cmp (cooper.CMPState): The CMP state.
        params (t.List[torch.nn.Parameter]): The parameters of the model.
        primal_lr (float): The learning rate for the primal optimizer.
        dual_lr (float): The learning rate for the dual optimizer.
    Returns:
        t.Tuple[cooper.LagrangianFormulation, cooper.ConstrainedOptimizer]: The lagrangian
            formulation and the constrained optimizer.
    """
    formulation = cooper.LagrangianFormulation(cmp)

    primal_optimizer = torch.optim.SGD(params, lr=primal_lr, momentum=0.7)
    dual_optimizer = cooper.optim.partial_optimizer(
        torch.optim.SGD, lr=dual_lr, momentum=0.7
    )

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )
    return formulation, coop


def train_log(cmp, dual, iter_num) -> None:
    """Logs the training information.
    Args:
        cmp (cooper.CMPState): The CMP state.
        dual (cooper.DualState): The dual state.
        iter_num (int): The current iteration.
    """
    loss = cmp.loss.item()
    ineq_defect = cmp.ineq_defect.data.numpy()
    mutipliers = dual[0].data.numpy()
    wandb.log({"iter": iter_num, "loss": loss})
    wandb.log({"iter": iter_num, "multipliers": mutipliers})
    wandb.log({"iter": iter_num, "ineq_defects": ineq_defect})

def train_model(
    model: torch.nn.Module,
    is_constrained: bool,
    data_loader: DataLoader,
    optimizer: Union[cooper.ConstrainedOptimizer, torch.optim.Optimizer],
    num_epochs: int,
    criterion: torch.nn.modules.loss._Loss = None,
    cmp: cooper.CMPState = None,
    formulation: cooper.LagrangianFormulation = None,
    k: float = None,
    use_wandb: bool = True,
) -> Tuple[torch.nn.Module, OrderedDict]:
    """Trains a model using the data_loader and the formulation, cmp, and coop objects for the
    optimization.
    Args:
        model (torch.nn.Module): The model to train.
        is_constrained (bool): Whether the model is constrained or not.
        data_loader (DataLoader): The data loader to use.
        optimizer (t.Union[cooper.ConstrainedOptimizer, torch.optim.Optimizer]): The optimizer to
            use.
        num_epochs (int): The number of epochs to train for.
        criterion (torch.nn.modules.loss._Loss): The loss function to use.
        cmp (cooper.CMPState): The CMP state to use.
        formulation (cooper.LagrangianFormulation): The formulation to use.
        k (float): The slack term of the inequality defects.

    Returns:
        torch.nn.Module: The trained model.
    """
    state_history = OrderedDict()
    iter_num = 0
    for _ in range(num_epochs):
        for inputs, targets in data_loader:
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Backward
            if is_constrained:
                lagrangian = formulation.composite_objective(
                    cmp.closure, model, inputs, targets, k
                )
                formulation.custom_backward(lagrangian)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
            # Update parameters
            if is_constrained:
                optimizer.step(cmp.closure, model, inputs, targets, k)
            else:
                optimizer.step()
            if iter_num % 5 == 0:
                if is_constrained:
                    state_history[iter_num] = {
                        "cmp": cmp.state,
                        "dual": deepcopy(formulation.state()),
                    }
                    if use_wandb:
                        train_log(cmp.state, formulation.state(), iter_num)
                else:
                    state_history[iter_num] = {
                        "loss": loss.item(),
                    }
                    if use_wandb:
                        wandb.log({"iter": iter_num,"loss": loss.item()})
            iter_num += 1
    return model, state_history


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    config: wandb.config,
    use_wandb: bool = True,
) -> Tuple[torch.nn.Module, OrderedDict]:
    """Train the model.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The data loader to use for training.
        num_epochs (int): The number of epochs to train for.
        is_constrained (bool): Whether to train the model constrained or unconstrained.
        unconst_lr (float, optional): The learning rate to use for the unconstrained model.
        primal_lr (float, optional): The learning rate to use for the primal variable.
        dual_lr (float, optional): The learning rate to use for the dual variable.
        k (int, optional): The slack term of the inequality defects.
    Returns:
        tuple: A tuple of the trained model and the state history.
    """
    is_constrained = config.is_constrained
    criterion = torch.nn.BCELoss()
    wandb.watch(model, criterion, log="all", log_freq=10)

    if is_constrained:
        cmp = NormConstrainedLogisticRegression()
        formulation, optimizer = set_optimizers(
            cmp, model.parameters(), config.primal_lr, config.dual_lr
        )
        model, state_history = train_model(
            model,
            is_constrained,
            train_loader,
            optimizer,
            num_epochs,
            cmp=cmp,
            formulation=formulation,
            k=config.k,
            use_wandb=use_wandb,
        )

    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.unconst_lr)
        model, state_history = train_model(
            model,
            is_constrained,
            train_loader,
            optimizer,
            num_epochs,
            criterion=criterion,
            use_wandb=use_wandb,
        )

    return model, state_history


def evaluate_model(
    model: torch.nn.Module, data_loader: DataLoader, mode: str, use_wandb: bool = True
) -> float:
    """Evaluate the model on the data.
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): The data loader to use.
        mode (str): The mode to use. Valid modes are "train", "val", and "test".
    Returns:
        float: The accuracy of the model on the data.
    """
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for inputs, targets in data_loader:
            pred_logits = model.forward(inputs)
            predicted = pred_logits > 0.5
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    if use_wandb:
        wandb.log({mode + "_accuracy": correct / total})
    return correct * 100 / total


def experimentation_pipeline(hyperparameters: dict) -> torch.nn.Module:
    """Experimentation pipeline.
    Args:
        is_constrained (bool): Whether to train a constrained model.
        unconst_lr (float, optional): The learning rate to use for the unconstrained model.
        primal_lr (float, optional): The learning rate of the primal optimizer.
        dual_lr (float, optional): The learning rate of the dual optimizer.
        k (float, optional): The slack term of the inequality defects.
        batch_size (int, optional): The batch size.
        n_iters (int, optional): The number of iterations.
        train_ratio (float, optional): The ratio of the training set.
    Returns:
        torch.nn.Module: The trained model."""

    with wandb.init(project="first_tests", config=hyperparameters):

        config = wandb.config

        train_loader, val_loader, test_loader, model, num_epochs = preparation(config)

        model, _ = train(model, train_loader, num_epochs, config)

        _ = evaluate_model(model, train_loader, "train")
        _ = evaluate_model(model, val_loader, "val")
        _ = evaluate_model(model, test_loader, "test")

    return model
