"""
Models module.
"""

import torch
import cooper

class LogisticRegression(torch.nn.Module):

    """Logistic Regression model."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize the model.
        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
        """
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the model.
        """
        return self.linear(inputs.view(-1, self.input_size))

class NormConstrainedLogisticRegression(cooper.ConstrainedMinimizationProblem):

    """Norm-constrained Logistic Regression model."""

    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        super().__init__(is_constrained=True)

    def closure(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        k: float,
    ) -> cooper.CMPState:
        """Compute the loss.
        Args:
            model (torch.nn.Module): The model to compute the loss for.
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target data.
            k (float): The slack term of the inequality defects.
        Returns:
            cooper.CMPState: The state of the CMP.
        """
        pred_logits = model.forward(inputs)
        loss = self.criterion(pred_logits, targets)
        ineq_defect = torch.sum(torch.pow(model.linear.weight, 2)) - k
        return cooper.CMPState(loss=loss, ineq_defect=ineq_defect)
