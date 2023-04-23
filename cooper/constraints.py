import warnings
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Sequence, Tuple, Union

import torch

from cooper import multipliers
from cooper.formulation import FORMULATION_TYPE, Formulation

CONSTRAINT_TYPE = Literal["eq", "ineq", "penalty"]


@dataclass
class ConstraintState:
    """State of a constraint group describing the current constraint violation.

    Args:
        violation: Measurement of the constraint violation at some value of the primal
            parameters. This is expected to be differentiable with respect to the
            primal parameters.
        strict_violation: Measurement of the constraint violation which may be
            non-differentiable with respect to the primal parameters. When provided,
            the (necessarily differentiable) `violation` is used to compute the gradient
            of the Lagrangian with respect to the primal parameters, while the
            `strict_violation` is used to compute the gradient of the Lagrangian with
            respect to the dual parameters. For more details, see the proxy-constraint
            proposal of :cite:t:`cotter2019JMLR`.
        constraint_features: The features of the constraint. This is used to evaluate
            the lagrange multiplier associated with a constraint group. For example,
            An `IndexedMultiplier` expects the indices of the constraints whose Lagrange
            multipliers are to be retrieved; while an `ImplicitMultiplier` expects
            general tensor-valued features for the constraints. This field is not used
            for `DenseMultiplier`//s.
            This can be used in conjunction with an `IndexedMultiplier` to indicate the
            measurement of the violation for only a subset of the constraints within a
            `ConstraintGroup`.
        skip_primal_conribution: When `True`, we ignore the contribution of the current
            observed constraint violation towards the primal Lagrangian, but keep their
            contribution to the dual Lagrangian. In other words, the observed violations
            affect the update for the dual variables but not the update for the primal
            variables.
        skip_dual_conribution: When `True`, we ignore the contribution of the current
            observed constraint violation towards the dual Lagrangian, but keep their
            contribution to the primal Lagrangian. In other words, the observed
            violations affect the update for the primal variables but not the update
            for the dual variables. This flag is useful for performing less frequent
            updates of the dual variables (e.g. after several primal steps).
    """

    violation: torch.Tensor
    strict_violation: Optional[torch.Tensor] = None
    constraint_features: Optional[torch.Tensor] = None
    skip_primal_contribution: bool = False
    skip_dual_contribution: bool = False


class ConstraintGroup:
    """Constraint Group."""

    # TODO(gallego-posada): Add documentation
    # TODO(gallego-posada): add docstring explaining that when passing the multiplier
    # directly, the other kwargs (shape, dtype, device) are ignored

    def __init__(
        self,
        constraint_type: CONSTRAINT_TYPE,
        formulation_type: Optional[FORMULATION_TYPE] = "lagrangian",
        formulation_kwargs: Optional[dict] = {},
        multiplier: Optional[multipliers.MULTIPLIER_TYPE] = None,
        multiplier_kwargs: Optional[dict] = {},
    ):

        self._state: ConstraintState = None

        self.constraint_type = constraint_type
        self.formulation = self.build_formulation(formulation_type, formulation_kwargs)

        if multiplier is None:
            multiplier = multipliers.build_explicit_multiplier(constraint_type, **multiplier_kwargs)
        self.sanity_check_multiplier(multiplier)
        self.multiplier = multiplier

    def build_formulation(self, formulation_type, formulation_kwargs):
        if self.constraint_type == "penalty":
            # `penalty` constraints must be paired with "penalized" formulations. If no formulation is provided, we
            # default to a "penalized" formulation.
            if formulation_type != "penalized":
                warning_message = (
                    "A constraint of type `penalty` must be used with a `penalized` formulation, but received"
                    f" formulation_type={formulation_type}. The formulation type will be set to `penalized`."
                    " Please review your configuration and override the default formulation_type='lagrangian'."
                )
                warnings.warn(warning_message)
            formulation_type = "penalized"

        return Formulation(formulation_type, **formulation_kwargs)

    def sanity_check_multiplier(self, multiplier: multipliers.MULTIPLIER_TYPE) -> None:

        if (self.constraint_type == "penalty") and not isinstance(multiplier, multipliers.ConstantMultiplier):
            # If a penalty "constraint" is used, then we must have been provided a ConstantMultiplier.
            raise ValueError("A ConstantMultiplier must be provided along with a `penalty` constraint.")

        if isinstance(multiplier, multipliers.ConstantMultiplier):
            if any(multiplier() < 0) and (self.constraint_type == "ineq"):
                raise ValueError("All entries of ConstantMultiplier must be non-negative for inequality constraints.")

        if isinstance(multiplier, multipliers.ExplicitMultiplier):
            if multiplier.implicit_constraint_type != self.constraint_type:
                raise ValueError(f"Provided multiplier is inconsistent with {self.constraint_type} constraint.")

    @property
    def state(self) -> ConstraintState:
        return self._state

    @state.setter
    def state(self, value: ConstraintState) -> None:
        if isinstance(self.multiplier, (multipliers.IndexedMultiplier, multipliers.ImplicitMultiplier)):
            if value.constraint_features is None:
                raise ValueError(f"Multipliers of type {type(self.multiplier)} expect constraint features.")

        self._state = value

    def compute_lagrangian_contribution(
        self, constraint_state: Optional[ConstraintState] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the contribution of the current constraint to the primal and dual
        Lagrangians, and evaluates the associated Lagrange multiplier."""

        if constraint_state is None and self.state is None:
            raise ValueError("A `ConstraintState` (provided or internal) is needed to compute Lagrangian contribution")
        elif constraint_state is None:
            constraint_state = self.state

        if constraint_state.constraint_features is None:
            multiplier_value = self.multiplier()
        else:
            multiplier_value = self.multiplier(constraint_state.constraint_features)

        # Strict violation represents the "actual" violation of the constraint.
        # We use it to update the value of the multiplier.
        if constraint_state.strict_violation is not None:
            strict_violation = constraint_state.strict_violation
        else:
            # If strict violation is not provided, we use the differentiable
            # violation (which always exists).
            strict_violation = constraint_state.violation

        primal_contribution, dual_contribution = self.formulation.compute_lagrangian_contribution(
            constraint_type=self.constraint_type,
            multiplier_value=multiplier_value,
            violation=constraint_state.violation,
            strict_violation=strict_violation,
            skip_primal_contribution=constraint_state.skip_primal_contribution,
            skip_dual_contribution=constraint_state.skip_dual_contribution,
        )

        return multiplier_value, primal_contribution, dual_contribution

    def state_dict(self):
        return {
            "constraint_type": self.constraint_type,
            "formulation": self.formulation.state_dict(),
            "multiplier_state_dict": self.multiplier.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.formulation.load_state_dict(state_dict["formulation"])
        self.multiplier.load_state_dict(state_dict["multiplier_state_dict"])

    def __repr__(self):
        return f"ConstraintGroup(constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier})"


def observed_constraints_iterator(
    observed_constraints: Sequence[Union[ConstraintGroup, Tuple[ConstraintGroup, ConstraintState]]]
) -> Iterator[Tuple[ConstraintGroup, ConstraintState]]:
    """Utility function to iterate over observed constraints. This allows for consistent
    iteration over `observed_constraints` which are a sequence of `ConstraintGroup`\\s
    (and hold the `ConstraintState` internally), or a sequence of
    `Tuple[ConstraintGroup, ConstraintState]`\\s.
    """

    for constraint_tuple in observed_constraints:
        if isinstance(constraint_tuple, ConstraintGroup):
            constraint_group = constraint_tuple
            constraint_state = constraint_group.state
        elif isinstance(constraint_tuple, tuple) and len(constraint_tuple) == 2:
            constraint_group, constraint_state = constraint_tuple
        else:
            error_message = f"Received invalid format for observed constraint. Expected {ConstraintGroup} or"
            error_message += f" {Tuple[ConstraintGroup, ConstraintState]}, but received {type(constraint_tuple)}"
            raise ValueError(error_message)

        yield constraint_group, constraint_state
