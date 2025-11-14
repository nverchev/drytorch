"""Configuration module with objects from the package."""

from tests.functional.conftest import (
    identity_dataset,
    identity_loader,
    identity_trainer,
    linear_model,
    run,
    square_loss_calc,
    standard_learning_schema,
    zero_metrics_calc,
)


_fixtures = (
    linear_model,
    identity_dataset,
    identity_loader,
    zero_metrics_calc,
    square_loss_calc,
    standard_learning_schema,
    identity_trainer,
    run,
)
