"""Functional tests for wandb tracker.

These tests are meant to be run manually to verify integration with the WandB website.
They are skipped during automated testing.
"""

import pytest
import time
import numpy as np
from wandb.sdk import wandb_settings

from dry_torch import log_events
from dry_torch.trackers import wandb


class TestWandbFunctional:
    """Functional tests for WandB tracker that create actual WandB entries.

    Skip these tests during automated runs. Run them manually when you need
    to verify the behavior on the WandB website.
    """

    @pytest.fixture(autouse=True)
    def setup(self, stop_experiment_event):
        """Setup unique experiment name for every test run."""
        self.unique_id = int(time.time())
        self.project_name = f"test_wandb_tracker_{self.unique_id}"

        # Create a tracker with a unique project name
        self.tracker = wandb.Wandb(
            settings=wandb_settings.Settings(project=self.project_name)
        )

        # Create start experiment event
        self.start_event = log_events.StartExperiment(
            exp_name=self.project_name,
            exp_dir="./wandb_test_output",
            config={
                "test_id": self.unique_id,
                "framework": "dry_torch",
                "test_type": "functional"
            }
        )

        # Start the experiment
        self.tracker.notify(self.start_event)

        yield

        # Clean up
        self.tracker.notify(stop_experiment_event)

        print(
            f"\nNavigate to W&B website to see the results for project: {self.project_name}")

    def test_basic_metrics_logging(self):
        """Test basic metrics logging that can be verified on the WandB website."""
        # Create model and test data
        model_name = "TestModel"

        # Log metrics for 10 epochs
        for epoch in range(10):
            # Create simulated metrics
            accuracy = 0.5 + (epoch * 0.05)  # Increasing accuracy
            loss = 1.0 - (epoch * 0.1)  # Decreasing loss

            # Add some noise to make the charts more realistic
            accuracy += np.random.normal(0, 0.02)
            loss += np.random.normal(0, 0.05)

            # Clip to reasonable values
            accuracy = min(max(accuracy, 0.5), 0.99)
            loss = max(loss, 0.1)

            # Create metrics event for training
            train_metrics_event = log_events.Metrics(
                model_name=model_name,
                source="train",
                epoch=epoch,
                metrics={
                    "accuracy": accuracy,
                    "loss": loss,
                    "learning_rate": 0.01 * (0.9 ** epoch)
                }
            )

            # Create metrics event for validation with slightly different values
            val_metrics_event = log_events.Metrics(
                model_name=model_name,
                source="val",
                epoch=epoch,
                metrics={
                    "accuracy": accuracy - 0.05,
                    "loss": loss + 0.2,
                }
            )

            # Notify the tracker
            self.tracker.notify(train_metrics_event)
            self.tracker.notify(val_metrics_event)

            # Small delay to ensure WandB processes events
            time.sleep(0.1)

        print(f"\nLogged 10 epochs of metrics for model '{model_name}'")
        print(
            f"Check the WandB project '{self.project_name}' to see the results")

    def test_multiple_models_comparison(self):
        """Test logging metrics from multiple models for comparison on WandB."""
        models = ["ModelA", "ModelB", "ModelC"]

        # Log metrics for 5 epochs
        for epoch in range(5):
            for i, model_name in enumerate(models):
                # Create different performance profiles for each model
                base_accuracy = 0.6 + (
                            i * 0.05)  # ModelC starts better than ModelA
                accuracy = base_accuracy + (epoch * 0.07)
                loss = 1.0 - (i * 0.1) - (epoch * 0.15)

                # Add some noise
                accuracy += np.random.normal(0, 0.01)
                loss += np.random.normal(0, 0.03)

                # Clip values
                accuracy = min(max(accuracy, 0.5), 0.99)
                loss = max(loss, 0.1)

                # Create metrics event
                metrics_event = log_events.Metrics(
                    model_name=model_name,
                    source="train",
                    epoch=epoch,
                    metrics={
                        "accuracy": accuracy,
                        "loss": loss,
                        "model_type": i
                        # Numeric parameter to show in parallel coords plot
                    }
                )

                # Notify the tracker
                self.tracker.notify(metrics_event)

            # Small delay
            time.sleep(0.1)

        print(f"\nLogged 5 epochs of metrics for models: {', '.join(models)}")
        print(
            f"Check the WandB project '{self.project_name}' to compare models")
        print(
            "Try using the 'parallel coordinates' view to compare model performance")