
import io
import logging
import time

import pytest

from dry_torch import log_events

from dry_torch.trackers.logging import BuiltinLogger
from dry_torch.trackers.tqdm import TqdmLogger


logger = logging.getLogger('dry_torch')


@pytest.mark.skip()
def test_epoch_logs(
        string_stream: io.StringIO,
        sample_metrics: dict[str, float],
        start_epoch_event: log_events.StartEpoch,
        iterate_batch_event: log_events.IterateBatch,
        end_epoch_event: log_events.EndEpoch,
        epoch_metrics_event: EpochMetrics,
):
    """
    Tests logging of epoch-related events.

    Args:
        start_epoch_event: StartEpoch event
        iterate_batch_event: IterateBatch event
        end_epoch_event: EndEpoch event
        final_metrics_event: FinalMetrics event
    """
    print('')
    builtin_logger = BuiltinLogger()
    tqdm_logger = TqdmLogger(out=string_stream, leave=True)

    def _notify(event: log_events.Event):
        builtin_logger.notify(event)
        tqdm_logger.notify(event)

    _notify(start_epoch_event)
    _notify(iterate_batch_event)
    for _ in range(iterate_batch_event.num_iter):
        time.sleep(1)
        iterate_batch_event.update(sample_metrics)
        print(string_stream.getvalue())
    _notify(epoch_metrics_event)
    _notify(end_epoch_event)
    print(string_stream.getvalue())
