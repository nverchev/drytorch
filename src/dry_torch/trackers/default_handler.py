import functools
from typing import Any

from src.dry_torch import log_events
from src.dry_torch import tracking


class DefaultHandler(tracking.Handler):

    def __init__(self) -> None:
        super().__init__()
        self.ctx: dict[str, Any] = {}

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return

