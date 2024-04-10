from typing import Any
import pandas as pd


class PandasPrintOptions:
    """
    Context manager to temporarily set Pandas display options.

    Args:
        precision (int): Number of digits of precision for floating point output.
        max_rows (int): Maximum number of rows to display.
        max_columns (int): Maximum number of columns to display.
    """

    def __init__(self, precision: int = 3, max_rows: int = 10, max_columns: int = 10) -> None:
        self.options: dict[str, int] = {
            'display.precision': precision,
            'display.max_rows': max_rows,
            'display.max_columns': max_columns
        }
        self.original_options: dict[str, Any] = {}

    def __enter__(self) -> None:
        self.original_options.update({key: pd.get_option(key) for key in self.options})
        for key, value in self.options.items():
            pd.set_option(key, value)

    def __exit__(self,  exc_type: None = None, exc_val: None = None,  exc_tb: None = None) -> None:
        for key, value in self.original_options.items():
            pd.set_option(key, value)


class UsuallyFalse:
    """
    Abuse the with statement for a temporary change.
    """

    def __init__(self) -> None:
        self._value: bool = False

    def __bool__(self) -> bool:
        """
        Usually, evaluate to False.

        Returns:
             False when outside the with statement else True.
        """
        return self._value

    def __enter__(self) -> None:
        self._value = True

    def __exit__(self,  exc_type: None = None, exc_val: None = None,  exc_tb: None = None) -> None:
        self._value = False

    def __repr__(self) -> str:
        return self._value.__repr__()
