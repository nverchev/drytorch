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
        self.options = {
            'display.precision': precision,
            'display.max_rows': max_rows,
            'display.max_columns': max_columns
        }
        self.original_options = {}

    def __enter__(self):
        self.original_options = {key: pd.get_option(key) for key in self.options}
        for key, value in self.options.items():
            pd.set_option(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.original_options.items():
            pd.set_option(key, value)


class UsuallyFalse:
    """
    Abuses the with statement for a temporary change
    """
    _value: bool = False

    def __bool__(self):
        """
        Usually, it evaluates to False

        Returns False when outside the with statement else True
        """
        return self._value

    def __enter__(self):
        self._value = True

    def __exit__(self, *_):
        self._value = False

    def __repr__(self):
        return self._value
