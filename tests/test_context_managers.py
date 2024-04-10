import pandas as pd
from dry_torch.context_managers import PandasPrintOptions, UsuallyFalse


def test_usually_false() -> None:
    attribute = UsuallyFalse()

    # False
    assert not attribute

    # True
    with attribute:
        assert attribute

    # back to False
    assert not attribute


def test_pandas_print_options() -> None:
    df = pd.DataFrame({'col_' + str(i): 10 * [3.141592] for i in range(10)})
    expected = ('    col_0  ...  col_9\n'
                '0    3.14  ...   3.14\n'
                '..    ...  ...    ...\n'
                '9    3.14  ...   3.14\n'
                '\n'
                '[10 rows x 10 columns]')
    with PandasPrintOptions(precision=2, max_rows=2, max_columns=2):
        assert str(rf'{df}') == expected
