from pathlib import Path

from numpy import int64
from pandas import Series


def binarize_column(series: Series, labels_path: str = None) -> Series:
    """
    A function that converts a series column into it's numeric representation.
    Args:
        series (Series): The series object

    Returns: the binarized series
    :param labels_path: the label path. If non the labels will not be written
    """
    series: Series = series.copy()
    col_mapping = {item: i for i, item in list(enumerate(series.unique()))}
    if labels_path is not None and Path(labels_path).exists():
        with open(labels_path, "rb") as file:
            file.write("\n".join(list(col_mapping.keys())))
    return series.apply(lambda x: col_mapping[x])


def make_integer(series: Series) -> Series:
    """
    Converts a series to numeric integer type.
    :param series: the series to convert
    :return: the converted series
    """
    return series.astype(int64)
