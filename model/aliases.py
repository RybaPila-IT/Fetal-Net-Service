from numpy import ndarray
from typing import Union

# Type representing the prediction obtained from the measurer.
prediction = tuple[str, str, ndarray]
classification = tuple[str, float]
img_size = tuple[float, float]
measurement_type = Union[float | tuple[float, float]]
measurement_with_img = tuple[measurement_type, ndarray]
