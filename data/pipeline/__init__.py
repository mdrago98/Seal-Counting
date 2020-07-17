from data.pipeline.pipeline_modules import binarize_column, make_integer

PIPELINE = [
    ("layer_name", binarize_column),
    ("x_pixel", make_integer),
    ("y_pixel", make_integer),
    ("image_width", make_integer),
    ("image_height", make_integer),
]
