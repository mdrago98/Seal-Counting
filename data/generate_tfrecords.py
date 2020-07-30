from collections import namedtuple
from functools import partial
from io import BytesIO
from itertools import permutations
from os import path
from random import randint
from typing import Callable
from numpy import int64

import tensorflow as tf
from pandas import DataFrame, read_excel, concat as pd_concat
from PIL import Image
from pathlib import Path
from absl import flags, app
from tqdm import tqdm
from os import getcwd

from helpers.utils import get_logger


from data.image_handler import (
    get_bbox,
    is_in_bounding_box,
    normalise_coordinates,
    generate_bbox,
    extract_intervals,
    BBox,
)
from data.utils import tf_example_utils
from data.pipeline import PIPELINE
from sklearn.model_selection import train_test_split

import multiprocessing

from helpers.utils import timer

Image.MAX_IMAGE_PIXELS = 99999999999999999999

# DEFINE the script flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pixel_coord",
    path.join(getcwd(), "data_files", "pixel_coord.xlsx"),
    "The excel file containing the locations and file sizes",
)
flags.DEFINE_string(
    "output_location",
    path.join(getcwd(), "output", "tfrecords"),
    "The output in which to store the output",
)
flags.DEFINE_boolean(
    "save_png", False, "A boolean flag indicating whether to save the cropped png image",
)
flags.DEFINE_integer("image_size", 416, "The image size to crop to.")
flags.DEFINE_float("train_size", 0.8, "The size of the training data")
flags.DEFINE_float("validation", 0.2, "The size of the training data")


data = namedtuple("data", ["filename", "object", "interval"])
transformed_example = namedtuple("TransformedExample", ["filename", "encoded", "object"])

logger = None


def get_random_crop_coord(
    image_size: tuple, crop_size: tuple, rand: Callable = randint, seed: int = 42
) -> BBox:
    """
    A function to get the random cropping region
    :param seed: the random state
    :param image_size: the image size
    :param crop_size: the tuple of the resultant crop region
    :param rand: the random function
    :return: the bbox
    """
    x_min = rand(0, image_size[0], random_state=seed)
    y_min = rand(0, image_size[1], random_state=seed)
    x_max = x_min + crop_size[0]
    y_max = y_min + crop_size[1]
    return get_bbox((x_min, x_max), (y_min, y_max))


def decode_img(img, locations: DataFrame, size: tuple = (416, 416), seed=42) -> tuple:
    """
    A function to decode an image and randomly crop to the required size.
    :param img: the image
    :param locations: the seal locations
    :param size: the crop size
    :param seed: the random state seed
    :return: A tuple of image tensor input and the output in pandas DataFrame
    representation.
    """

    img = tf.image.decode_png(img, channels=3)
    if not locations.empty:
        crop_box = locations.sample(1, random_state=seed)
        crop_box = get_seal_cropping_region(crop_box, size)
        crop_box = get_bbox(
            (crop_box["x_min"].iloc[0], crop_box["x_max"].iloc[0]),
            (crop_box["y_min"].iloc[0], crop_box["y_max"].iloc[0]),
        )
    else:
        # TODO: convert to int 32
        crop_box = get_random_crop_coord(tf.shape(img), crop_size=size, seed=seed)
    img = tf.image.crop_to_bounding_box(img, crop_box.x_min, crop_box.y_min, size[0], size[0])
    exists_in_image = locations[["x_pixel", "y_pixel"]].apply(
        lambda x: is_in_bounding_box(crop_box, x), axis=1
    )
    normalised_coord = normalise_coordinates(crop_box, locations[exists_in_image])
    normalised_coord = generate_bbox(normalised_coord)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, normalised_coord.values


def get_seal_cropping_region(
    location: DataFrame, size=(416, 416), x_name="x", y_name="y"
) -> DataFrame:
    """
    A function that maps a data frame of seal locations to random cropped images
    Args:
        location (DataFrame): the seal location dataframe
        size (tuple): the tuple of width and height representing the output size
        x_name (str): the x name to use in the resulting frame
        y_name (str): the y name to use in the resulting frame

    Returns: the augmented dataframe
    """
    location_frame = location.copy()
    location_frame.dropna()
    y_displacement = size[1] / 2
    x_displacement = size[0] / 2
    location_frame[f"{y_name}_min"] = location_frame[f"y_pixel"].apply(
        lambda x: max(x - y_displacement, 0)
    )
    location_frame[f"{x_name}_min"] = location_frame[f"x_pixel"].apply(
        lambda x: max(x - x_displacement, 0)
    )
    location_frame[f"{x_name}_max"] = location_frame[f"{x_name}_min"].apply(
        lambda x: min(x + size[0], location_frame["image_width"].iloc[0])
    )
    location_frame[f"{y_name}_max"] = location_frame[f"{y_name}_min"].apply(
        lambda x: min(x + size[1], location_frame["image_height"].iloc[0])
    )
    return location_frame


def clean_data(
    dataset: DataFrame, columns: list = None, pipeline: list = None, drop_na: bool = True
) -> DataFrame:
    """
    A function that cleans the dataframe.
     Args: pipeline (list): a list of tuples representing the column name and
     the
    function it is applied on taking in a series
     columns (list): the columns to select data (DataFrame): the
    dataframe to clean

    Returns: the transformed dataset
    """
    if pipeline is None:
        pipeline = PIPELINE
    if columns is None:
        columns = ["tiff_file", "layer_name", "x_pixel", "y_pixel", "image_width", "image_height"]
    dataset = dataset[columns]
    if drop_na:
        dataset = dataset.copy().dropna()
    for column_name, transformation in pipeline:
        dataset[column_name] = transformation(dataset[column_name])
    return dataset


def generate_crop_locations(
    locations: DataFrame,
    file_overview: DataFrame,
    initial_size: tuple,
    max_iter: int = 20,
    step: int = 2,
) -> tuple:
    """
    A function that generate the crop locations into a dataframe
    :param locations: the locations dataframe
    :param file_overview: a dataframe of the image widths heights
    :param initial_size: the initial size to crop
    :param max_iter: the number of sizes to generate
    :param step: the step size
    :return: the tuple of augmented dataframe and the step sizes
    """
    locations = locations.merge(
        file_overview[["tiff_file", "image_width", "image_height"]], how="inner", on="tiff_file",
    )
    size_increments = [
        tuple([size_param * i for size_param in initial_size]) for i in range(1, max_iter, step)
    ]
    cropping_regions = locations.copy()
    for increment in size_increments:
        region = get_seal_cropping_region(
            locations, increment, x_name=f"x_{increment[0]}", y_name=f"y_{increment[0]}"
        )
        cropping_regions = cropping_regions.merge(region)
    return cropping_regions, size_increments


def extrapolate_crops_output(
    image: str, outputs: DataFrame, size: tuple, object_bbox: tuple = (60, 60), ignore_extrema=True
) -> list:
    """
    A function to extrapolate the cropped regions and normalise the related coordinates
    :param object_bbox: the seal bounding box size
    :param outputs: the output dataframe
    :param image: the image file name
    :param size: the resulting size
    :param ignore_extrema: a boolean flag that when set True ignores the border whiteboxes
    :return: the list of data named tuples consisting of the cropped input image tensor and the output
    """
    transformed = []
    # TODO: parameterize path + get absolute path
    img_path = path.join("/data2/seals/TIFFs", image)
    if Path(img_path).exists():
        try:
            with Image.open(img_path) as img:
                # TODO refactor code to crop image using PIL + refactor extrapolate image to
                # crop_box = get_seal_cropping_region(outputs, size)[
                #     ["x_min", "y_min", "x_max", "y_max"]
                # ]
                intervals = extract_intervals(img.size, size)
                intervals = list(permutations(intervals[0], r=2))
                for i, (x, y) in enumerate(intervals):
                    box: BBox = get_bbox(
                        (x[0], x[1]), (y[0], y[1]),
                    )
                    cropped = img.crop((box.x_min, box.y_min, box.x_max, box.y_max,))
                    extrema = cropped.convert("L").getextrema()
                    if extrema[0] == extrema[1] and ignore_extrema:
                        continue
                    img_val = BytesIO()
                    cropped.save(img_val, format="PNG")
                    box_filter = outputs[["x_pixel", "y_pixel"]].apply(
                        lambda x: is_in_bounding_box(box, x), axis=1
                    )
                    transformed_out = outputs[box_filter]
                    transformed_out = normalise_coordinates(box, transformed_out)
                    transformed_out = generate_object_bbox(transformed_out, size, object_bbox)
                    transformed_out["image_width"] = transformed_out["image_width"].apply(
                        lambda _: size[0]
                    )
                    transformed_out["image_height"] = transformed_out["image_height"].apply(
                        lambda _: size[1]
                    )
                    transformed += [data(img_val.getvalue(), transformed_out, box)]
        except Exception as e:
            logger.error(e.__str__())
    else:
        logger.warning(f"Image {img_path} not found")
    return transformed


def generate_object_bbox(
    locations: DataFrame, crop_image_size: tuple = (416, 416), bbox_size: tuple = (80, 80)
):
    # TODO: refactor logic to crop the image first and get the bbox afterwards
    locations = locations.copy()
    locations["xmin"] = locations.x_pixel.apply(lambda x: max(x - (bbox_size[0] / 2), 0)).astype(
        int64
    )
    locations["ymin"] = locations.y_pixel.apply(lambda x: max(x - (bbox_size[1] / 2), 0)).astype(
        int64
    )
    locations["xmax"] = locations.xmin.apply(
        lambda x: min(x + bbox_size[0], crop_image_size[0])
    ).astype(int64)
    locations["ymax"] = locations.ymin.apply(
        lambda x: min(x + bbox_size[1], crop_image_size[1])
    ).astype(int64)
    return locations


def convert_to_tf_records(group: data, size: tuple, name) -> tf.train.Example:
    """
    A function to convert the group into a tf record
    :param group: the data named tuple group
    :return: the the Example object
    """
    object = group.object
    image = group.filename
    interval = group.interval
    height = object["image_height"].get(0, size[0])
    width = object["image_width"].get(0, size[0])
    xmin = object["xmin"].apply(lambda x: x / width)
    xmax = object["xmax"].apply(lambda x: x / width)
    ymin = object["ymin"].apply(lambda y: y / height)
    ymax = object["ymax"].apply(lambda y: y / height)
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": tf_example_utils.int64_feature(height),
                "image/width": tf_example_utils.int64_feature(width),
                "image/source_id": tf_example_utils.bytes_feature(
                    object["tiff_file"].get(0, name).encode()
                ),
                "image/encoded": tf_example_utils.bytes_feature(image),
                "image/format": tf_example_utils.bytes_feature("tif".encode()),
                "image/object/bbox/xmin": tf_example_utils.float_list_feature(xmin.tolist()),
                "image/object/bbox/xmax": tf_example_utils.float_list_feature(xmax.tolist()),
                "image/object/bbox/ymin": tf_example_utils.float_list_feature(ymin.tolist()),
                "image/object/bbox/ymax": tf_example_utils.float_list_feature(ymax.tolist()),
                "image/object/class/text": tf_example_utils.string_list(object["layer_name"]),
                "image/object/bbox/x_pixel": tf_example_utils.float_list_feature(
                    object["x_pixel"].tolist()
                ),
                "image/object/bbox/y_pixel": tf_example_utils.float_list_feature(
                    object["y_pixel"].tolist()
                ),
                "patch/xmin": tf_example_utils.int64_feature(interval[0]),
                "patch/ymin": tf_example_utils.int64_feature(interval[1]),
                "patch/xmax": tf_example_utils.int64_feature(interval[2]),
                "patch/ymax": tf_example_utils.int64_feature(interval[3]),
            }
        )
    )


def split(dataset: DataFrame, group_key: str) -> list:
    """
    A function to split the dataset into groups by a given key
    :param dataset: the dataset to group by
    :param group_key: the grouping key
    :return: a list of tuples such that each tuple consists of the key and the grouped dataframe
    """
    gb = dataset.groupby(group_key)
    return [
        data(filename, gb.get_group(x), None) for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


@timer(logger)
def write_to_record(
    dataset: DataFrame,
    output_dir: str,
    name: str,
    size: tuple = (416, 416),
    ignore_border: bool = True,
    n_jobs: int = 3,
):
    """
    A function to write the dataset to tf records
    :param ignore_border: a boolean flag indicating whether to ignore patches with the white border
    :param n_jobs: number of concurrent processes
    :param dataset: the dataset dataframe
    :param output_dir: the output directory
    :param name: the name of the tf record
    :param size: the image crop size
    :return: None
    """

    # TODO: remove 10 index slice
    grouped_train = split(dataset, "tiff_file")
    filenames = [filename for filename, _, _ in grouped_train]
    all_data = DataFrame()
    Path(path.join(output_dir, name)).mkdir(exist_ok=True, parents=True)
    extract_island = partial(
        extract_island_crops,
        name=name,
        output_dir=output_dir,
        size=size,
        ignore_border=ignore_border,
    )
    total_patches = 0
    with multiprocessing.Pool(n_jobs) as pool:
        for island_records, n_island_patches in tqdm(
            pool.imap_unordered(extract_island, grouped_train), total=len(grouped_train)
        ):
            all_data = pd_concat([all_data, island_records])
            total_patches += n_island_patches
            logger.info(f"Extracted {n_island_patches} patches. Total {total_patches}")
    all_records_path = path.join(output_dir, f"{size[0]}_{name}_all_records.csv")
    logger.info(f"Writing all normalised records to {all_records_path=}")
    all_data.drop_duplicates(keep=False, inplace=True)
    all_data.to_csv(path.join(output_dir, name, f"records.csv"))


def extract_island_crops(
    record: data, name: str, output_dir: str, size: tuple, ignore_border: bool
):
    island, output, _ = record
    output_path = path.join(output_dir, name, f"{path.splitext(island)[0]}.tfrecord")
    writer = tf.compat.v1.python_io.TFRecordWriter(output_path)
    height = output["image_height"].iloc[0]
    transformed_out = output.copy()
    transformed_out["y_pixel"] = transformed_out["y_pixel"].apply(lambda y: height - y)
    converted = extrapolate_crops_output(island, transformed_out, size)
    island_records = DataFrame()
    if converted is not None:
        island_records = pd_concat([island_records, *[data.object for data in converted]])
        for converted_object in converted:
            tf_example = convert_to_tf_records(converted_object, size, name)
            writer.write(tf_example.SerializeToString())
    writer.flush()
    writer.close()
    return island_records, len(converted)


def create_output_dir(output_dir: str, image_size: int) -> str:
    """
    A function that checks the output dir and prepares it for  any modifications.
    :param output_dir: the output directory
    :param image_size: the image size
    :return: the fully formed output path
    """
    output_path = path.join(output_dir, str(image_size))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path


def write_classes(output_dir, classes: list, name="seals.name"):
    output_dir = path.join(output_dir, name)
    Path(output_dir).parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing classes to {output_dir}")
    with open(output_dir, "a") as file:
        logger.info("Writing class names to")
        file.write("\n".join(classes))
    return True


def main(argv: list):
    # TODO: add flag for validation dataset
    global logger
    logger = get_logger(FLAGS.output_location)
    out_size = (FLAGS.image_size, FLAGS.image_size)
    locations = read_excel(FLAGS.pixel_coord, sheet_name="PixelCoordinates",)
    file_props = read_excel(FLAGS.pixel_coord, sheet_name="FileOverview",).dropna()
    locations = locations.merge(
        file_props[["tiff_file", "image_width", "image_height"]], how="inner"
    )
    write_classes(FLAGS.output_location, list(locations["layer_name"].dropna().unique()))
    locations = clean_data(locations)
    logger.info("Splitting the dataset into train, validation, testing")
    train, test = split_frame(locations)
    logger.info(f"Cleaned the dataset generating tf_records")

    logger.info("Generating training records")
    write_to_record(train, FLAGS.output_location, "train", size=out_size)

    logger.info("Generating testing records")
    write_to_record(test, FLAGS.output_location, "test", size=out_size, ignore_border=False)


def split_frame(locations):
    file_names = locations["tiff_file"].unique()
    train, test = train_test_split(file_names, train_size=FLAGS.train_size, random_state=42)
    return (
        locations[locations["tiff_file"].isin(train)],
        locations[locations["tiff_file"].isin(test)],
    )


if __name__ == "__main__":
    app.run(main)
