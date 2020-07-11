from collections import namedtuple
from os import path
from random import randint
from typing import Callable
from numpy import array, int64, random

import tensorflow as tf
from pandas import DataFrame, read_excel
from PIL import Image
from pathlib import Path

from tqdm import tqdm

from data import (
    BBox,
    dataset_util,
    generate_bbox,
    get_bbox,
    is_in_bounding_box,
    normalise_coordinates,
)
from data.pipeline import PIPELINE

Image.MAX_IMAGE_PIXELS = 99999999999999999999

logger = tf.get_logger()


data = namedtuple("data", ["filename", "object"])


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

    location_frame[f"{y_name}_min"] = location_frame[f"y_pixel"].apply(
        lambda x: max(x - randint(0, size[1]), 0)
    )
    location_frame[f"{x_name}_min"] = location_frame[f"x_pixel"].apply(
        lambda x: max(x - randint(0, size[0]), 0)
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


def extrapolate_crops_output(image: str, outputs: DataFrame, size: tuple, random_state=42) -> data:
    """
    A function to extrapolate the cropped regions and normalise the related coordinates
    :param outputs: the output dataframe
    :param image: the image file name
    :param size: the resulting size
    :param random_state: the random seed to use when randomising crops
    :return: the data named tuple consisting of the cropped input image tensor and the output
    """
    transformed = None
    # TODO: parameterize path + get absolute path
    img_path = path.join("/data2/seals/TIFFs", image)
    if Path(img_path).exists():
        try:
            with Image.open(img_path) as img:
                # TODO refactor code to crop image using PIL + refactor extrapolate image to
                #  take one image at a time
                bounding_box = get_seal_cropping_region(outputs, size).sample(
                    random_state=random_state
                )[["x_min", "y_min", "x_max", "y_max",]]
                bounding_box: BBox = get_bbox(
                    (bounding_box["x_min"], bounding_box["x_max"]),
                    (bounding_box["y_min"], bounding_box["y_max"]),
                )
                img = img.crop(
                    (
                        bounding_box.x_min,
                        bounding_box.y_min,
                        bounding_box.x_max,
                        bounding_box.y_max,
                    ),
                )
                img_tensor = tf.convert_to_tensor(array(img))
            box_filter = outputs[["x_pixel", "y_pixel"]].apply(
                lambda x: is_in_bounding_box(bounding_box, x), axis=1
            )
            outputs = outputs[box_filter]
            outputs = normalise_coordinates(bounding_box, outputs)
            # normalise_coordinates(bounding_box, outputs)
            transformed = data(img_tensor, outputs[box_filter])
        except Exception as e:
            logger.error(e.__str__())
    else:
        logger.warn(f"Image {img_path} not found")
    return transformed


def generate_object_bbox(
    locations: DataFrame, crop_image_size: tuple = (416, 416), bbox_size: tuple = (60, 60)
):
    locations = locations.copy()
    locations["xmin"] = locations.x_pixel.apply(lambda x: max(x - (bbox_size[0] / 2), 0)).astype(
        int64
    )
    locations["ymin"] = locations.y_pixel.apply(lambda x: max(x - (bbox_size[1] / 2), 0)).astype(
        int64
    )
    locations["xmax"] = locations.x_pixel.apply(
        lambda x: min(x + (bbox_size[0] / 2), crop_image_size[0])
    ).astype(int64)
    locations["ymax"] = locations.y_pixel.apply(
        lambda x: min(x + (bbox_size[1] / 2), crop_image_size[1])
    ).astype(int64)
    return locations


def convert_to_tf_records(group: data) -> tf.train.Example:
    """
    A function to convert the group into a tf record
    :param group: the data named tuple group
    :return: the the Example object
    """
    object = group.object
    image = group.filename
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(object["image_height"].iloc[0]),
                "image/width": dataset_util.int64_feature(object["image_width"].iloc[0]),
                "image/source_id": dataset_util.bytes_feature(object["tiff_file"].iloc[0].encode()),
                "image/encoded": dataset_util.bytes_feature(image.numpy().tostring()),
                "image/format": dataset_util.bytes_feature("tif".encode()),
                "image/object/bbox/xmin": dataset_util.float_list_feature(object["xmin"].tolist()),
                "image/object/bbox/xmax": dataset_util.float_list_feature(object["xmax"].tolist()),
                "image/object/bbox/ymin": dataset_util.float_list_feature(object["ymin"].tolist()),
                "image/object/bbox/ymax": dataset_util.float_list_feature(object["ymax"].tolist()),
                "image/object/class/label": dataset_util.int64_list_feature(
                    object["layer_name"].tolist()
                ),
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
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def write_to_record(dataset: DataFrame, output_dir: str, name: str, size: tuple = (416, 416)):
    """
    A function to write the dataset to tf records
    :param dataset: the dataset dataframe
    :param output_dir: the output directory
    :param name: the name of the tf record
    :param size: the image crop size
    :return:
    """
    output_dir = path.join(output_dir, f"{size[0]}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = path.join(output_dir, f"{name}.tfrecord")
    writer = tf.compat.v1.python_io.TFRecordWriter(output_path)
    grouped_train = split(dataset, "tiff_file")
    for image, output in tqdm(grouped_train):
        converted = extrapolate_crops_output(image, output, size)
        if converted is not None:
            converted = convert_to_tf_records(converted)
            writer.write(converted.SerializeToString())
    writer.flush()
    writer.close()


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


def write_classes(output_dir, classes: list, name="classes.txt"):
    output_dir = path.join(output_dir, name)
    logger.info(f"Writing classes to {output_dir}")
    with open(output_dir, "a") as file:
        logger.info("Writing class names to")
        file.write("\n".join(classes))
    return True


def main(out_size=(416 * 2, 416 * 2)):
    output_dir = create_output_dir("/data2/seals/tfrecords", out_size[0])
    locations = read_excel(
        "/home/md273/CS5099-working-copy/data"
        "/PixelCoordinates_HgPupCounts2016_VersionToUse_20181017-1.xlsx",
        sheet_name="PixelCoordinates",
    )
    file_props = read_excel(
        "/home/md273/CS5099-working-copy/data/pixel_coord.xlsx", sheet_name="FileOverview",
    ).dropna()
    locations = locations.merge(
        file_props[["tiff_file", "image_width", "image_height"]], how="inner"
    )
    write_classes(output_dir, list(locations["layer_name"].dropna().unique()))
    locations = clean_data(locations)
    locations = generate_object_bbox(locations, out_size)
    msk = random.rand(len(locations)) < 0.8
    train = locations[msk]
    test = locations[~msk]
    logger.info(f"Cleaned the dataset generating tf_records")
    # TODO convert to flag
    logger.info("Generating training records")
    write_to_record(train, "/data2/seals/tfrecords", "train", size=out_size)
    logger.info("Generating testing records")
    write_to_record(test, "/data2/seals/tfrecords", "test", size=out_size)


main()
