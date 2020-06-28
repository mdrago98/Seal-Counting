import plac
from os import path, listdir
from glob import glob
from image_handler import crop, extract_intervals
from pandas import DataFrame, read_excel, concat
from PIL import Image
from typing import Generator
from pathlib import Path
from multiprocessing import Pool
from typing import Iterator
from itertools import repeat


processed: list = []


def get_images(base_dir: str, extension: str = ".tif"):
    """A function to get all images from the base directory

    Args:
        base_dir (str): the base directory
        extension (str): the file extension. Defaults to '.tif'
    """
    return [file for file in glob(f"{base_dir}/*{extension}")]


def write_image(crop_function: Generator, directory: str) -> Iterator:
    """
    A function that consumes a crop generator function and writes it's result to disk

    Args:
        crop_function (Generator): the crop function generator
        directory (str): the directory to save in

    Returns:
        tuple: [description]
    """
    for cropped, normalised_coord, file_name in crop_function:
        cropped.save(path.join(directory, file_name))
        cropped.close()
        yield normalised_coord, file_name


def process_image(
    image: Image, image_size: tuple, directory: str, n_process: int, seal_csv: str
):
    cropped_image_loc = DataFrame(
        {"tiff_file": [], "layer_name": [], "x_pixel": [], "y_pixel": []}
    )
    original_image_name = path.basename(image)
    with Image.open(image) as image:
        for location_frame, _ in write_image(
            crop(
                image,
                extract_intervals(image.size, image_size),
                locations[locations["tiff_file"] == original_image_name],
                filename=f"{path.splitext(original_image_name)[0]}_{image_size[0]}_{image_size[1]}",
            ),
            directory,
        ):
            location_frame["original_image"] = location_frame["tiff_file"].apply(
                lambda _: original_image_name
            )
            cropped_image_loc = concat([cropped_image_loc, location_frame])
    processed.append(original_image_name)
    print(f"Processed {len(processed)} from {n_process}")
    cropped_image_loc = generate_bbox(cropped_image_loc)
    cropped_image_loc.to_csv(seal_csv, mode="a", header=None)
    return cropped_image_loc


def generate_bbox(location_dataset: DataFrame, box_size: tuple = (60, 60)) -> DataFrame:
    locations = location_dataset.copy()
    locations["x_pixel"] = locations["x_pixel"].apply(lambda x: x - box_size[0] / 2)
    locations["x_pixel_end"] = locations["x_pixel"].apply(lambda x: x + box_size[0])
    locations["y_pixel"] = locations["y_pixel"].apply(lambda x: x - box_size[1] / 2)
    locations["y_pixel_end"] = locations["y_pixel"].apply(lambda x: x + box_size[1])
    return locations


def generate_subset(
    images: list, image_size: tuple, locations: DataFrame, directory: str
):
    image_records = []
    cropped_image_loc = DataFrame(
        columns=[
            "tiff_file",
            "layer_name",
            "original_image",
            "x_pixel",
            "y_pixel",
            "x_pixel_end",
            "y_pixel_end",
        ]
    )
    seal_csv = path.join(directory, "location.csv")
    cropped_image_loc.to_csv(seal_csv, header=True)
    with Pool(6) as p:
        seal_locations = p.starmap(
            process_image,
            zip(
                images,
                repeat(image_size),
                repeat(directory),
                repeat(len(images)),
                repeat(seal_csv),
            ),
            chunksize=3,
        )
    new_locations = concat(seal_locations)
    new_locations = generate_bbox(new_locations)
    new_locations.to_csv(path.join(directory, "final_location.csv"))
    pass


def main(
    original_dir,
    locations,
    output_dir,
    extension=".tif",
    base_image_size=416,
    max_size=416,
    step=2,
):
    """The main entry point that generates chunks of images from large images

    Args:
        original_dir (str): The original tiff directory
        locations (str): The location of each seal in the image in the original directory
        output_dir (str): The output directory
        output_dir (str): The extension
        base_image_size (int, optional): The smallest image size to generate. Defaults to 416.
        max_size (int, optional): The maximum image size to generate. Defaults to 10000.
        step (int, optional): The image size interval step. Defaults to 2.
    """
    # TODO read locations file
    # TODO: iterate through steps
    Image.MAX_IMAGE_PIXELS = 99999999999999999999
    image_paths = get_images(original_dir)
    final_output_path = path.join(output_dir, str(base_image_size))
    Path(final_output_path).mkdir(parents=True, exist_ok=True)
    generate_subset(
        image_paths,
        (base_image_size, base_image_size),
        locations,
        path.join(output_dir, str(base_image_size)),
    )


if __name__ == "__main__":
    locations = read_excel(
        "/home/md273/CS5099-working-copy/data/pixel_coord.xlsx",
        sheet_name="PixelCoordinates",
    )[["tiff_file", "layer_name", "x_pixel", "y_pixel"]]
    main("/data2/seals/TIFFs", locations, "/data2/seals/extracted")
    pass
