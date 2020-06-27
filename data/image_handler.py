from PIL import Image
from pandas import read_excel, DataFrame, Series
from collections import namedtuple
from os import path

BBox = namedtuple('BBox', 'x_min y_min x_max y_max')
Point = namedtuple('Point', 'x y')


def get_bbox(x_dims: tuple, y_dims: tuple):
    return BBox(x_dims[0], y_dims[0], x_dims[1], x_dims[1])


def is_in_bounding_box(bbox: BBox, point: Series) -> bool:
    """
    A function that determines if the point lies within a bounding box
    :param bbox: the named tuple BBox representing the bounding box
    :param point: the series object of (x, y) coordinates of the point
    :return: true IFF the point lies in the box
    """
    return True if bbox.x_min <= point[0] <= bbox.x_max and bbox.y_min <= point[1] <= bbox.y_max else False


def normalise_coordinates(box: BBox, coordinates: DataFrame):
    """
    A function to normalise the coordinates
    :param box:
    :param coordinates:
    :return:
    """
    coordinates = coordinates.copy()
    coordinates.x_pixel = coordinates.x_pixel.apply(lambda x: x - box.x_min)
    coordinates.y_pixel = coordinates.y_pixel.apply(lambda x: x - box.y_min)
    return coordinates


def extract_int_through_nsplits(size, split):
    """
    A function to get the pixel intervals to crop the image.
    :param size: the tuple of (length and height) representing the original image size
    :param split: the split
    :return:
    """
    og_width, og_height = size[0], size[1]
    width_interval, height_interval = int(og_width / split[0]), int(og_height / split[1])
    result_widths = [(i, i + width_interval) for i in range(0, og_width, width_interval)]
    result_heights = [(i, i + height_interval) for i in range(0, og_height, height_interval)]
    return result_widths, result_heights


def extract_intervals(size: tuple, split_size: tuple):
    """
    A function to generate pixel intervals
    :param size: the tuple representing (length, height)
    :param split_size: the split size
    :return:
    """
    og_width, og_height = size[0], size[1]
    if og_width < split_size[0] or og_height < split_size[1]:
        raise ValueError('The split size is larger than the image')
    result_widths = [(i, i + split_size[0]) for i in range(0, og_width, split_size[0])
                     if (split_size[0] + i) <= og_width]
    result_heights = [(i, i + split_size[1]) for i in range(0, og_height, split_size[1])
                      if (split_size[1] + i) <= og_height]
    return result_widths, result_heights


def crop(original_image, intervals: tuple, seal_loc: DataFrame) -> list:
    """
    A function to crop an image and update the seal location.
    :param intervals: the tuple representing the x, y image intervals to split at
    :param original_image: the image
    :param seal_loc: a DataFrame representing the seal locations
    :return: a list of seal images and a DataFrame of new locations
    """
    height_intervals, width_intervals = intervals[0], intervals[1]

    for x_size, y_size in zip(width_intervals, height_intervals):
        box = get_bbox(x_size, y_size)
        loc_existence = seal_loc[['x_pixel', 'y_pixel']].dropna().apply(lambda x: is_in_bounding_box(box, x), axis=1)
        cropped = original_image.crop(box)
        yield cropped, normalise_coordinates(box, seal_loc[loc_existence]), seal_loc[loc_existence].size > 0


if __name__ == '__main__':
    locations = read_excel('pixel_coord.xlsx', sheet_name='PixelCoordinates')[['tiff_file', 'layer_name', 'x_pixel',
                                                                               'y_pixel']]
    filename = '../data/StitchMICE_FoFcr16_2_1024_CP_FINAL.tif'
    im = Image.open(filename)
    im = im.crop((0, 0, 40, 40))
    im.load()
    loc = {'tiff_file': ['test', 'test'],
           'layer_name': ['whitecoat', 'harbour'],
           'x_pixel': [10, 23],
           'y_pixel': [10, 24]
           }

    next(crop(im, extract_intervals(im.size, (412, 412)),
              DataFrame(loc)))
