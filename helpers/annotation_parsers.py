from xml.etree import ElementTree
from time import perf_counter
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from helpers.visual_tools import visualization_wrapper
from helpers.utils import ratios_to_coordinates, default_logger


def get_tree_item(parent, tag, file_path, find_all=False):
    """
    Get item from xml tree element.
    Args:
        parent: Parent in xml element tree
        tag: tag to look for.
        file_path: Current xml file being handled.
        find_all: If True, all elements found will be returned.

    Returns:
        Tag item.
    """
    target = parent.find(tag)
    if find_all:
        target = parent.findall(tag)
    if target is None:
        raise ValueError(f'Could not find {tag} in {file_path}')
    return target


def parse_voc_file(file_path, voc_conf):
    """
    Parse voc annotation from xml file.
    Args:
        file_path: Path to xml file.
        voc_conf: voc configuration file.

    Returns:
        A list of image annotations.
    """
    assert os.path.exists(file_path)
    image_data = []
    with open(voc_conf) as json_data:
        tags = json.load(json_data)
    tree = ElementTree.parse(file_path)
    image_path = get_tree_item(tree, tags['Tree']['Path'], file_path).text
    size_item = get_tree_item(tree, tags['Size']['Size Tag'], file_path)
    image_width = get_tree_item(
        size_item, tags['Size']['Width'], file_path
    ).text
    image_height = get_tree_item(
        size_item, tags['Size']['Height'], file_path
    ).text
    for item in get_tree_item(
        tree, tags['Object']['Object Tag'], file_path, True
    ):
        name = get_tree_item(
            item, tags['Object']['Object Name'], file_path
        ).text
        box_item = get_tree_item(
            item, tags['Object']['Object Box']['Object Box Tag'], file_path
        )
        x0 = get_tree_item(
            box_item, tags['Object']['Object Box']['X0'], file_path
        ).text
        y0 = get_tree_item(
            box_item, tags['Object']['Object Box']['Y0'], file_path
        ).text
        x1 = get_tree_item(
            box_item, tags['Object']['Object Box']['X1'], file_path
        ).text
        y1 = get_tree_item(
            box_item, tags['Object']['Object Box']['Y1'], file_path
        ).text
        image_data.append(
            [image_path, name, image_width, image_height, x0, y0, x1, y1]
        )
    return image_data


def adjust_frame(frame, cache_file=None):
    """
    Add relative width, relative height and object ids to annotation pandas DataFrame.
    Args:
        frame: pandas DataFrame containing coordinates instead of relative labels.
        cache_file: cache_file: csv file name containing current session labels.

    Returns:
        Frame with the new columns
    """
    object_id = 1
    for item in frame.columns[2:]:
        frame[item] = frame[item].astype(float).astype(int)
    frame['Relative Width'] = (frame['X_max'] - frame['X_min']) / frame[
        'Image Width'
    ]
    frame['Relative Height'] = (frame['Y_max'] - frame['Y_min']) / frame[
        'Image Height'
    ]
    for object_name in list(frame['Object Name'].drop_duplicates()):
        frame.loc[frame['Object Name'] == object_name, 'Object ID'] = object_id
        object_id += 1
    if cache_file:
        frame.to_csv(
            os.path.join('../yolo', 'Output', 'Data', cache_file), index=False
        )
    print(f'Parsed labels:\n{frame["Object Name"].value_counts()}')
    return frame


@visualization_wrapper
def parse_voc_folder(folder_path, voc_conf):
    """
    Parse a folder containing voc xml annotation files.
    Args:
        folder_path: Folder containing voc xml annotation files.
        voc_conf: Path to voc json configuration file.

    Returns:
        pandas DataFrame with the annotations.
    """
    assert os.path.exists(folder_path)
    cache_path = os.path.join('../yolo', 'Output', 'Data', 'parsed_from_xml.csv')
    if os.path.exists(cache_path):
        frame = pd.read_csv(cache_path)
        print(
            f'Labels retrieved from cache:'
            f'\n{frame["Object Name"].value_counts()}'
        )
        return frame
    image_data = []
    frame_columns = [
        'Image Path',
        'Object Name',
        'Image Width',
        'Image Height',
        'X_min',
        'Y_min',
        'X_max',
        'Y_max',
    ]
    xml_files = [
        file_name
        for file_name in os.listdir(folder_path)
        if file_name.endswith('.xml')
    ]
    for file_name in xml_files:
        annotation_path = os.path.join(folder_path, file_name)
        image_labels = parse_voc_file(annotation_path, voc_conf)
        image_data.extend(image_labels)
    frame = pd.DataFrame(image_data, columns=frame_columns)
    classes = frame['Object Name'].drop_duplicates()
    default_logger.info(f'Read {len(xml_files)} xml files')
    default_logger.info(
        f'Received {len(frame)} labels containing ' f'{len(classes)} classes'
    )
    if frame.empty:
        raise ValueError(
            f'No labels were found in {os.path.abspath(folder_path)}'
        )
    frame = adjust_frame(frame, 'parsed_from_xml.csv')
    return frame


@visualization_wrapper
def adjust_non_voc_csv(csv_file, image_path, image_width, image_height):
    """
    Read relative data and return adjusted frame accordingly.
    Args:
        csv_file: .csv file containing the following columns:
        [Image, Object Name, Object Index, bx, by, bw, bh]
        image_path: Path prefix to be added.
        image_width: image width.
        image_height: image height
    Returns:
        pandas DataFrame with the following columns:
        ['Image Path', 'Object Name', 'Image Width', 'Image Height', 'X_min',
       'Y_min', 'X_max', 'Y_max', 'Relative Width', 'Relative Height',
       'Object ID']
    """
    image_path = Path(image_path).absolute().resolve()
    coordinates = []
    old_frame = pd.read_csv(csv_file)
    new_frame = pd.DataFrame()
    new_frame['Image Path'] = old_frame['Image'].apply(
        lambda item: os.path.join(image_path, item)
    )
    new_frame['Object Name'] = old_frame['Object Name']
    new_frame['Image Width'] = image_width
    new_frame['Image Height'] = image_height
    new_frame['Relative Width'] = old_frame['bw']
    new_frame['Relative Height'] = old_frame['bh']
    new_frame['Object ID'] = old_frame['Object Index'] + 1
    for index, row in old_frame.iterrows():
        image, object_name, object_index, bx, by, bw, bh = row
        co = ratios_to_coordinates(bx, by, bw, bh, image_width, image_height)
        coordinates.append(co)
    (
        new_frame['X_min'],
        new_frame['Y_min'],
        new_frame['X_max'],
        new_frame['Y_max'],
    ) = np.array(coordinates).T
    new_frame[['X_min', 'Y_min', 'X_max', 'Y_max']] = new_frame[
        ['X_min', 'Y_min', 'X_max', 'Y_max']
    ].astype('int64')
    print(f'Parsed labels:\n{new_frame["Object Name"].value_counts()}')
    classes = new_frame['Object Name'].drop_duplicates()
    default_logger.info(
        f'Adjustment from existing received {len(new_frame)} labels containing '
        f'{len(classes)} classes'
    )
    default_logger.info(f'Added prefix to images: {image_path}')
    return new_frame[
        [
            'Image Path',
            'Object Name',
            'Image Width',
            'Image Height',
            'X_min',
            'Y_min',
            'X_max',
            'Y_max',
            'Relative Width',
            'Relative Height',
            'Object ID',
        ]
    ]
