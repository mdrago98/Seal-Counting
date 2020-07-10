import json
import os


def set_voc_tags(
    tree,
    folder,
    filename,
    path,
    size,
    width,
    height,
    depth,
    obj,
    obj_name,
    box,
    x0,
    y0,
    x1,
    y1,
    conf_file='voc_conf.json',
    indent=4,
    sort_keys=False,
):
    """
    Create/modify json voc annotation tags.
    Args:
        tree: xml tree tag.
        folder: Image folder tag.
        filename: Image file tag.
        path: Path to image tag.
        size: Image size tag.
        width: Image width tag.
        height: Image height tag.
        depth: Image depth tag.
        obj: Object tag.
        obj_name: Object name tag.
        box: Bounding box tag.
        x0: Start x coordinate tag.
        y0: Start y coordinate tag.
        x1: End x coordinate tag.
        y1: End y coordinate tag.
        conf_file: Configuration file name.
        indent: json output indent.
        sort_keys: Sort json output keys.

    Returns:
        None.
    """
    if conf_file in os.listdir('.'):
        os.remove(os.path.join(os.getcwd(), conf_file))
    conf = {
        'Tree': {
            'Tree Tag': tree,
            'Folder': folder,
            'Filename': filename,
            'Path': path,
        },
        'Size': {
            'Size Tag': size,
            'Width': width,
            'Height': height,
            'Depth': depth,
        },
        'Object': {
            'Object Tag': obj,
            'Object Name': obj_name,
            'Object Box': {
                'Object Box Tag': box,
                'X0': x0,
                'Y0': y0,
                'X1': x1,
                'Y1': y1,
            },
        },
    }

    with open(conf_file, 'w') as conf_out:
        json.dump(conf, conf_out, indent=indent, sort_keys=sort_keys)


if __name__ == '__main__':
    tree_tag = 'annotation'
    folder_tag = 'folder'
    file_tag = 'filename'
    path_tag = 'path'
    size_tag = 'size'
    width_tag = 'width'
    height_tag = 'height'
    depth_tag = 'depth'
    object_tag = 'object'
    object_name = 'name'
    box_tag = 'bndbox'
    x_min_tag = 'xmin'
    y_min_tag = 'ymin'
    x_max_tag = 'xmax'
    y_max_tag = 'ymax'
    set_voc_tags(
        tree_tag,
        folder_tag,
        file_tag,
        path_tag,
        size_tag,
        width_tag,
        height_tag,
        depth_tag,
        object_tag,
        object_name,
        box_tag,
        x_min_tag,
        y_min_tag,
        x_max_tag,
        y_max_tag,
    )
