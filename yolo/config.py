from yolo import darknet_backbone
from yolo.models import dense_darknet, yolo_3, yolo_3_dual_scale, yolo_anchor_masks, yolo_dense_2_layer_anchor_masks, \
    yolo_3_single_scale, yolo_dense_1_layer_anchor_masks

backbones = {
    'original': darknet_backbone,
    'dense': dense_darknet
}

heads = {
    'yolo3': yolo_3,
    'yolo3_dense_2': yolo_3_dual_scale,
    'yolo3_dense_1': yolo_3_single_scale
}

anchor_size = {
    'yolo3': 9,
    'yolo3_dense_2': 6,
    'yolo3_dense_1': 6
}

down_sampling_factor = {
    'original': 32,
    'dense': 16
}

masks = {
    'yolo3': yolo_anchor_masks,
    'yolo3_dense_2': yolo_dense_2_layer_anchor_masks,
    'yolo3_dense_1': yolo_dense_1_layer_anchor_masks
}