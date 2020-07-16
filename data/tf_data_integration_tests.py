import unittest
from .tf_data_integration import *
from .image_handler import get_bbox
from pandas import DataFrame
from numpy.testing import assert_array_equal
from numpy import all


class TestGetCroppingRegion(unittest.TestCase):
    test_case: DataFrame = DataFrame(
        {
            "tiff_file": {
                1: "StitchMICE_FoFcr16_2_1024_CP_FINAL.tif",
                2: "StitchMICE_FoFcr16_2_1024_CP_FINAL.tif",
            },
            "layer_name": {1: 0, 2: 0},
            "x_pixel": {1: 1067, 2: 279},
            "y_pixel": {1: 1289, 2: 1210},
            "image_width": {1: 3391, 2: 3391},
            "image_height": {1: 3191, 2: 3191},
        }
    )

    test_case_norm: DataFrame = DataFrame(
        {
            "tiff_file": {
                13: "StitchMICE_FoFcr16_3_1102_CP_FINAL.tif",
                14: "StitchMICE_FoFcr16_3_1102_CP_FINAL.tif",
                15: "StitchMICE_FoFcr16_3_1102_CP_FINAL.tif",
            },
            "layer_name": {13: 0, 14: 0, 15: 0},
            "x_pixel": {13: 1385, 14: 1409, 15: 1209},
            "y_pixel": {13: 1322, 14: 1371, 15: 1423},
            "image_width": {13: 3479, 14: 3479, 15: 3479},
            "image_height": {13: 3252, 14: 3252, 15: 3252},
        }
    )

    def test_crops_are_in_bounds(self):
        results = get_seal_cropping_region(self.test_case)
        assert all(results["y_min"] <= results["image_height"])
        assert all(results["y_max"] <= results["image_height"])
        assert all(results["x_min"] <= results["image_width"])
        assert all(results["x_max"] <= results["image_width"])

    def test_pixel_normalisation(self):
        crops = get_seal_cropping_region(self.test_case)
        result = normalise_coordinates(get_bbox((859, 1257), (1081, 1497)), crops)
        assert result.iloc[0]["y_pixel"] == 208
        assert result.iloc[0]["x_pixel"] == 208

    def test_pixel_normalisation_multiple(self):
        crops = get_seal_cropping_region(self.test_case_norm)
        bbox = BBox(x_min=1201, y_min=1163, x_max=1617, y_max=1579)
        result = normalise_coordinates(bbox, crops)
        assert_array_equal([159, 208, 260], result["y_pixel"].to_numpy())
        assert_array_equal([184, 208, 8], result["x_pixel"].to_numpy())
