import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from pandas import read_excel, DataFrame, notnull
from math import sqrt
from numpy import all
from os import path

SEAL_BOX_SIZE = 80


class SealDataset(tfds.core.GeneratorBasedBuilder):
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=(
                "This is the dataset for xxx. It contains yyy. The "
                "images are kept at their original dimensions."
            ),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict(
                {
                    "image_description": tfds.features.Text(),
                    "image": tfds.features.Image(),
                    # Here, labels can be of 5 distinct values.
                    "label": tfds.features.ClassLabel(num_classes=5),
                    "area": tf.int64,
                    "bbox": tfds.features.BBoxFeature(),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("image", "bbox"),
            # Homepage of the dataset for documentation
            # homepage="https://dataset-homepage.org",
            # Bibtex citation for the dataset
            citation=r"""@article{my-awesome-dataset-2020,
                                author = {Smith, John},"}""",
        )

    VERSION = tfds.core.Version("0.0.1")

    def _generate_examples(self, image_path: str, locations: DataFrame = None):
        if locations is None:
            locations = self.get_seal_locations(
                "data_files/pixel_coord.xlsx", sheet_name="PixelCoordinates", bbox_size=80
            )
        for row in locations.iterrows():
            yield {
                "image_description": row["tiff_file"],
                "image": path.join(image_path, row["tiff_file"]),
                "area": SEAL_BOX_SIZE * 2,
                "label": row["layer_name"],
                "bbox": row["bbox_1"],
            }
        pass

    def _split_generators(self):
        # return [
        #     tfds.core.SplitGenerator(
        #         name=tfds.Split.TRAIN,
        #         gen_kwargs={
        #             "images_dir_path": path.join("TIFFs"),
        #             "locations": path.join("pixel_coord.xlsx"),
        #         },
        #     ),
        #     tfds.core.SplitGenerator(
        #         name=tfds.Split.TEST,
        #         gen_kwargs={
        #             "images_dir_path": path.join("TIFFs"),
        #             "locations": path.join("pixel_coord.xlsx"),
        #         },
        #     ),
        # ]
        pass

    def get_seal_locations(
        self, seal_loc_file: str, sheet_name="PixelCoordinates", bbox_size=80
    ) -> DataFrame:
        radius = SEAL_BOX_SIZE / 2
        locations = read_excel(seal_loc_file, sheet_name=sheet_name)[
            ["tiff_file", "layer_name", "x_pixel", "y_pixel"]
        ]
        locations["bbox_1"] = locations.apply(
            lambda x: [x.x_pixel - radius, x.y_pixel - radius] if notnull(x.pixel) else []
        )
        locations["bbox_2"] = locations.apply(
            lambda x: [x.x_pixel + radius, x.y_pixel + radius] if notnull(x.pixel) else []
        )
        return locations


dataset = SealDataset()
locations = dataset.get_seal_locations("/home/md273/CS5099-working-copy/data/pixel_coord.xlsx")
dataset._generate_examples("/home/md273/CS5099-working-copy/TIFFs", locations)
