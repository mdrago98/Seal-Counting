import os
import cv2
import imagesize
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug as ia
from pathlib import Path
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers.utils import ratios_to_coordinates, timer, default_logger
from helpers.annotation_parsers import adjust_non_voc_csv


class DataAugment:
    """
    A tool for augmenting image data sets with bounding box support.
    """

    def __init__(
        self,
        labels_file,
        augmentation_map,
        workers=32,
        converted_coordinates_file=None,
        image_folder=None,
    ):
        """
        Initialize augmentation session.
        Args:
            labels_file: cvv file containing relative image labels
            augmentation_map: A structured dictionary containing categorized augmentation
            sequences.
            workers: Parallel threads.
            converted_coordinates_file: csv file containing converted from relative
            to coordinates.
            image_folder: Folder containing images other than Data/Photos/
        """
        assert all([ia, iaa, iap])
        self.labels_file = labels_file
        self.mapping = pd.read_csv(labels_file)
        self.image_folder = (
            Path(os.path.join('../yolo', 'Data', 'Photos')).absolute().resolve()
        )
        if image_folder:
            self.image_folder = Path(image_folder).absolute().resolve()
        self.image_paths = [
            Path(os.path.join(self.image_folder, image)).absolute().resolve()
            for image in os.listdir(self.image_folder)
            if not image.startswith('.')
        ]
        self.image_paths_copy = self.image_paths.copy()
        if not self.image_paths:
            default_logger.error(
                f'Augmentation aborted: no photos found in {self.image_folder}'
            )
            raise ValueError(f'No photos given')
        self.image_width, self.image_height = imagesize.get(
            self.image_paths[0]
        )
        self.converted_coordinates = (
            pd.read_csv(converted_coordinates_file)
            if converted_coordinates_file
            else self.relative_to_coordinates()
        )
        self.converted_groups = self.converted_coordinates.groupby('image')
        self.augmentation_data = []
        self.augmentation_sequences = []
        self.augmentation_map = augmentation_map
        self.workers = workers
        self.augmented_images = 0
        self.total_images = len(self.image_paths)
        self.session_id = np.random.randint(10 ** 6, (10 ** 7))

    def create_sequences(self, sequences):
        """
        Create sequences for imgaug.augmenters.Sequential().
        Args:
            sequences: A list of dictionaries with each dictionary
            containing the following:
            -sequence_group: str, one of self.augmentation_map keys including:
                ['meta', 'arithmetic', 'artistic', 'blend', 'gaussian_blur', 'color',
                'contrast', 'convolution', 'edges', 'flip', 'geometric', 'corrupt_like',
                'pi_like', 'pooling', 'segmentation', 'size']
            -no: augmentation number ('no' key in self.augmentation_map > chosen sequence_group)
                Example:
                    sequences = (
                    [[{'sequence_group': 'meta', 'no': 5},
                      {'sequence_group': 'arithmetic', 'no': 3}],
                     [{'sequence_group': 'arithmetic', 'no': 2}]]
                    )
        Returns:
            The list of augmentation sequences that will be applied over images.
        """
        total_target = (len(sequences) * len(self.image_paths)) + len(
            self.image_paths
        )
        default_logger.info(f'Total images(old + augmented): {total_target}')
        for group in sequences:
            some_ofs = [
                self.augmentation_map[item['sequence_group']][item['no'] - 1]
                for item in group[0]
            ]
            one_ofs = [
                self.augmentation_map[item['sequence_group']][item['no'] - 1]
                for item in group[1]
            ]
            some_of_aug = [item['augmentation'] for item in some_ofs]
            one_of_aug = [item['augmentation'] for item in one_ofs]
            some_of_seq = iaa.SomeOf(
                (0, 5), [eval(item) for item in some_of_aug]
            )
            one_of_seq = iaa.OneOf([eval(item) for item in one_of_aug])
            self.augmentation_sequences.append(
                iaa.Sequential([some_of_seq, one_of_seq], random_order=True)
            )
        return self.augmentation_sequences

    @staticmethod
    def load_image(image_path, new_size=None):
        """
        Load image.
        Args:
            image_path: Path to image to load.
            new_size: new image dimensions(tuple).

        Returns:
            numpy array(image), image_path
        """
        assert os.path.exists(image_path), f'{image_path} does not exist'
        image = cv2.imread(image_path)
        if new_size:
            return cv2.resize(image, new_size)
        return image, image_path

    def calculate_ratios(self, x1, y1, x2, y2):
        """
        Calculate relative object ratios in the labeled image.
        Args:
            x1: Start x coordinate.
            y1: Start y coordinate.
            x2: End x coordinate.
            y2: End y coordinate.

        Return:
            bx: Relative center x coordinate.
            by: Relative center y coordinate.
            bw: Relative box width.
            bh: Relative box height.
        """
        box_width = abs(x2 - x1)
        box_height = abs(y2 - y1)
        bx = 1 - (
            (self.image_width - min(x1, x2) + (box_width / 2))
            / self.image_width
        )
        by = 1 - (
            (self.image_height - min(y1, y2) + (box_height / 2))
            / self.image_height
        )
        bw = box_width / self.image_width
        bh = box_height / self.image_height
        return bx, by, bw, bh

    def relative_to_coordinates(self, out_file=None):
        """
        Convert relative coordinates in self.mapping
        to coordinates.
        Args:
            out_file: path to new converted csv.

        Returns:
            pandas DataFrame with the new coordinates.
        """
        items_to_save = []
        for index, data in self.mapping.iterrows():
            image_name, object_name, object_index, bx, by, bw, bh = data
            x1, y1, x2, y2 = ratios_to_coordinates(
                bx, by, bw, bh, self.image_width, self.image_height
            )
            items_to_save.append(
                [
                    image_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    object_name,
                    object_index,
                    bx,
                    by,
                    bw,
                    bh,
                ]
            )
        new_data = pd.DataFrame(
            items_to_save,
            columns=[
                'image',
                'x1',
                'y1',
                'x2',
                'y2',
                'object_type',
                'object_id',
                'bx',
                'by',
                'bw',
                'bh',
            ],
        )
        new_data[['x1', 'y1', 'x2', 'y2']] = new_data[
            ['x1', 'y1', 'x2', 'y2']
        ].astype('int64')
        if out_file:
            new_data.to_csv(out_file, index=False)
        default_logger.info(
            f'Converted labels in {self.labels_file} to coordinates'
        )
        return new_data

    def get_image_data(self, image_path):
        """
        Get image data including bounding boxes and object names.
        Args:
            image_path: Path to image.

        Returns:
            pandas DataFrame with full data for image.
        """
        image_name = os.path.basename(image_path)
        return self.converted_groups.get_group(image_name)

    def get_bounding_boxes_over_image(self, image_path):
        """
        Get BoundingBoxesOnImage object.
        Args:
            image_path: single path.

        Returns:
            BoundingBoxesOnImage, frame_before.
        """
        boxes = []
        frame_before = self.get_image_data(image_path)
        for item in frame_before[['x1', 'y1', 'x2', 'y2']].values:
            boxes.append(BoundingBox(*item))
        return (
            BoundingBoxesOnImage(
                boxes, shape=(self.image_height, self.image_width)
            ),
            frame_before,
        )

    def load_batch(self, new_size, batch_size):
        """
        Load a batch of images in memory for augmentation.
        Args:
            new_size: new image size(tuple).
            batch_size: Number of images to load for augmentation.

        Returns:
            numpy array of shape (batch_size, height, width, channels)
        """
        batch = [
            f'{self.image_paths_copy.pop()!s}'
            for _ in range(batch_size)
            if self.image_paths_copy
        ]
        loaded = []
        paths = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_images = {
                executor.submit(
                    self.load_image, image_path, new_size
                ): image_path
                for image_path in batch
            }
            for future_image in as_completed(future_images):
                image, image_path = future_image.result()
                loaded.append(image)
                paths.append(image_path)
        return np.array(loaded), paths

    def update_data(
        self, bbs_aug, frame_before, image_aug, new_name, new_path
    ):
        """
        Update new bounding boxes data and save augmented image.
        Args:
            bbs_aug: Augmented bounding boxes
            frame_before: pandas DataFrame containing pre-augmentation data.
            image_aug: Augmented image as numpy nd array.
            new_name: new image name to save image.
            new_path: path to save the image.

        Returns:
            None
        """
        frame_after = pd.DataFrame(
            bbs_aug.bounding_boxes, columns=['x1y1', 'x2y2']
        )
        if (
            frame_after.empty
        ):  # some post-augmentation photos do not contain bounding boxes
            default_logger.warning(
                f'\nskipping image: {new_name}: no bounding boxes after '
                f'augmentation'
            )
            return
        frame_after = pd.DataFrame(
            np.hstack(
                (frame_after['x1y1'].tolist(), frame_after['x2y2'].tolist())
            ),
            columns=['x1', 'y1', 'x2', 'y2'],
        ).astype('int64')
        frame_after['object_type'] = frame_before['object_type'].values
        frame_after['object_id'] = frame_before['object_id'].values
        frame_after['image'] = new_name
        for index, row in frame_after.iterrows():
            x1, y1, x2, y2, object_type, object_id, image_name = row
            bx, by, bw, bh = self.calculate_ratios(x1, y1, x2, y2)
            self.augmentation_data.append(
                [image_name, object_type, object_id, bx, by, bw, bh]
            )
        cv2.imwrite(new_path, image_aug)

    def augment_image(self, image, image_path):
        """
        Perform augmentation and save image.
        Args:
            image: image to augment.
            image_path: Path to image.

        Returns:
            None
        """
        current_sequence = 1
        for augmentation_sequence in self.augmentation_sequences:
            new_image_name = (
                f'aug-{self.session_id}-sequence-{current_sequence}'
                f'-{os.path.basename(image_path)}'
            )
            new_image_path = os.path.join(self.image_folder, new_image_name)
            bbs, frame_before = self.get_bounding_boxes_over_image(image_path)
            augmented_image, augmented_boxes = augmentation_sequence(
                image=image, bounding_boxes=bbs
            )
            self.update_data(
                augmented_boxes,
                frame_before,
                augmented_image,
                new_image_name,
                new_image_path,
            )
            current_sequence += 1
            self.augmented_images += 1
        current = os.path.basename(image_path)
        completed = (
            f'{self.augmented_images}/'
            f'{self.total_images * len(self.augmentation_sequences)}'
        )
        percent = (
            self.augmented_images
            / (self.total_images * len(self.augmentation_sequences))
            * 100
        )
        print(
            f'\raugmenting {current}\t{completed}\t{percent}% completed',
            end='',
        )

    @timer(default_logger)
    def augment_photos_folder(self, batch_size=64, new_size=None):
        """
        Augment photos in Data/Photos/
        Args:
            batch_size: Size of each augmentation batch.
            new_size: tuple, new image size.

        Returns:
            None
        """
        default_logger.info(
            f'Started augmentation with {self.workers} workers'
        )
        default_logger.info(f'Total images to augment: {self.total_images}')
        default_logger.info(f'Session assigned id: {self.session_id}')
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while self.image_paths_copy:
                current_batch, current_paths = self.load_batch(
                    new_size, batch_size
                )
                future_augmentations = {
                    executor.submit(self.augment_image, image, path): path
                    for image, path in zip(current_batch, current_paths)
                }
                for future_augmented in as_completed(future_augmentations):
                    future_augmented.result()
        default_logger.info(f'Augmentation completed')
        augmentation_frame = pd.DataFrame(
            self.augmentation_data, columns=self.mapping.columns
        )
        saving_path = os.path.join(
                '../yolo', 'Output', 'Data', f'augmented_data_plus_original.csv'
        )
        combined = pd.concat([self.mapping, augmentation_frame])
        for item in ['bx', 'by', 'bw', 'bh']:
            combined = combined.drop(combined[combined[item] > 1].index)
        combined.to_csv(saving_path, index=False)
        default_logger.info(f'Saved old + augmented labels to {saving_path}')
        adjusted_combined = adjust_non_voc_csv(
            saving_path, self.image_folder, self.image_width, self.image_height
        )
        adjusted_saving_path = saving_path.replace('augmented', 'adjusted_aug')
        adjusted_combined.to_csv(adjusted_saving_path, index=False)
        default_logger.info(
            f'Saved old + augmented (adjusted) labels to {adjusted_saving_path}'
        )
        return adjusted_combined

    @staticmethod
    def display_windows(out, move_after):
        """
        Display image before/after augmentation.
        Args:
            out: A list containing images to display
            move_after: tuple(x, y) coordinates to move the 'after' window
            display window.

        Returns:
            None
        """
        for bef, aft in out:
            cv2.imshow('before', bef)
            cv2.imshow('after', aft)
            cv2.moveWindow('after', *move_after)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def display_with_bbs(self, augment, images, image_paths, out, move_after):
        """
        Display augmentation results(before/after) with bounding boxes.
        Args:
            augment: imgaug augmentation sequence.
            images: numpy array of images.
            image_paths: A list of paths of images.
            out: A list to hold display results.
            move_after: tuple(x, y) coordinates to move the 'after' window.

        Returns:
            None
        """
        for image, image_path in zip(images, image_paths):
            bbs, frame_before = self.get_bounding_boxes_over_image(image_path)
            image_aug, bbs_aug = augment(image=image, bounding_boxes=bbs)
            image_before = bbs.draw_on_image(image, size=2)
            image_after = bbs_aug.draw_on_image(
                image_aug, size=2, color=[0, 0, 255]
            )
            out.append([image_before, image_after])
        self.display_windows(out, move_after)
        out.clear()

    def preview_augmentations(
        self,
        tensor_size,
        sequence_group,
        display_boxes=False,
        move_after=(600, 0),
    ):
        """
        Preview all augmentation options on images in self.paths
         in selected sequence_group
        Args:
            tensor_size: The number of images to display from self.paths
            sequence_group: str, one of the following:
                - meta
                - arithmetic
                - artistic
                - blend
                - gaussian_blur
                - color
                - contrast
                - convolution
                - edges
                - flip
                - geometric
                - corrupt_like
                - pi_like
                - pooling
                - segmentation
                - size
            display_boxes: Preview augmentations with bounding boxes before/after(if any)
            move_after: tuple(x, y) coordinates to move the 'after' window.

        Returns:
            None
        """
        images = [
            cv2.imread(f'{image}') for image in self.image_paths[:tensor_size]
        ]
        image_tensor = np.array(images)
        to_display = []
        for item in self.augmentation_map[sequence_group]:
            print(item['no'], item['augmentation'])
            try:
                current_augment = eval(item['augmentation'])
                if not display_boxes:
                    to_display = current_augment(images=image_tensor)
                    self.display_windows(
                        zip(image_tensor, to_display), move_after
                    )
                if display_boxes:
                    self.display_with_bbs(
                        current_augment,
                        images,
                        self.image_paths[:tensor_size],
                        to_display,
                        move_after,
                    )
            except Exception as e:
                print(f'{e} for {item}')
