import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys


from concurrent.futures import ThreadPoolExecutor, as_completed
from models import BaseModel
from Helpers.dataset_handlers import read_tfr, get_feature_map
from utils import (
    transform_images,
    get_detection_data,
    default_logger,
    timer,
)
from utils.visual_utils import visualize_pr, visualize_evaluation_stats

sys.path.append('..')

class Evaluator(BaseModel):
    def __init__(
        self,
        input_shape,
        model_configuration,
        train_tf_record,
        valid_tf_record,
        classes_file,
        anchors=None,
        masks=None,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
    ):
        """
        Evaluate a trained model.
        Args:
            input_shape: input_shape: tuple, (n, n, c)
            model_configuration: Path to model configuration file.
            train_tf_record: Path to training TFRecord file.
            valid_tf_record: Path to validation TFRecord file.
            classes_file: File containing class names \n delimited.
            anchors: numpy array of (w, h) pairs.
            masks: numpy array of masks.
            max_boxes: Maximum boxes of the TFRecords provided.
            iou_threshold: Minimum overlap value.
            score_threshold: Minimum confidence for detection to count
                as true positive.
        """
        self.classes_file = classes_file
        self.class_names = [
            item.strip() for item in open(classes_file).readlines()
        ]
        super().__init__(
            input_shape,
            model_configuration,
            len(self.class_names),
            anchors,
            masks,
            max_boxes,
            iou_threshold,
            score_threshold,
        )
        self.train_tf_record = train_tf_record
        self.valid_tf_record = valid_tf_record
        self.train_dataset_size = sum(
            1 for _ in tf.data.TFRecordDataset(train_tf_record)
        )
        self.valid_dataset_size = sum(
            1 for _ in tf.data.TFRecordDataset(valid_tf_record)
        )
        self.dataset_size = self.train_dataset_size + self.valid_dataset_size
        self.predicted = 1

    def predict_image(self, image_data, features):
        """
        Make predictions on a single image from the TFRecord.
        Args:
            image_data: image as numpy array
            features: features of the TFRecord.

        Returns:
            pandas DataFrame with detection data.
        """
        image_path = bytes.decode(features['image_path'].numpy())
        image_name = os.path.basename(image_path)
        image = tf.expand_dims(image_data, 0)
        resized = transform_images(image, self.input_shape[0])
        outs = self.inference_model(resized)
        adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
        result = (
            get_detection_data(adjusted, image_name, outs, self.class_names),
            image_name,
        )
        return result

    @staticmethod
    def get_dataset_next(dataset):
        try:
            return next(dataset)
        except tf.errors.UnknownError as e:  # sometimes encountered when reading from google drive
            default_logger.error(
                f'Error occurred during reading from dataset\n{e}'
            )

    def predict_dataset(
        self, dataset, workers=16, split='train', batch_size=64
    ):
        """
        Predict entire dataset.
        Args:
            dataset: MapDataset object.
            workers: Parallel predictions.
            split: str representation of the dataset 'train' or 'valid'
            batch_size: Prediction batch size.

        Returns:
            pandas DataFrame with entire dataset predictions.
        """
        predictions = []
        sizes = {
            'train': self.train_dataset_size,
            'valid': self.valid_dataset_size,
        }
        size = sizes[split]
        current_prediction = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while current_prediction < size:
                current_batch = []
                for _ in range(min(batch_size, size - current_prediction)):
                    item = self.get_dataset_next(dataset)
                    if item is not None:
                        current_batch.append(item)
                future_predictions = {
                    executor.submit(
                        self.predict_image, img_data, features
                    ): features['image_path']
                    for img_data, labels, features in current_batch
                }
                for future_prediction in as_completed(future_predictions):
                    result, completed_image = future_prediction.result()
                    predictions.append(result)
                    completed = f'{self.predicted}/{self.dataset_size}'
                    percent = (self.predicted / self.dataset_size) * 100
                    print(
                        f'\rpredicting {completed_image} {completed}\t{percent}% completed',
                        end='',
                    )
                    self.predicted += 1
                    current_prediction += 1
        return pd.concat(predictions)

    @timer(default_logger)
    def make_predictions(
        self,
        trained_weights,
        merge=False,
        workers=16,
        shuffle_buffer=512,
        batch_size=64,
    ):
        """
        Make predictions on both training and validation data sets
            and save results as csv in Output folder.
        Args:
            trained_weights: Trained .tf weights or .weights file(in case self.classes = 80).
            merge: If True a single file will be saved for training
                and validation sets predictions combined.
            workers: Parallel predictions.
            shuffle_buffer: int, shuffle dataset buffer size.
            batch_size: Prediction batch size.

        Returns:
            1 combined pandas DataFrame for entire dataset predictions
                or 2 pandas DataFrame(s) for training and validation
                data sets respectively.
        """
        self.create_models()
        self.load_weights(trained_weights)
        features = get_feature_map()
        train_dataset = read_tfr(
            self.train_tf_record,
            self.classes_file,
            features,
            self.max_boxes,
            get_features=True,
        )
        valid_dataset = read_tfr(
            self.valid_tf_record,
            self.classes_file,
            features,
            self.max_boxes,
            get_features=True,
        )
        train_dataset.shuffle(shuffle_buffer)
        valid_dataset.shuffle(shuffle_buffer)
        train_dataset = iter(train_dataset)
        valid_dataset = iter(valid_dataset)
        train_predictions = self.predict_dataset(
            train_dataset, workers, 'train', batch_size
        )
        valid_predictions = self.predict_dataset(
            valid_dataset, workers, 'valid', batch_size
        )
        if merge:
            predictions = pd.concat([train_predictions, valid_predictions])
            save_path = os.path.join(
                '..', 'Output', 'Data', 'full_dataset_predictions.csv'
            )
            predictions.to_csv(save_path, index=False)
            return predictions
        train_path = os.path.join(
            '..', 'Output', 'Data', 'train_dataset_predictions.csv'
        )
        valid_path = os.path.join(
            '..', 'Output', 'Data', 'valid_dataset_predictions.csv'
        )
        train_predictions.to_csv(train_path, index=False)
        valid_predictions.to_csv(valid_path, index=False)
        return train_predictions, valid_predictions

    @staticmethod
    def get_area(frame, columns):
        """
        Calculate bounding boxes areas.
        Args:
            frame: pandas DataFrame that contains prediction data.
            columns: column names that represent x1, y1, x2, y2.

        Returns:
            pandas Series(area column)
        """
        x1, y1, x2, y2 = [frame[column] for column in columns]
        return (x2 - x1) * (y2 - y1)

    def get_true_positives(self, detections, actual, min_overlaps):
        """
        Filter True positive detections out of all detections.
        Args:
            detections: pandas DataFrame with all detections.
            actual: pandas DataFrame with real data.
            min_overlaps: a float value between 0 and 1, or a dictionary
                containing each class in self.class_names mapped to its
                minimum overlap

        Returns:
            pandas DataFrame that contains detections that satisfy
                True positive constraints.
        """
        if detections.empty:
            raise ValueError(f'Empty predictions frame')
        if isinstance(min_overlaps, float):
            assert 0 <= min_overlaps < 1, (
                f'min_overlaps should be '
                f'between 0 and 1, {min_overlaps} is given'
            )
        if isinstance(min_overlaps, dict):
            assert all(
                [0 < min_overlap < 1 for min_overlap in min_overlaps.values()]
            )
            assert all([obj in min_overlaps for obj in self.class_names]), (
                f'{[item for item in self.class_names if item not in min_overlaps]} '
                f'are missing in min_overlaps'
            )
        actual = actual.rename(
            columns={'Image Path': 'image', 'Object Name': 'object_name'}
        )
        actual['image'] = actual['image'].apply(lambda x: os.path.split(x)[-1])
        random_gen = np.random.default_rng()
        if 'detection_key' not in detections.columns:
            detection_keys = random_gen.choice(
                len(detections), size=len(detections), replace=False
            )
            detections['detection_key'] = detection_keys
        total_frame = actual.merge(detections, on=['image', 'object_name'])
        assert (
            not total_frame.empty
        ), 'No common image names found between actual and detections'
        total_frame['x_max_common'] = total_frame[['X_max', 'x2']].min(1)
        total_frame['x_min_common'] = total_frame[['X_min', 'x1']].max(1)
        total_frame['y_max_common'] = total_frame[['Y_max', 'y2']].min(1)
        total_frame['y_min_common'] = total_frame[['Y_min', 'y1']].max(1)
        true_intersect = (
            total_frame['x_max_common'] > total_frame['x_min_common']
        ) & (total_frame['y_max_common'] > total_frame['y_min_common'])
        total_frame = total_frame[true_intersect]
        actual_areas = self.get_area(
            total_frame, ['X_min', 'Y_min', 'X_max', 'Y_max']
        )
        predicted_areas = self.get_area(total_frame, ['x1', 'y1', 'x2', 'y2'])
        intersect_areas = self.get_area(
            total_frame,
            ['x_min_common', 'y_min_common', 'x_max_common', 'y_max_common'],
        )
        iou_areas = intersect_areas / (
            actual_areas + predicted_areas - intersect_areas
        )
        total_frame['iou'] = iou_areas
        if isinstance(min_overlaps, float):
            return total_frame[total_frame['iou'] >= min_overlaps]
        if isinstance(min_overlaps, dict):
            class_data = [
                (name, total_frame[total_frame['object_name'] == name])
                for name in self.class_names
            ]
            thresholds = [min_overlaps[item[0]] for item in class_data]
            frames = [
                item[1][item[1]['iou'] >= threshold]
                for (item, threshold) in zip(class_data, thresholds)
                if not item[1].empty
            ]
            return pd.concat(frames)

    @staticmethod
    def get_false_positives(detections, true_positive):
        """
        Filter out False positives in all detections.
        Args:
            detections: pandas DataFrame with detection data.
            true_positive: pandas DataFrame with True positive data.

        Returns:
            pandas DataFrame with False positives.
        """
        keys_before = detections['detection_key'].values
        keys_after = true_positive['detection_key'].values
        false_keys = np.where(np.isin(keys_before, keys_after, invert=True))
        false_keys = keys_before[false_keys]
        false_positives = detections.set_index('detection_key').loc[false_keys]
        return false_positives.reset_index()

    @staticmethod
    def combine_results(true_positive, false_positive):
        """
        Combine True positives and False positives.
        Args:
            true_positive: pandas DataFrame with True positive data.
            false_positive: pandas DataFrame with False positive data.

        Returns:
            pandas DataFrame with all detections combined.
        """
        true_positive['true_positive'] = 1
        true_positive['false_positive'] = 0
        true_positive = true_positive[
            [
                'image',
                'object_name',
                'score',
                'x_min_common',
                'y_min_common',
                'x_max_common',
                'y_max_common',
                'iou',
                'image_width',
                'image_height',
                'true_positive',
                'false_positive',
                'detection_key',
            ]
        ]
        true_positive = true_positive.rename(
            columns={
                'x_min_common': 'x1',
                'y_min_common': 'y1',
                'x_max_common': 'x2',
                'y_max_common': 'y2',
            }
        )
        false_positive['iou'] = 0
        false_positive['true_positive'] = 0
        false_positive['false_positive'] = 1
        false_positive = false_positive[
            [
                'image',
                'object_name',
                'score',
                'x1',
                'y1',
                'x2',
                'y2',
                'iou',
                'image_width',
                'image_height',
                'true_positive',
                'false_positive',
                'detection_key',
            ]
        ]
        return pd.concat([true_positive, false_positive])

    def calculate_stats(
        self,
        actual_data,
        detection_data,
        true_positives,
        false_positives,
        combined,
    ):
        """
        Calculate prediction statistics for every class in self.class_names.
        Args:
            actual_data: pandas DataFrame with real data.
            detection_data: pandas DataFrame with all detection data before filtration.
            true_positives: pandas DataFrame with True positives.
            false_positives: pandas DataFrame with False positives.
            combined: pandas DataFrame with True and False positives combined.

        Returns:
            pandas DataFrame with statistics for all classes.
        """
        class_stats = []
        for class_name in self.class_names:
            stats = dict()
            stats['Class Name'] = class_name
            stats['Average Precision'] = (
                combined[combined['object_name'] == class_name][
                    'average_precision'
                ].sum()
                * 100
            )
            stats['Actual'] = len(
                actual_data[actual_data["Object Name"] == class_name]
            )
            stats['Detections'] = len(
                detection_data[detection_data["object_name"] == class_name]
            )
            stats['True Positives'] = len(
                true_positives[true_positives["object_name"] == class_name]
            )
            stats['False Positives'] = len(
                false_positives[false_positives["object_name"] == class_name]
            )
            stats['Combined'] = len(
                combined[combined["object_name"] == class_name]
            )
            class_stats.append(stats)
        total_stats = pd.DataFrame(class_stats).sort_values(
            by='Average Precision', ascending=False
        )
        return total_stats

    @staticmethod
    def calculate_ap(combined, total_actual):
        """
        Calculate average precision for a single object class.
        Args:
            combined: pandas DataFrame with True and False positives combined.
            total_actual: Total number of actual object class boxes.

        Returns:
            pandas DataFrame with average precisions calculated.
        """
        combined = combined.sort_values(
            by='score', ascending=False
        ).reset_index(drop=True)
        combined['acc_tp'] = combined['true_positive'].cumsum()
        combined['acc_fp'] = combined['false_positive'].cumsum()
        combined['precision'] = combined['acc_tp'] / (
            combined['acc_tp'] + combined['acc_fp']
        )
        combined['recall'] = combined['acc_tp'] / total_actual
        combined['m_pre1'] = combined['precision'].shift(1, fill_value=0)
        combined['m_pre'] = combined[['m_pre1', 'precision']].max(axis=1)
        combined['m_rec1'] = combined['recall'].shift(1, fill_value=0)
        combined.loc[
            combined['m_rec1'] != combined['recall'], 'valid_m_rec'
        ] = 1
        combined['average_precision'] = (
            combined['recall'] - combined['m_rec1']
        ) * combined['m_pre']
        return combined

    @timer(default_logger)
    def calculate_map(
        self,
        prediction_data,
        actual_data,
        min_overlaps,
        display_stats=False,
        fig_prefix='',
        save_figs=True,
        plot_results=True,
    ):
        """
        Calculate mAP(mean average precision) for the trained model.
        Args:
            prediction_data: pandas DataFrame containing predictions.
            actual_data: pandas DataFrame containing actual data.
            min_overlaps: a float value between 0 and 1, or a dictionary
                containing each class in self.class_names mapped to its
                minimum overlap
            display_stats: If True, statistics will be displayed.
            fig_prefix: Prefix for plot titles.
            save_figs: If True, figures will be saved.
            plot_results: If True, results will be calculated.

        Returns:
            pandas DataFrame with statistics, mAP score.
        """
        actual_data['Object Name'] = actual_data['Object Name'].apply(
            lambda x: x.replace("b'", '').replace("'", '')
        )
        class_counts = actual_data['Object Name'].value_counts().to_dict()
        true_positives = self.get_true_positives(
            prediction_data, actual_data, min_overlaps
        )
        false_positives = self.get_false_positives(
            prediction_data, true_positives
        )
        combined = self.combine_results(true_positives, false_positives)
        class_groups = combined.groupby('object_name')
        calculated = pd.concat(
            [
                self.calculate_ap(group, class_counts.get(object_name))
                for object_name, group in class_groups
            ]
        )
        stats = self.calculate_stats(
            actual_data,
            prediction_data,
            true_positives,
            false_positives,
            calculated,
        )
        map_score = stats['Average Precision'].mean()
        if display_stats:
            pd.set_option(
                'display.max_rows',
                None,
                'display.max_columns',
                None,
                'display.width',
                None,
            )
            print(stats.sort_values(by='Average Precision', ascending=False))
            print(f'mAP score: {map_score}%')
            pd.reset_option('display.[max_rows, max_columns, width]')
        if plot_results:
            visualize_pr(calculated, save_figs, fig_prefix)
            visualize_evaluation_stats(stats, fig_prefix)
        return stats, map_score
