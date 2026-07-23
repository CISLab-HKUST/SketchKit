import numpy as np
import torch

from sketchkit.datasets import *
from typing import List

from metrics import segmentation_metrics


class SketchDatasetGroupSeg:
    """
    A class for a group of sketch datasets that is used to compute benchmark metrics of sketch segmentation.
    """

    def __init__(
        self,
        datasets: List[str],
    ):
        """
        Initialize the dataset group.
        """
        self.dataset_names = datasets
        self.dataset_group = {}
        self.part_list_all = []  # in "class-part" format, such as "dog-head"

        for dataset_name in datasets:
            if dataset_name == "SketchXPRIS":
                ds = SketchXPRIS()
                self.dataset_group[dataset_name] = ds
                self.part_list_all += ds.all_class_part_labels
                print(len(ds.all_class_part_labels))
            elif dataset_name == "CreativeSketch":
                ds = CreativeSketch(cislab_source=True)
                self.dataset_group[dataset_name] = ds
                self.part_list_all += ds.all_class_part_labels
                print(len(ds.all_class_part_labels))
            elif dataset_name == "SketchIME":
                ds = SketchIME()
                self.dataset_group[dataset_name] = ds
                self.part_list_all += ds.all_class_part_labels
                print(len(ds.all_class_part_labels))
            else:
                raise Exception("TBD")

        self.part_list_all = list(map(str.lower, self.part_list_all))
        self.part_list_all = sorted(list(set(self.part_list_all)))
        self.num_part_label_all = len(self.part_list_all)
        print('Number of all parts:', self.num_part_label_all)

    def compute_segmentation_metrics(
        self,
        model,
        train_part_label_list: List[str],
        data_split: str
    ):
        """
        Compute segmentation metrics (point-based accuracy (P_metric) and component/path-based accuracy (C_metric)) from SketchGNN
        for a given sketch segmentation model.

        Args:
            model: The trained segmentation model.
            train_part_label_list (List[str]): The part label list used during training.
                Each item should be in "class-part" format, such as "dog-head".
            data_split (str): "valid" or "test".

        Returns:

        """
        # assert data_split in ["valid", "test"]

        gt_labels_list = []  # A list of GT labels
        predict_part_label_list = []  # Predicted part labels for all examples

        for dataset_name_i in self.dataset_names:
            dataset_i = self.dataset_group[dataset_name_i]
            dataset_i_split = dataset_i.items_metadata[
                dataset_i.items_metadata["split"] == data_split]

            for _, row in dataset_i_split.iterrows():
                sketch = dataset_i[row.id]
                sketch_category = row.category
                sketch_label_names = row.label_names.split(',')

                # print(sketch.width, sketch.height)
                # print(sketch.path_num, sketch.curve_num)
                # print(sketch_category, sketch_label_names)

                # TODO: First, preprocess the input sketch into the input format of the model
                input_sketch = convert_format(sketch)

                # TODO: Then, input to the model and output the probability
                output_prob = model(input_sketch)  # (num_point, num_part_label_raw), probability after applying softmax operation to logits

                # Get predicted part label for each point according to the maximum probability
                output_part_labels_raw = torch.argmax(output_prob, dim=1).cpu().numpy()  # (num_point)

                # Here, the predicted part labels correspond to those in training dataset.
                # We should convert them into labels corresponding to the combined part label list
                output_part_labels_comb = self._adapt_to_combined_part_label_list(output_part_labels_raw, train_part_label_list)  # (num_point)
                predict_part_label_list.append(output_part_labels_comb)

                # GT label should correspond to the combined part label list
                gt_labels = []
                for path_i in sketch.paths:
                    gt_labels_path = []
                    for i, curve_i in enumerate(path_i):
                        p_start_label = curve_i.p_start.attributes["group_id"]
                        p_start_label_name = "-".join([sketch_category, sketch_label_names[int(p_start_label)]])
                        gt_labels_path.append(self.part_list_all.index(p_start_label_name.lower()))

                        if i == len(path_i) - 1:
                            p_end_label = curve_i.p_end.attributes["group_id"]
                            p_end_label_name = "-".join([sketch_category, sketch_label_names[int(p_end_label)]])
                            gt_labels_path.append(self.part_list_all.index(p_end_label_name.lower()))
                    gt_labels.append(gt_labels_path)
                gt_labels_list.append(gt_labels)

        ## Compute the metrics
        segmentation_metrics(gt_labels_list, predict_part_label_list)

    def _adapt_to_combined_part_label_list(
        self,
        raw_part_labels,
        train_part_label_list,
    ):
        """

        Args:
            raw_part_labels: (num_point), nd.array.
            train_part_label_list (List[str]): The part label list used during training.
                Each item should be in "class-part" format, such as "dog-head".

        Returns:
            comb_part_labels: (num_point), nd.array. A part label list corresponding to the combined part label list.
        """
        comb_part_labels = []
        for pi in range(len(raw_part_labels)):
            raw_part_label = raw_part_labels[pi]
            raw_part_label_name = train_part_label_list[raw_part_label]
            assert '-' in raw_part_label_name, "The part label name should be in \"class-part\" format, such as \"dog-head\"."
            comb_part_label = self.part_list_all.index(raw_part_label_name.lower())
            comb_part_labels.append(comb_part_label)
        return comb_part_labels


if __name__ == '__main__':
    datasets = ["SketchXPRIS",
                "CreativeSketch",
                "SketchIME"
    ]
    sdg = SketchDatasetGroupSeg(datasets)

    sdg.compute_segmentation_metrics(None, ["none"], "test")