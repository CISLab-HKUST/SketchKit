import numpy as np

from sketchkit.datasets import *
from typing import List

from metrics import recognition_metrics


class SketchDatasetGroup:
    """
    A class for a group of sketch datasets that is used to compute benchmark metrics.
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
        self.class_list_all = []

        for dataset_name in datasets:
            if dataset_name == "TUBerlin":
                ds = TUBerlin()
                self.dataset_group[dataset_name] = ds
                self.class_list_all += ds.all_categories
                print(len(ds.all_categories))
            elif dataset_name == "QuickDraw":
                ds = QuickDraw()
                self.dataset_group[dataset_name] = ds
                self.class_list_all += ds.all_categories
                print(len(ds.all_categories))
            elif dataset_name == "Sketchy":
                ds = Sketchy()
                self.dataset_group[dataset_name] = ds
                self.class_list_all += ds.all_categories
                print(len(ds.all_categories))
            else:
                raise Exception("TBD")

        self.class_list_all = list(map(str.lower, self.class_list_all))
        self.class_list_all = sorted(list(set(self.class_list_all)))
        self.num_class_all = len(self.class_list_all)
        print('Number of all classes:', self.num_class_all)

    def compute_recognition_metrics(
        self,
        model,
        train_class_list: List[str],
        data_split: str
    ):
        """
        Compute recognition metrics (ACC@1, ACC@5, mAP) for a given sketch recognition model.

        Args:
            model: The trained recognition model.
            train_class_list (List[str]): The class name list used during training.
            data_split (str): "valid" or "test".

        Returns:

        """
        assert data_split in ["valid", "test"]

        gt_labels_list = []  # A list of GT labels
        predict_probs_list = []  # A list of predicted probability

        for dataset_name_i in self.dataset_names:
            dataset_i = self.dataset_group[dataset_name_i]
            dataset_i_split = dataset_i.items_metadata[
                dataset_i.items_metadata["split"] == data_split]

            for _, row in dataset_i_split.iterrows():
                sketch = dataset_i[row.id]
                sketch_category = row.category

                # print(sketch.width, sketch.height)
                # print(sketch.path_num, sketch.curve_num)
                # print(sketch_category)

                # TODO: First, preprocess the input sketch into the input format of the model
                input_sketch = convert_format(sketch)

                # TODO: Then, input to the model and output the probability
                output_prob = model(input_sketch)  # (num_class_raw), probability after applying softmax operation to logits
                # Convert to numpy array
                output_prob = output_prob.detach().cpu().numpy()

                # Here, the probability list corresponds to the class list of training dataset.
                # We should convert it into a list corresponding to the combined class list
                output_prob_comb = self._adapt_to_combined_class_list(output_prob, train_class_list)  # (num_class_all)
                predict_probs_list.append(output_prob_comb)

                # GT label should correspond to the combined class list
                gt_labels_list.append(self.class_list_all.index(sketch_category.lower()))

        ## Compute the metrics
        gt_labels_list = np.array(gt_labels_list)  # (num_example_all)
        predict_probs_list = np.stack(predict_probs_list, axis=0)  # (num_example_all, num_class_all)
        recognition_metrics(gt_labels_list, predict_probs_list, self.num_class_all)

    def _adapt_to_combined_class_list(
        self,
        raw_probs,
        train_class_list,
    ):
        """

        Args:
            raw_probs: (num_class_raw), nd.array
            train_class_list (List[str]): The class name list used during training.

        Returns:
            comb_probs: (num_class_all), nd.array. A probability list corresponding to the combined class list.
        """
        comb_probs = np.zeros(shape=(self.num_class_all), dtype=np.float32)
        for raw_prob_i in range(len(raw_probs)):
            raw_prob = raw_probs[raw_prob_i]
            raw_class_name = train_class_list[raw_prob_i]
            comb_probs[self.class_list_all.index(raw_class_name.lower())] = raw_prob
        return comb_probs


if __name__ == '__main__':
    datasets = ["TUBerlin",
                # "QuickDraw",
                # "Sketchy"
    ]
    sdg = SketchDatasetGroup(datasets)

    sdg.compute_recognition_metrics(None, ["none"], "valid")