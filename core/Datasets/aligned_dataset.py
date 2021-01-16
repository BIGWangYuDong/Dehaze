from Dehaze.core.Datasets.base_dataset import BaseDataset
from Dehaze.core.Datasets.builder import DATASETS
from Dehaze.core.Datasets.Pipelines import Compose
import copy


@DATASETS.register_module()
class AlignedDataset(BaseDataset):
    def __init__(self, **kwargs):
        super(AlignedDataset, self).__init__( **kwargs)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_infos.append({
                    "image_path": self.img_prefix + data,
                    "gt_path": self.img_prefix + data
                })
        return data_infos

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

