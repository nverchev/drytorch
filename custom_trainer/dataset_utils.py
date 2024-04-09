from torch.utils.data import Dataset


class IndexDataset(Dataset):
    """
    This class is used to create a dataset that can be used in a DataLoader.
    """

    def __getitem__(self, index):
        return index
