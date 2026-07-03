
from torch.utils.data import Dataset

from utils.common import randidx


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                item["retain"] = self.retain[randidx(len(self.retain))]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                item["forget"] = self.forget[randidx(len(self.forget))]
        return item
