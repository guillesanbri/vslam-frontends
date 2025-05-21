from abc import ABC, abstractmethod


class BaseStereoSource(ABC):
    def __init__(self):
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        frame = self.get_frame(self._index)
        self._index += 1
        return frame

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_frame(self, idx):
        """Returns (left_img, right_img)"""
        pass
