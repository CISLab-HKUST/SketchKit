from abc import ABC, abstractmethod
import os
from pathlib import Path as PathLibPath
from sketchkit.core.sketch import Sketch


class SketchDataset(ABC):
    """Abstract base class for sketch datasets.

    This class provides a common interface for all sketch datasets in SketchKit.
    It handles dataset initialization, caching, downloading, and provides standard
    methods for accessing sketch data.

    Attributes:
        root (str): Root directory where the dataset is stored.
        items_metadata (pandas.DataFrame): DataFrame containing metadata for all dataset items.
        load_all (bool): Whether to load all data into memory at initialization.
        cislab_source (bool): Whether to use CISLAB as the data source.
        raw_data: Container for all loaded data when load_all is True.
        items_metadata (pandas.DataFrame): DataFrame containing metadata for all dataset items.
        raw_data (list): List containing memory-loaded data. If the item is not loaded, raw_data[id] is None. __getitem__ return item from this list.
    Note:
        Subclasses must implement all abstract methods to provide dataset-specific
        functionality for integrity checking, downloading, metadata loading, and
        data access.
    """

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """
        Initialize the dataset.
        Args:
            root (str | PathLibPath | None, optional): Root directory for the dataset.
                If None, defaults to ~/.cache/sketchkit/datasets/{ClassName}/. Defaults to None.
            load_all (bool, optional): Whether to load all data into memory at initialization.
                Defaults to False.
            cislab_source (bool, optional): Whether to use CISLAB as the data source.
                Defaults to False.
        Raises:
            RuntimeError: If the dataset is not found or corrupted after download attempt.
        Note:
            The method will automatically download the dataset if it's not found or corrupted
            in the specified root directory.
        """

        if root is None:
            root = os.path.join(
                os.environ.get("HOME", "/"),
                ".cache/sketchkit/datasets",
                self.__class__.__name__,
            )
        if isinstance(root, str):
            root = os.path.expanduser(root)
        os.makedirs(root, exist_ok=True)
        self.root = root

        self.items_metadata = []
        self.load_all = load_all
        self.cislab_source = cislab_source
        self.raw_data = None

        if not self._check_integrity():
            print(
                f"Cache not found or corrupted. Start downloading the dataset to {self.root}."
            )
            self._download()
            if not self._check_integrity():
                raise RuntimeError("Dataset not found or corrupted.")
        self._load_items_metadata()
        if self.load_all:
            self._load_all()

    @abstractmethod
    def _check_integrity(self) -> bool:
        """Check if the dataset files are complete and valid.

        This method should verify that all necessary dataset files exist
        and are not corrupted. It's called during initialization to determine
        if the dataset needs to be downloaded.

        Returns:
            bool: True if the dataset is complete and valid, False otherwise.
        """
        pass

    @abstractmethod
    def _download(self):
        """Download the dataset files to the root directory.

        This method should implement the logic to download all necessary
        dataset files from their source location to the local root directory.
        It's called when the integrity check fails during initialization.

        Raises:
            RuntimeError: If download fails or encounters an error.
        """
        pass

    @abstractmethod
    def _load_items_metadata(self):
        """Load metadata for all items in the dataset into a pandas DataFrame.

        This method should populate the items_metadata DataFrame with information
        about all available items in the dataset. The metadata is used to
        provide quick access to item information without loading the actual data.

        Note:
            This method is called after successful integrity check and should
            populate self.items_metadata as a pandas DataFrame with appropriate
            columns for dataset-specific metadata.
        """
        pass

    @abstractmethod
    def _load_all(self):
        """Load all dataset items into memory.

        This method should load all dataset items into memory for faster
        access. It's called when load_all is True during initialization
        or when explicitly requested.

        Note:
            The loaded data should be stored in self.raw_data for later
            access by __getitem__.
        """
        pass

    def unload(self):
        """
        Unloads the raw data from memory by deleting the reference and setting it to None.
        This method is used to free up memory by removing the raw_data attribute
        from the object. After calling this method, the raw_data will no longer
        be accessible and memory previously occupied by it can be garbage collected.
        """
        del self.raw_data
        self.raw_data = None

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: The total number of items available in the dataset.
        """
        return len(self.items_metadata)

    @abstractmethod
    def __getitem__(self, idx: int) -> Sketch:
        """Retrieve a sketch item by index.

        Args:
            idx (int): Index of the item to retrieve. Must be in range [0, len(dataset)).

        Returns:
            Sketch: The sketch object at the specified index.

        Raises:
            IndexError: If idx is out of bounds.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Creates a formatted string showing the dataset class name, number of samples,
        root location, and any additional information from extra_repr().

        Returns:
            str: Formatted string representation of the dataset.
        """
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of samples: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
            body.append(f"Load all data into memory: {self.load_all}")
            body.append(f"Use CISLAB as data source: {self.cislab_source}")
        body += self.extra_repr().splitlines()

        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """Return extra information for the string representation.

        This method can be overridden by subclasses to provide additional
        dataset-specific information in the string representation.

        Returns:
            str: Additional information to include in __repr__. Empty by default.
        """
        return ""
