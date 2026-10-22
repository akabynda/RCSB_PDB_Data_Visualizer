"""Public RCSB client assembled from focused service capabilities."""

from .homology import SequenceHomologyMixin
from .metadata import EntryMetadataMixin
from .membrane import MembraneEntriesMixin
from .nmr import SolutionNMRMixin
from .nmr_stride import SolutionNMRStrideMixin
from .search import EntrySearchMixin
from .transport import RCSBTransport


class RCSBClient(
    EntrySearchMixin,
    EntryMetadataMixin,
    MembraneEntriesMixin,
    SequenceHomologyMixin,
    SolutionNMRMixin,
    SolutionNMRStrideMixin,
    RCSBTransport,
):
    """Query RCSB services and retrieve cached coordinate files."""
