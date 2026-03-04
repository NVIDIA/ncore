"""pai-clip-dl: Download and stream clip data from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset."""

from pai_clip_dl.config import Config
from pai_clip_dl.downloader import ClipDownloader
from pai_clip_dl.index import ClipIndex, FeatureSpec
from pai_clip_dl.remote import HFRemote
from pai_clip_dl.streaming import StreamingZipAccess


__all__ = [
    "Config",
    "HFRemote",
    "ClipIndex",
    "FeatureSpec",
    "StreamingZipAccess",
    "ClipDownloader",
]
