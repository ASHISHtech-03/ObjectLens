"""Custom exception hierarchy so callers can handle failures precisely."""


class ObjectLensError(Exception):
    """Base class for all application-specific errors."""


class ImageLoadError(ObjectLensError):
    """Raised when an image cannot be found, decoded or is unsupported."""


class ConfigError(ObjectLensError):
    """Raised when configuration values are invalid."""


class ModelNotTrainedError(ObjectLensError):
    """Raised when a classifier is used before it was trained / loaded."""


class CalibrationError(ObjectLensError):
    """Raised when pixel-to-millimetre calibration cannot be computed."""


class StorageError(ObjectLensError):
    """Raised for database related failures."""
