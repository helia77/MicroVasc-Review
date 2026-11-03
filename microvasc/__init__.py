from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("microvasc")
except PackageNotFoundError:
    # running from a source checkout that is not installed
    __version__ = "unknown"
