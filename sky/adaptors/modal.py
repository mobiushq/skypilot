"""Modal cloud adaptor."""

from sky.adaptors import common

_IMPORT_ERROR_MESSAGE = ('Failed to import dependencies for Modal. '
                         'Try running: pip install "skypilot[modal]"')

modal = common.LazyImport('modal', import_error_message=_IMPORT_ERROR_MESSAGE)
