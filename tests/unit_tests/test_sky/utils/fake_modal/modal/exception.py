"""The parts of modal.exception that the exec helper uses."""


class Error(Exception):
    pass


class AuthError(Error):
    pass


class InvalidError(Error):
    pass


class NotFoundError(Error):
    pass


class PermissionDeniedError(Error):
    pass


class AlreadyExistsError(Error):
    pass
