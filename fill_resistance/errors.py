"""User-facing error hierarchy.

Every UserFacingError message is shown both on stdout (KiCad status bar)
and in a matplotlib error figure, so keep messages self-contained and
actionable.
"""


class UserFacingError(Exception):
    pass


class ApiVersionError(UserFacingError):
    pass


class SelectionError(UserFacingError):
    pass


class CandidateError(UserFacingError):
    pass


class ElectrodeError(UserFacingError):
    pass


class ConnectivityError(UserFacingError):
    pass


class GridSizeError(UserFacingError):
    pass
