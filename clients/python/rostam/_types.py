"""Transport-agnostic result types + error hierarchy for the Rostam client."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


class RostamError(Exception):
    """Base for all client-raised errors.

    ``status`` is the HTTP status code when the error came from the HTTP
    transport (0 for an HTTP-level transport failure with no status, ``None``
    on the TCP transport, which has no such concept at all). ``message`` is
    the plain error text, mirroring ``status`` as an attribute (in addition to
    being embedded in the exception's ``str()``/args) so callers can inspect
    it without re-parsing the exception."""

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status
        self.message = message


class TransportError(RostamError):
    """A requested operation is not available on the active transport."""


@dataclass
class SearchResult:
    id: int
    distance: float
    score: float


@dataclass
class Document:
    id: int
    distance: float
    score: float
    content: str
    metadata: Dict[str, Any]


@dataclass
class Group:
    key: Any
    hits: List[Document]


@dataclass
class Point:
    id: int
    vector: Optional[Sequence[float]]
    content: str
    metadata: Dict[str, Any]


class SearchResults(list):
    """A list of results that also carries partial-read status. On a healthy
    single-node server these are (degraded=False, missing=[])."""

    def __init__(self, items: Sequence[Any] = (), *, degraded: bool = False, missing: Sequence[int] = ()):
        super().__init__(items)
        self.degraded = bool(degraded)
        self.missing = list(missing)


class GroupResults(list):
    """A list of Group that also carries partial-read status."""

    def __init__(self, items: Sequence[Any] = (), *, degraded: bool = False, missing: Sequence[int] = ()):
        super().__init__(items)
        self.degraded = bool(degraded)
        self.missing = list(missing)


class ScrollPage(list):
    """A page of Document that also carries the next cursor ('' when exhausted)."""

    def __init__(self, items: Sequence[Any] = (), *, next_cursor: str = ""):
        super().__init__(items)
        self.next_cursor = next_cursor
