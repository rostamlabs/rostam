import pytest
from rostam.rostam import _parse_target
from rostam import TransportError


def test_http_scheme():
    assert _parse_target("http://h:8080")[0] == "http"
    assert _parse_target("https://h")[0] == "http"


def test_tcp_scheme_and_bare():
    assert _parse_target("tcp://h:7000")[:3] == ("tcp", "h", 7000)
    assert _parse_target("h:7000")[:3] == ("tcp", "h", 7000)   # bare = native


def test_bare_without_port_errors():
    with pytest.raises(TransportError):
        _parse_target("justahost")


def test_malformed_url_port_is_transport_error():
    # urlsplit(...).port raises a bare ValueError for a non-numeric port —
    # must surface as TransportError, not leak out.
    with pytest.raises(TransportError):
        _parse_target("tcp://h:not-a-port")
    with pytest.raises(TransportError):
        _parse_target("http://h:not-a-port")


def test_out_of_range_url_port_is_transport_error():
    with pytest.raises(TransportError):
        _parse_target("tcp://h:65536")
    with pytest.raises(TransportError):
        _parse_target("http://h:65536")


def test_zero_port_is_rejected_everywhere():
    # `port or default` would silently turn :0 into the scheme default —
    # must reject it outright instead, on every code path.
    with pytest.raises(TransportError):
        _parse_target("tcp://h:0")
    with pytest.raises(TransportError):
        _parse_target("http://h:0")
    with pytest.raises(TransportError):
        _parse_target("h:0")


def test_bare_out_of_range_port_is_transport_error():
    with pytest.raises(TransportError):
        _parse_target("h:65536")


def test_bare_malformed_port_is_transport_error():
    with pytest.raises(TransportError):
        _parse_target("h:not-a-port")


def test_valid_edge_ports_are_accepted():
    assert _parse_target("tcp://h:1")[:3] == ("tcp", "h", 1)
    assert _parse_target("tcp://h:65535")[:3] == ("tcp", "h", 65535)
    assert _parse_target("http://h:65535")[:3] == ("http", "h", 65535)


def test_https_without_explicit_port_still_defaults():
    assert _parse_target("https://h")[:3] == ("http", "h", 443)
    assert _parse_target("http://h")[:3] == ("http", "h", 80)
