from rostam import _types as t


def test_search_results_is_list_with_status():
    rs = t.SearchResults([{"id": 1}], degraded=True, missing=[2])
    assert list(rs) == [{"id": 1}]          # list-compatible
    assert len(rs) == 1 and rs[0]["id"] == 1
    assert rs.degraded is True and rs.missing == [2]


def test_search_results_defaults_healthy():
    rs = t.SearchResults([])
    assert rs.degraded is False and rs.missing == []


def test_scroll_page_is_list_with_cursor():
    p = t.ScrollPage([{"id": 1}], next_cursor="abc")
    assert list(p) == [{"id": 1}] and p.next_cursor == "abc"


def test_transport_error_is_rostam_error():
    assert issubclass(t.TransportError, t.RostamError)
