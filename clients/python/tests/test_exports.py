"""The unified package surface: what ``import rostam`` gives you.

``Rostam`` is the one entry point (transport chosen from the target); the old
transport-specific classes (``RostamClient``, ``RostamKV``) are gone, both as
imports and as attributes on the package.
"""

import rostam


def test_unified_exports_present():
    assert hasattr(rostam, "Rostam")
    for name in (
        "SearchResult", "Document", "Group", "Point", "ScrollPage",
        "SearchResults", "GroupResults", "MultiResult", "TransportError", "RostamError",
        "filters", "Embedder", "FunctionEmbedder", "OpenAIEmbedder", "TextStore",
    ):
        assert hasattr(rostam, name), name


def test_old_classes_gone():
    assert not hasattr(rostam, "RostamClient")
    assert not hasattr(rostam, "RostamKV")


def test_rostam_is_the_unified_facade():
    # from rostam import Rostam gives the facade (rostam.rostam.Rostam), not a
    # transport-specific class.
    from rostam.rostam import Rostam as FacadeRostam

    assert rostam.Rostam is FacadeRostam


def test_rostam_error_carries_status_and_message():
    err = rostam.RostamError("boom", status=404)
    assert err.status == 404
    assert err.message == "boom"
    assert str(err) == "boom"

    # status defaults to None (the TCP transport has no HTTP status concept).
    assert rostam.RostamError("boom").status is None
