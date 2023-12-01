def test_version():
    """ Test that the version attribute exists """
    import geowatch
    assert isinstance(geowatch.__version__, str)
