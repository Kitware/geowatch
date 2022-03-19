class MetadataNotFound(Exception):
    """
    Thrown when metadata does not exist
    """
    pass


class GeoMetadataNotFound(MetadataNotFound):
    """
    Thrown when geographic metadata does not exist
    """
    pass
