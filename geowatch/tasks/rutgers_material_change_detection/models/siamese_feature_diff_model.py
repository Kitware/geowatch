# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import watch.tasks.rutgers_material_change_detection.models.siamese_feature_diff_model as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.rutgers_material_change_detection.models.siamese_feature_diff_model as mirror
    return dir(mirror)


if __name__ == '__main__':
    # Create fake input data (cpu)
    data = {"images": torch.zeros(5, 2, 3, 256, 256)}

    # Build model object
    model = SiameseDifference()

    # Pass data through model
    pred = model(data)
    assert "change_pred" in pred.keys()
    assert len(pred["change_pred"].shape) == 4
