
# Land Cover Segmentation

DZYNE Technologies

Usage:

1. Download weights from `data.kitware.com` in `IARPA SMART WATCH / Model Weights / Land Cover`

2. Run the prediction:
    ```
    python -m watch.tasks.landcover.predict --dataset /dvc/drop0_aligned/data.kwcoco.json --deployed weights/visnav_osm.pt
    ```

