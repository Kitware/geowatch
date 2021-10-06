
# Land Cover Segmentation

DZYNE Technologies

Usage:

1. Download weights from `data.kitware.com` in `IARPA SMART WATCH / Model Weights / Land Cover` into `weights`.
2. Build the container.

   ```
   docker build -t watch .
   ```
   
3. Start the container.  Adjust your `smart_watch_dvc` path as necessary.

   ```
   docker run --rm -it --gpus all \
     -v $(pwd)/output:/output \
     -v $(pwd)/../smart_watch_dvc:/dvc:ro \
     watch bash
   ```

4. Run the prediction:
    ```
    python -m watch.tasks.landcover.predict \
      --dataset /dvc/drop1-S2-L8-aligned/data.kwcoco.json \
      --output /output/landcover.kwcoco.json \ 
      --deployed weights/visnav_sentinel2.pt
    ```

