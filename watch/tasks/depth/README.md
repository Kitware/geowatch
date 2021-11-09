
# Depth detector

DZYNE Technologies

Usage:

1. Clone the `smart_watch_dvc` and `watch` repositories.
2. From, the `watch` repository, build the container.

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
    # DVC_DPATH=/dvc

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
    KWCOCO_BUNDLE=$DVC_DPATH/Drop1-Aligned-L1
    KWCOCO_FPATH=$KWCOCO_BUNDLE/data.kwcoco.json

    python -m watch.tasks.depth.predict \
        --dataset  $KWCOCO_BUNDLE/data.kwcoco.json \
        --output   $KWCOCO_BUNDLE/dzyne_depth/depth1.kwcoco.json \ 
        --deployed $DVC_DPATH/models/depth/weights_v1.pt
    ```
