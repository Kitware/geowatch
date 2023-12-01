
# Depth detector

DZYNE Technologies

Usage:

1. Clone the `smart_watch_dvc` and `geowatch` repositories.
2. From, the `geowatch` repository, build the container.

   ```
   docker build -t geowatch .
   ```
   
3. Start the container.  Adjust your `smart_watch_dvc` path as necessary.

   ```
   docker run --rm -it --gpus all \
     -v $(pwd)/output:/output \
     -v $(pwd)/../smart_watch_dvc:/dvc:ro \
     geowatch bash
   ```

4. Run the prediction:
    ```
    # DVC_DPATH=/dvc
    # DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/

    DVC_DPATH=$(geowatch_dvc)
    KWCOCO_BUNDLE=$DVC_DPATH/Drop1-Aligned-L1
    KWCOCO_FPATH=$KWCOCO_BUNDLE/data.kwcoco.json

    python -m geowatch.tasks.depth.predict \
        --dataset  $KWCOCO_BUNDLE/data.kwcoco.json \
        --output   $KWCOCO_BUNDLE/dzyne_depth/depth1.kwcoco.json \
        --deployed $DVC_DPATH/models/depth/weights_v1.pt

    kwcoco subset --src $KWCOCO_BUNDLE/dzyne_depth/depth1.kwcoco.json \
            --dst $KWCOCO_BUNDLE/dzyne_depth/depth1_wv.kwcoco.json \
            --select_images '.sensor_coarse == "WV"' --channels='red|green|blue'

    python -m geowatch stats $KWCOCO_BUNDLE/dzyne_depth/depth1_wv.kwcoco.json

    # Visualize results
    python -m geowatch visualize --src $KWCOCO_BUNDLE/dzyne_depth/depth1_wv.kwcoco.json \
        --viz_dpath=$KWCOCO_BUNDLE/dzyne_depth/_vizdepth  \
        --channels "depth" --draw_anns=False --animate=True
    ```
