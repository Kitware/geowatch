
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
    python -m watch.tasks.depth.predict \
      --dataset /dvc/drop1/data.kwcoco.json \
      --output /output/depth1.kwcoco.json \ 
      --deployed /dvc/models/depth/weights_v1.pt
    ```

