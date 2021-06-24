import predict
import pathlib

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data_path", type=pathlib.Path)
#     parser.add_argument("channel_set")
    parser.add_argument("model_checkpoint_root", type=pathlib.Path)
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--checkpoint_pattern", type=str,
                        default="{channel_set}/lightning_logs/version_0/checkpoints/*.ckpt")
    args = parser.parse_args()

    for channel_set in ["all", "uv", "bgr", "vnir", "swir"]:

        pattern = args.checkpoint_pattern.format(channel_set=channel_set)
        model_checkpoint_path = next(args.model_checkpoint_root.glob(pattern))

        args.channel_set = channel_set
        args.model_checkpoint_path = model_checkpoint_path
        predict.main(args)
