import torch


class VideoPatchifyzer:
    def __init__(self, frame_size, patch_size):
        """Divide and reshape a video into patches.

        Input video shape [batch_size, n_frames, n_channels, height, width] gets divded into a
        patch video of shape [batch_size, n_frames, n_channels, n_patches, patch_size, patch_size].
            Where patch_size = frame_height/patch_height.

        NOTE: The size of the frame and the patch must be squres and must be divisible.

        Args:
            frame_size (tupe(int, int)): A tuple containing each frames height and width.
            patch_size (tupe(int, int)): A tuple containing each patch height and width.
        """
        frame_height, frame_width = frame_size
        patch_height, patch_width = patch_size

        assert frame_height == frame_width, f"Frame size is not square: HW = [{frame_height},{frame_width}]"
        assert patch_height == patch_width, f"Patch size is not square: HW = [{patch_height},{patch_width}]"

        assert (
            frame_height % patch_height
        ) == 0, f"Frame size is not divisible by patch size equally: {frame_height} !/ {patch_height}"

        self.frame_size = frame_height

    def __call__(self, video):
        """Reshape the frames of a video into equal patches.

        Args:
            video (torch.tensor): A tensor of shape [batch_size, n_frames, n_channels, n_patches, patch_size, patch_size].
        """
        return video.unfold(3, 3, 3).unfold(4, 3, 3).flatten(5).permute(0, 1, 2, 5, 3, 4)


if __name__ == "__main__":
    # Test 1: Make sure that video patchifyzer is generate correct shapes.
    patchifyzer = VideoPatchifyzer(frame_size=(9, 9), patch_size=(3, 3))
    batch_size, n_frames, n_channels, height, width = 2, 5, 4, 9, 9
    test_video = torch.zeros([batch_size, n_frames, n_channels, height, width])
    target_size = torch.Size([2, 5, 4, 9, 3, 3])
    test_patch_video = patchifyzer(test_video)
    output_shape = test_patch_video.shape
    assert output_shape == target_size, f"Output patch video is shape {output_shape} but should be {target_size}"
