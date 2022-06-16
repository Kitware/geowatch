import kwcoco
import pathlib


def split_onera_all(all_src, train_split_src, test_split_src):

    # Open full dataset
    onera_all = kwcoco.CocoDataset(str(all_src))

    # Open location splits, saved as one-line files with location
    # names separated by commas
    train_locations = open(train_split_src).read().strip().split(",")
    test_locations = open(test_split_src).read().strip().split(",")

    # Identify gids for train and test
    # Catch locations in all that are not listed in train or test
    train_split, test_split, other_split = [], [], []

    for image_id, image in onera_all.imgs.items():
        image_location = image["name"].split("-")[0]  # image["name"] = "{location}-{frame number}"

        if image_location in train_locations:
            train_split.append(image_id)
        elif image_location in test_locations:
            test_split.append(image_id)
        else:
            other_split.append(image_id)

    # Split into subsets
    onera_train = onera_all.subset(train_split, copy=True)
    onera_test = onera_all.subset(test_split, copy=True)
    onera_other = onera_all.subset(other_split, copy=True)

    # Save subsets to disk
    onera_train.dump(str(all_src.parents[0] / "onera_train.kwcoco.json"))
    onera_test.dump(str(all_src.parents[0] / "onera_test.kwcoco.json"))

    # Only save other if non-empty, and alert if that happens
    if onera_other.n_images > 0:
        print("some locations not included in either train or test split!")
        onera_other.dump(str(all_src.parents[0] / "onera_other.kwcoco.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("all_src", type=pathlib.Path)
    parser.add_argument("train_split_src", type=pathlib.Path)
    parser.add_argument("test_split_src", type=pathlib.Path)
    args = parser.parse_args()

    split_onera_all(args.all_src, args.train_split_src, args.test_split_src)
