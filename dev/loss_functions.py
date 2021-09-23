
def test_loss():
    import monai
    import torch
    import kwarray
    focal = monai.losses.FocalLoss(reduction='none')
    dice = monai.losses.DiceLoss(reduction='none')
    dfloss = monai.losses.DiceFocalLoss(reduction='none')

    logits = torch.ones(10, 1, 1)
    true = torch.ones(10, 1, 1)
    focal(logits, true)

    num_classes = 2
    N = 100

    def inverse_sigmoid(x):
        return -torch.log((1 / (x + 1e-8)) - 1)

    H = 1
    W = 1

    probs = torch.empty((N, num_classes, H, W))
    probs[:, 0, 0, 0] = torch.linspace(1e-6, 1 - 1e-6, N)
    probs[:, 1, 0, 0] = 1 - probs[:, 0, 0, 0]

    logits = inverse_sigmoid(probs)

    true_labels = torch.ones((N, H, W)).long()
    true_ohe = kwarray.one_hot_embedding(true_labels, num_classes=num_classes, dim=1)

    focal_losses = focal(logits, true_ohe)
    dice_losses = dice(logits, true_ohe)
    df_losses = dfloss(probs, true_ohe)

    import kwplot
    import pandas as pd

    sns = kwplot.autosns()

    df = pd.concat([
        # pd.DataFrame({
        #     'probs1':  probs[:, 1, 0, 0].numpy(),
        #     'loss': dice_losses[:, 1].numpy(),
        #     'type': ['dice'] * N
        # }),
        pd.DataFrame({
            'probs1':  probs[:, 1, 0, 0].numpy(),
            'loss': focal_losses[:, 1].numpy(),
            'type': ['focal'] * N
        }),
        pd.DataFrame({
            'probs1':  probs[:, 1, 0, 0].numpy(),
            'loss': df_losses[:, 1].numpy(),
            'type': ['dice-focal'] * N
        }),

    ])

    sns.lineplot(data=df, x='probs1', y='loss', hue='type')
