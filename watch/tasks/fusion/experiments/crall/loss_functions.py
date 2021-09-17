def devcheck_loss_functions():
    """
    TODO:
        figure out how to gracefully swap between monai Focal Loss and others

    """
    import monai
    import torch

    pred = torch.FloatTensor([
        [ 10, 1],
        [  1, 10],
        [ 100, 0],
        [ 1, 10],
        [ 10, 1],
        [ 0, 100],
    ])
    true = torch.FloatTensor([1, 1, 1, 0, 0, 0]).long()
    crit = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([2, 1]), reduction='none')
    crit(pred, true.long())

    change_criterion = monai.losses.FocalLoss(reduction='none', to_onehot_y=True, weight=[0, 1])
    print(change_criterion.forward(pred, true))

    crit = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.ones(1) * 2.)
    crit(pred, true)
