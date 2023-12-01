# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.tasks.rutgers_material_seg.experiments.spacenet2_finetune import experiment, torch, torch, np, random, torch, print, experiment, experiment, experiment, experiment, experiment, experiment, print, model


def __getattr__(key):
    import geowatch.tasks.rutgers_material_seg.experiments.spacenet2_finetune as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.tasks.rutgers_material_seg.experiments.spacenet2_finetune as mirror
    return dir(mirror)


if __name__ == '__main__':
    project_root = '/'.join(current_path[:-1])
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    main_config_path = f"{project_root}/configs/main.yaml"

    initial_config = utils.load_yaml_as_dict(main_config_path)
    # experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"
    experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"
    # config_path = utils.dictionary_contents(os.getcwd()+"/",types=["*.yaml"])[0]

    experiment_config = utils.config_parser(
        experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime(
        '%Y-%m-%d-%H:%M:%S')

    # _{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    project_name = f"{current_path[-3]}_{current_path[-1]}_{config['dataset']}"
    experiment_name = f"attention_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    experiment = comet_ml.Experiment(api_key=config['cometml']['api_key'],
                                     project_name=project_name,
                                     workspace=config['cometml']['workspace'],
                                     display_summary_level=0)

    config['experiment_url'] = str(experiment.url)

    experiment.set_name(experiment_name)

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.set_default_dtype(torch.float32)

    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)
    # config['device_ids'] = device_ids
    # gpu_devices = ','.join([str(id) for id in device_ids])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = torch.device('cuda')

    # config['devices_used'] = gpu_devices
    experiment.log_asset_data(config)
    experiment.log_text(config)
    experiment.log_parameters(config)
    experiment.log_parameters(config['training'])
    experiment.log_parameters(config['evaluation'])
    experiment.log_parameters(config['visualization'])

    if config['visualization']['train_imshow'] or config['visualization']['val_imshow']:
        matplotlib.use('TkAgg')

    if config['training']['resume'] != False:
        base_path = '/'.join(config['training']['resume'].split('/')[:-1])
        pretrain_config_path = f"{base_path}/config.yaml"
        pretrain_config = utils.load_yaml_as_dict(pretrain_config_path)
        # print(config['training']['model_feats_channels'])
        # print(pretrain_config_path['training']['model_feats_channels'])
        config['data']['channels'] = pretrain_config['data']['channels']
        # if not config['training']['model_feats_channels'] == pretrain_config_path['training']['model_feats_channels']:
        #     print("the loaded model does not have the same number of features as configured in the experiment yaml file. Matching channel sizes to the loaded model instead.")
        # config['training']['model_feats_channels'] = pretrain_config_path['training']['model_feats_channels']
        config['data']['num_classes'] = pretrain_config['data']['num_classes']
        config['training']['model_feats_channels'] = pretrain_config['training']['model_feats_channels']

    if config['data']['name'] == 'geowatch' or config['data']['name'] == 'onera':
        coco_fpath = ub.expandpath(config['data'][config['location']]['train_coco_json'])
        dset = kwcoco.CocoDataset(coco_fpath)
        sampler = ndsampler.CocoSampler(dset)

        window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
        input_dims = (config['data']['image_size'], config['data']['image_size'])

        channels = config['data']['channels']
        num_channels = len(channels.split('|'))
        config['training']['num_channels'] = num_channels

        dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
        train_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])

        test_coco_fpath = ub.expandpath(config['data'][config['location']]['test_coco_json'])
        test_dset = kwcoco.CocoDataset(test_coco_fpath)
        test_sampler = ndsampler.CocoSampler(test_dset)

        test_dataset = SequenceDataset(test_sampler, window_dims, input_dims, channels)
        test_dataloader = test_dataset.make_loader(batch_size=config['evaluation']['batch_size'])
    else:
        train_dataloader = build_dataset(dataset_name=config['data']['name'],
                                        root=config['data'][config['location']]['train_dir'],
                                        batch_size=config['training']['batch_size'],
                                        num_workers=config['training']['num_workers'],
                                        split='train',
                                        crop_size=config['data']['image_size'],
                                        channels=config['data']['channels'],
                                        )

        test_dataloader = build_dataset(dataset_name=config['data']['name'],
                                        root=config['data'][config['location']]['train_dir'],
                                        batch_size=config['evaluation']['batch_size'],
                                        num_workers=config['training']['num_workers'],
                                        split='val',
                                        crop_size=config['data']['image_size'],
                                        channels=config['data']['channels'],
                                        )

    if not config['training']['model_diff_input']:
        config['training']['num_channels'] = 2 * config['training']['num_channels']

    model = build_model(model_name=config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'],
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'],
                        num_channels=config['training']['num_channels'],
                        out_dim=config['training']['out_features_dim'],
                        feats=config['training']['model_feats_channels'])

    # model = SupConResNet(name=config['training']['backbone'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader),
                                                     eta_min=config['training']['learning_rate'])

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            # model_dict = model.state_dict()
            # if model_dict == checkpoint['model']:
            #     print(f"Succesfuly loaded model from {config['training']['resume']}")
            # else:
            #     pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            #     model_dict.update(pretrained_dict)
            #     model.load_state_dict(model_dict)
            #     print("There was model mismatch. Matching elements in the pretrained model were loaded.")
            missing_keys, unexpexted_keys = model.load_state_dict(checkpoint['model'], strict=False)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"loadded model succeffuly from: {config['training']['resume']}")
            print(f"Missing keys from loaded model: {missing_keys}, unexpected keys: {unexpexted_keys}")
        else:
            print("no checkpoint found at {}".format(
                config['training']['resume']))
            exit()

    trainer = Trainer(model,
                      train_dataloader,
                      test_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=test_dataloader,
                      test_with_full_supervision=config['training']['test_with_full_supervision']
                      )
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
