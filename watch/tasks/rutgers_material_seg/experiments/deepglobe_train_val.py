# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.tasks.rutgers_material_seg.experiments.deepglobe_train_val import experiment, torch, torch, np, random, torch, experiment, experiment, experiment, experiment, experiment, experiment, print, model


def __getattr__(key):
    import geowatch.tasks.rutgers_material_seg.experiments.deepglobe_train_val as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.tasks.rutgers_material_seg.experiments.deepglobe_train_val as mirror
    return dir(mirror)


if __name__ == '__main__':

    main_config_path = f"{os.getcwd()}/configs/main.yaml"
    initial_config = utils.load_yaml_as_dict(main_config_path)
    experiment_config_path = f"{os.getcwd()}/configs/{initial_config['dataset']}.yaml"

    experiment_config = utils.config_parser(experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    project_name = f"{current_path[-3]}_{current_path[-1]}"  # _{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment_name = f"SMART_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    experiment = comet_ml.Experiment(api_key=config['cometml']['api_key'],
                                     project_name=project_name,
                                     workspace=config['cometml']['workspace'],
                                     display_summary_level=0)
    experiment.set_name(experiment_name)

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.set_default_dtype(torch.float32)

    # device_cpu = torch.device('cpu')
    # print(config['data']['image_size'])
    device_ids = list(range(torch.cuda.device_count()))
    config['device_ids'] = device_ids
    gpu_devices = ','.join([str(id) for id in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = torch.device('cuda')

    config['devices_used'] = gpu_devices
    experiment.log_asset_data(config)
    experiment.log_text(config)
    experiment.log_parameters(config)
    experiment.log_parameters(config['training'])
    experiment.log_parameters(config['evaluation'])
    experiment.log_parameters(config['visualization'])

    train_dataloader = build_dataset(dataset_name=config['data']['name'],
                                     root=config['data'][config['location']]['test_dir'],
                                     batch_size=config['training']['batch_size'],
                                     num_workers=config['training']['num_workers'],
                                     split="train",
                                     image_size=config['data']['image_size'],
                                     )

    validation_dataloader = build_dataset(dataset_name=config['data']['name'],
                                          root=config['data'][config['location']]['test_dir'],
                                          batch_size=config['training']['batch_size'],
                                          num_workers=config['training']['num_workers'],
                                          split="val",
                                          image_size=config['data']['image_size'],
                                          )

    fs_test_loader = build_dataset(dataset_name=config['data']['name'],
                                   root=config['data'][config['location']]['test_dir'],
                                   batch_size=config['training']['batch_size'],
                                   num_workers=config['training']['num_workers'],
                                   split="test",
                                   image_size=config['data']['image_size'],
                                   )

    model = build_model(model_name=config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'] + 1,
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)
    model.to(device)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    optimizer = optim.SGD(model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader),
                                                     eta_min=config['training']['learning_rate'])

    if not config['training']['resume']:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"loaded model from {config['training']['resume']}")
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
            exit()

    trainer = Trainer(model,
                      train_dataloader,
                      validation_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=fs_test_loader
                      )
    train_losses, val_losses, mean_ious_val = trainer.forward(experiment)
