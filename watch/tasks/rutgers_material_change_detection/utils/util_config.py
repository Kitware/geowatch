def update_options(cfg):
    if cfg.debug:
        cfg.save = False
        cfg.db.hp.n_workers = 0
    return cfg
