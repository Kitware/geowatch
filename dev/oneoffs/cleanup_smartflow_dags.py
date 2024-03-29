def cleanup_smartflow_dags():
    """
    Move dag files older than 1 month out of the main bucket
    """
    from geowatch.utils import util_fsspec
    from kwutil.util_time import datetime
    from kwutil.util_time import timedelta
    import ubelt as ub
    remote_root = util_fsspec.FSPath.coerce('s3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0')

    live_dag_dpath = remote_root / 'dags'
    old_dag_dpath = remote_root / 'dags-old'

    now = datetime.coerce('now')
    delta_thresh = timedelta.coerce('1 month')

    groups = ub.udict({'move': [], 'keep': []})
    for path in live_dag_dpath.ls():
        action = 'keep'
        if path.is_file():
            # Weird that this seems backend specific
            mtime = datetime.coerce(path.stat()['LastModified'])
            age = now - mtime
            if age > delta_thresh:
                action = 'move'
        groups[action].append(path)

    action_hist = groups.map_values(len)
    print(f'action_hist = {ub.urepr(action_hist, nl=1)}')

    tasks = []
    for fpath in groups['move']:
        tasks.append({
            'src': fpath,
            'dst': old_dag_dpath / fpath.name,
        })

    # Execute tasks
    old_dag_dpath.ensuredir()
    for task in ub.ProgIter(tasks, desc='moving'):
        task['src'].copy(task['dst'])

    for task in ub.ProgIter(tasks, desc='fixing'):
        task['src'].delete()
