'''
ipython -i -c "if 1:
    fpath = '/home/local/KHQ/jon.crall/.cache/xdev/snapshot_states/state_2024-04-30T152406-5.pkl'
    from xdev.embeding import load_snapshot
    load_snapshot(fpath, globals())
"
'''

import kwplot
import ubelt as ub
sns = kwplot.autosns()

# agg = NotImplemented


# table = agg.table
# agg.build_macro_tables(rois=['KR_R002', 'CN_C000', 'KW_C001', 'CO_C001'])
agg.build_macro_tables(rois=['KR_R002'])
macro_key = list(agg.macro_key_to_regions)[-1]
table = agg.region_to_tables[macro_key]

paths = table['resolved_params.bas_pxl.package_fpath'].tolist()

step_col = []
variant_col = []
epoch_col = []
for fpath in paths:
    fpath = ub.Path(fpath)
    parts = ub.Path(fpath).name.split('_')[-2:]
    print(f'parts={parts}')
    p1, p2 = parts
    assert p1.startswith('epoch')
    assert p2.startswith('step')
    step_num = int(p2.split('.')[0].split('step')[1])
    step_col.append(step_num)

    variant_col.append(fpath.parent.name)
    epoch_col.append(int(p1[5:]))

table['step'] = step_col
table['epoch'] = epoch_col

table['resolved_params.bas_pxl_fit.model.init_args.name']
table['training_session'] = variant_col

fig = kwplot.figure(fnum=1, doclf=1)
ax = fig.gca()
# sns.lineplot(data=table, ax=ax, x='step', y='metrics.bas_poly_eval.bas_faa_f1', hue='training_session')

ax.cla()
sns.lineplot(data=table, ax=ax, x='epoch', y='metrics.bas_poly_eval.bas_faa_f1', hue='training_session')
