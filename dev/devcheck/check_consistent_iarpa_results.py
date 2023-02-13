'''
cd /data/joncrall/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_6b00fb69
'''
# flake8: noqa
import pandas as pd
import ubelt as ub
import math
paths = list(ub.Path('.').glob('results_*'))


class Result:
    def __init__(self, path, csv_paths):
        self.name = path.name
        self.path = path
        self.csv_paths = csv_paths
        self.data = {}

    def load_data(self):
        for p in self.csv_paths:
            self.data[p.name] = pd.read_csv(p)


results = {}

for path in paths:
    csv_paths = list((path / 'AE_R001/overall').glob('*/*.csv'))
    r = Result(path, csv_paths)
    results[r.name] = self = r
    r.load_data()



import xdev
comparisons = {}


def pandas_issame(d1, d2):
    is_null1 = pd.isnull(d1)
    is_null2 = pd.isnull(d2)
    is_eq = (d1 == d2)
    is_same = is_eq | (is_null1 & is_null2)
    return is_same


def normalize_vs_text(text):
    parts = text.split('vs.')
    new_parts = []
    for p in parts:
        new_parts.append(ub.urepr(set(sorted(eval(p))), sort=1, nl=0))
    fixed = ' vs. '.join(new_parts)
    return fixed


main = results['results_main']
for r in results.values():
    key = (main.name, r.name)
    comparisons[key] = {}
    s1 = set(main.data.keys())
    s2 = set(r.data.keys())
    comparisons[key]['csv_name_overlap'] = xdev.set_overlaps(s1, s2)

    bad_csvs = comparisons[key]['bad_csvs'] = []

    for k in main.data.keys():
        d1 = main.data[k]
        d2 = r.data[k]
        is_same = pandas_issame(d1, d2)
        flag = is_same.all().all()
        if not flag:
            # Might need to normalize the order, which seems to be different
            # from run to run.
            if 'site model' in d1.columns:
                d1_norm = d1.sort_values('site model').reset_index(drop=True).drop(['Unnamed: 0'], axis=1)
                d2_norm = d2.sort_values('site model').reset_index(drop=True).drop(['Unnamed: 0'], axis=1)
                is_same2 = pandas_issame(d1_norm, d2_norm)
                flag = is_same2.all().all()
            else:
                is_row_bad = ~is_same.all(axis=1)
                bad1_rows = d1.loc[is_row_bad]
                bad2_rows = d2.loc[is_row_bad]
                bad_issame = pandas_issame(bad1_rows, bad2_rows)
                is_col_bad = ~bad_issame.all(axis=0)
                bad1 = bad1_rows.loc[:, is_col_bad]
                bad2 = bad2_rows.loc[:, is_col_bad]
                try:
                    for row1, row2 in zip(bad1.to_dict('records'), bad2.to_dict('records')):
                        for rk in row1.keys():
                            v1 = row1[rk]
                            v2 = row2[rk]
                            if isinstance(v1, str):
                                v1 = normalize_vs_text(v1)
                            if isinstance(v2, str):
                                v2 = normalize_vs_text(v2)
                            assert (isinstance(v1, float) and isinstance(v2, float) and math.isnan(v1) and math.isnan(v2)) or v1 == v2
                except AssertionError:
                    flag = False
                else:
                    flag = True
            if not flag:
                # assert False
                # assert flag
                bad_csvs.append(k)

print('comparisons = {}'.format(ub.urepr(comparisons, nl=3)))
