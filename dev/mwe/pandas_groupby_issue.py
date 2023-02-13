
def main():
    import pandas as pd
    import numpy as np
    models = ['a', 'b', 'c']
    rng = np.random.RandomState(0)
    data = pd.DataFrame([
        {
            'tpr': rng.rand(),
            'fpr': rng.rand(),
            'f1': rng.rand(),
            'model': rng.choice(models),
        } for _ in range(10)])

    # Variant 1 (no warning)
    groups = data.groupby('model')
    for group_key, group in groups:
        print(f'group_key={group_key}')
        print(group)
        break

    # Variant 2 (warning)
    groups = data.groupby(['model'])
    for group_key, group in groups:
        print(f'group_key={group_key}')
        print(group)
        break

    import wrapt
    class GroupbyFutureWrapper(wrapt.ObjectProxy):
        """
        Wraps a groupby object to get the new behavior sooner.
        """
        def __iter__(self):
            keys = self.keys
            if isinstance(keys, list) and len(keys) == 1:
                # Handle this special case to avoid a warning
                for key, group in self.grouper.get_iterator(self._selected_obj, axis=self.axis):
                    yield (key,), group
            else:
                # Otherwise use the parent impl
                yield from self.__wrapped__.__iter__()

    # Variant 3 (no warning)
    groups = GroupbyFutureWrapper(data.groupby(['model']))
    for group_key, group in groups:
        print(f'group_key={group_key}')
        print(group)
        break


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/mwe/pandas_groupby_issue.py
    """
    main()
