<!--
    Merge request template for feature branches.

    Should be used for significant features that increment WATCH's version number.
    (Ex. 1.5.0 -> 1.6.0)
    Major version number is 1.0 for Phase 1.
    Patch version number is not being tracked.
-->
## Short Description


## Related MRs, Issues, and Other Links


## Types of Changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Documentation/tests
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change


## Changelog


## Checklist

```bash
# should be run before merging:
cd watch/
./run_developer_setup.sh  # make sure you're in a virtualenv
python dev/lint.py directory_i_changed/ [--mode=apply to fix]
python run_tests.py
```
- [ ] New code is covered by tests
- [ ] New code is documented
- [ ] Linting passes
- [ ] Tests pass
- [ ] This branch is rebased on master
- [ ] `watch/__init__.py:__version__` is incremented

