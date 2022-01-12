<p>
<details>
<summary>Merge request template for feature branches. [click to expand]</summary>

See https://semver.org for versioning information:
>    Given a version number MAJOR.MINOR.PATCH, increment the:
>
>        MAJOR version when you make incompatible API changes,
>        MINOR version when you add functionality in a backwards compatible manner, and
>        PATCH version when you make backwards compatible bug fixes.

Major version number is 0 for no expectation of backwards comatibility.

"feature" template should be used for significant features that increment WATCH's minor version number.
(Ex. 0.5.0 -> 0.6.0)

"patch" template should be used for bugfixes or minor features that increment WATCH's patch version number.
(Ex. 0.5.0 -> 0.5.1)
</details>
</p>

## Short Description


## Related MRs, Issues, and Other Links


## Types of Changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Documentation/tests
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change


## Changelog

### Fixed

### Added

### Changed


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
- [ ] This branch is rebased on master               [maintainer]
- [ ] `watch/__init__.py:__version__` is incremented [maintainer]

