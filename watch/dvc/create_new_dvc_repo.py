"""
HIGHLY EXPERIMENTAL

Our goal here is to script the creation of new DVC repos as we need them.

Initially this will start out as just the logic to create one specific new
repo, and we may generalize this or deprecate this later.

Requires:
    # pip install xcookie > 0.3.0
    pip install python-gitlab
"""
import ubelt as ub


def source_bash_script(code, cwd=None, system=False):
    import ubelt as ub
    name = 'script_' + ub.hash_data(code)[0:16]
    dpath = ub.Path.appdir('cmd_queue/bash_scripts').ensuredir()
    fpath = dpath / f'{name}.sh'
    fpath.write_text(code)
    info = ub.cmd(
        f'bash -ci "source {fpath}"', verbose=3, check=True, shell=0, cwd=cwd)
    return info


def new_gitlab_repo(proj_name, proj_group, url, visibility='private'):
    # Depends on the xcookie API, which is currently unstable.
    # See: ~/misc/tests/bash/check_alias_in_script.sh
    from xcookie.vcs_remotes import GitlabRemote
    code = ub.codeblock(
        '''
        #!/bin/bash -i
        # shopt -s expand_aliases
        load_secrets
        HOST=https://gitlab.kitware.com
        export PRIVATE_GITLAB_TOKEN=$(git_token_for "$HOST")
        # Careful, make sure not to print this
        echo "$PRIVATE_GITLAB_TOKEN"
        ''')
    PRIVATE_GITLAB_TOKEN = source_bash_script(code)['out'].strip()

    # Ensure that the repo exists on gitlab
    self = GitlabRemote(proj_name, proj_group, url, visibility=visibility,
                        private_token=PRIVATE_GITLAB_TOKEN)
    self.auth()
    self.new_project()


def create_experiment_dvc_repo():
    url = 'https://gitlab.kitware.com'
    proj_group = 'smart'
    proj_name = 'smart_expt_dvc'
    visibility = 'private'

    # Ensure that the repo exists locally
    DVC_REPOS_DPATH = ub.Path('$HOME/data/dvc-repos').expand()
    repo_dpath = DVC_REPOS_DPATH / proj_name

    git_dpath = repo_dpath / '.git'
    if not git_dpath.exists():
        new_gitlab_repo(proj_name, proj_group, url, visibility)
        git_base = url.replace('https://', 'git@')
        remote_url = f'{git_base}:{proj_group}/{proj_name}.git'
        _ = ub.cmd('git init', cwd=repo_dpath, verbose=2, check=1)
        _ = ub.cmd(f'git remote add origin {remote_url}', cwd=repo_dpath, verbose=2, check=1)

    dvc_dpath = repo_dpath / '.dvc'
    if not dvc_dpath.exists():
        _ = ub.cmd('dvc init', cwd=repo_dpath, verbose=2, check=1)

        source_bash_script(ub.codeblock(
            '''
            set -x
            dvc config core.autostage true
            dvc config core.check_update false
            dvc config core.analytics false
            dvc config core.remote aws
            dvc config cache.shared group
            dvc config cache.type "reflink,symlink,hardlink,copy" # to enable symlinks to avoid copying
            dvc config cache.protected true # to make links RO so that we you don't corrupt them accidentally
            '''), cwd=repo_dpath)

        source_bash_script(ub.codeblock(
            '''
            set -x
            dvc remote add aws s3://kitware-smart-watch-data/dvc
            dvc remote modify aws profile iarpa

            dvc remote add horologic ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc
            '''), cwd=repo_dpath)

        __note__ = """
        If you need to setup a shared cache on a local machine

        dvc cache dir --local /data/dvc-caches/smart_watch_dvc
        """
        __note__


def update_data_ignores():
    import watch
    import ubelt as ub
    ignore_text = ub.codeblock(
        '''
        Aligned-*/*.kwcoco.json
        Aligned-*/*.kwcoco.json.hashid.cache
        Aligned-*/*/subdata.kwcoco.json
        Aligned-*/_*
        Uncropped-*
        ''')

    dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_data')
    fpath1 = dvc_expt_dpath / '.gitignore'
    fpath2 = dvc_expt_dpath / '.dvcignore'
    fpath1.write_text(ignore_text)
    fpath2.write_text(ignore_text)

    ub.cmd(f'git add {fpath1} {fpath2}', cwd=dvc_expt_dpath)


def update_expt_ignores():
    import watch
    import ubelt as ub
    ignore_text = ub.codeblock(
        '''
        training
        emissions.csv
        ''')

    dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt')
    fpath1 = dvc_expt_dpath / '.gitignore'
    fpath2 = dvc_expt_dpath / '.dvcignore'
    fpath1.write_text(ignore_text)
    fpath2.write_text(ignore_text)

    ub.cmd(f'git add {fpath1} {fpath2}', cwd=dvc_expt_dpath)
