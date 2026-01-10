import os
from git import Repo

def clone_repo(github_url, dest_folder='source_repo'):
    if os.path.exists(dest_folder):
        import shutil
        shutil.rmtree(dest_folder)
    Repo.clone_from(github_url, dest_folder)
