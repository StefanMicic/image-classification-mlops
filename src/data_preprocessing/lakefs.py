import configparser
import os
from typing import Dict, List

import lakefs_client
from lakefs_client import ApiException, models
from lakefs_client.client import LakeFSClient


class LakeFSManipulator:
    """A class for git-like actions for data versioning."""

    def __init__(self, configuration_path: str = "/home/stefan/Documents/HTEC/MLOps/config.ini"):
        """
        Creates and instance with configuration for lakeFS data versioning.

        Args:
            configuration_path: Path to configuration for lakeFS.
        """
        config = configparser.ConfigParser()
        config.read(configuration_path)
        configuration = lakefs_client.Configuration()
        configuration.username = config['LAKEFS']['username']
        configuration.password = config['LAKEFS']['password']
        configuration.host = config['LAKEFS']['host']

        self.client = LakeFSClient(configuration)

    def create_branch(self, repository: str = "example", branch: str = "branch_new", source: str = "main") -> None:
        """
        Creates new branch if it not exists.

        Args:
            repository: Name of the repository.
            branch: Name of the new repository.
            source: Name of the source branch from which new one will be created.
        """
        try:
            self.client.branches.create_branch(repository=repository,
                                               branch_creation=models.BranchCreation(name=branch, source=source))
        except ApiException:
            print("Branch with tht name already exist")

    def add_files_from_dir(self, directory: str, classes: List[str], repository: str = "example",
                           branch: str = "branch") -> None:
        """
        Stages changes. Something like git add command.

        Args:
            directory: Path to dataset.
            classes: List of class names.
            repository: Name of the repository.
            branch: Name of the new repository.
        """
        for class_name in classes:
            for file in os.listdir(f"{directory}/{class_name}"):
                file_path = f"{directory}/{class_name}/{file}"
                with open(file_path, 'rb') as f:
                    self.client.objects.upload_object(repository=repository, branch=branch,
                                                      path=file_path, content=f)

    def commit_changes(self, repository: str = "example", branch: str = "branch", message: str = "commit message",
                       metadata: Dict[str, str] = None) -> None:
        """
        Commits changes.

        Args:
            repository: Name of the repository.
            branch: Name of the new repository.
            message: Commit message.
            metadata: Metadata of commit. It can be used for useful information such as version, date etc.
        """
        if metadata is None:
            metadata = {}
        self.client.commits.commit(
            repository=repository, branch=branch,
            commit_creation=models.CommitCreation(message=message, metadata=metadata))

    def merge(self, repository: str = "example", branch: str = "branch", destination: str = "main") -> None:
        """
        Merge branch with destination branch.

        Args:
            repository: Name of the repository.
            branch: Name of the new repository.
            destination: Name of the destination branch.
        """
        self.client.refs.merge_into_branch(repository=repository, source_ref=branch,
                                           destination_branch=destination)
