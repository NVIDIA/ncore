.. _installation: 

Wheel Installation
==================

The NRECore-SDK python APIs are provided as ``pip``-installable wheels maintained in the repositories `package registry <https://gitlab-master.nvidia.com/Toronto_DL_Lab/ncore/-/packages>`_.

The wheels are called ``ncore`` and can be installed by specifying the project's index URL as follows::

  pip install ncore --extra-index-url https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/pypi/simple

For authentication, this requires a one-time setup of the gitlab personal 
access token (see `link <https://gitlab-master.nvidia.com/Toronto_DL_Lab/ncore#setup-gitlab-personal-access-token-docker-credentials>`_ for details).

Alternatively, gitlab personal access tokens can be included in the index url via::

    pip install ncore --extra-index-url https://__token__:<GITLAB_TOKEN>@gitlab-master.nvidia.com/api/v4/projects/61004/packages/pypi/simple

Currently, ``ncore`` wheels support all python version starting from ``python3.10``.
