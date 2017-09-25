# Contributing to typhon
Thank you for considering to contribute to the typhon project! :tada: :+1:

As a community-driven project we rely on the work and contribution of
volunteers. By participating in this project, you agree to abide by the
[code of conduct](CODE_OF_CONDUCT.md).

In order to contribute to typhon you need to be registered on
[GitHub](https://github.com/join).

## Checklist
This checklist also serves as table of contents.
1. [Fork the project](#fork-the-project)
2. [Clone your fork](#clone-the-forked-project)
3. [Set up your fork](#set-up-cloned-fork)
4. [Create a branch](#create-a-branch)
5. [Work on your contribution](#work-on-your-contribution)
6. [Pull request](#pull-request)
7. [Code review](#code-review)
8. [Follow up](#follow-up)

## Fork the project
The first step is to fork the typhon project. This is done by clicking the Fork
button on the project site.

## Clone the forked project
After forking the project you can clone the project to create a local working
copy.
```bash
$ git clone https://github.com/YOUR_USERNAME/typhon.git
```

You can also connect to GitHub
[using SSH](https://help.github.com/articles/connecting-to-github-with-ssh/).

## Set up cloned fork
Add the URL of the original project to your local repository to be able to pull
changes from it:
```bash
$ git remote add upstream https://github.com/atmtools/typhon.git
```

Listing the remote repositories will show something like:
```bash
$ git remote -v
origin https://github.com/YOUR_USERNAME/typhon.git (fetch)
origin https://github.com/YOUR_USERNAME/typhon.git (push)
upstream https://github.com/atmtools/typhon.git (fetch)
upstream https://github.com/atmtools/typhon.git (push)
```

Make sure to pull in changes from the upstream master branch at regular
intervals to keep track of changes done to the project.  We recommend to use
the `--rebase` flag to maintain a linear history.
```bash
$ git pull --rebase upstream master
```

## Create a branch
Before starting to work on your feature or bugfix you need to create a local
branch where to keep all your work. Branches help to organize the changes
related to different developments in the project.

You can do that with the following git command:
```bash
$ git checkout -b BRANCH_NAME
```

This will create a new branch and will make it the active one in your local
repository. Be sure to use a descriptive name for the branch name.

You can check you are in the right branch using git:
```bash
$ git branch
  master
* BRANCH_NAME
```
The current active branch is the one with a ``*`` on the left.

## Work on your contribution
Now you can start with the development of your new feature (or bug fix).
During the work you can use git's commit and push mechanism to save and track
your changes.

You can push those changes to your personal fork.
```bash
$ git push origin BRANCH_NAME
```

## Pull request
After pushing your changes to your fork navigate to the GitHub page of your
work.  Click on the Pull request button. Add the needed information to the web
form and submit your request.

## Code review
The developer team will review your changes and decide whether to accept your
changes. This process might include some discussion or even further changes to
the code (this is the reason why [branches](#create-a-branch) are important).

## Follow up
After your contribution has been merged to the main project (or rejected) you
can delete the branch you used for it.

To delete the branch in your local repository and on GitHub:
```bash
$ git branch -D BRANCH_NAME
$ git push origin --delete BRANCH_NAME
```

# Credits
This file is closely based on a
[blog post](http://blog.davidecoppola.com/2016/11/howto-contribute-to-open-source-project-on-github/)
by Davide Coppola
