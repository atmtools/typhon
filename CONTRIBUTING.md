# Contributing to typhon
In order to contribute to typhon you need to be registered on
[GitHub](https://github.com/join).

## Fork the project
The first step is to fork the typhon project. This is done by clicking the Fork
button on the project site.

## Clone the forked project
After forking the project you can clone the project to create a local working
copy.
```bash
git clone https://github.com/YOUR_USERNAME/typhon.git
```

## Set up cloned fork
Add the URL of the original project to your local repository to be able to pull
changes from it:
```bash
git remote add upstream https://github.com/atmtools/typhon.git
```

Listing the remote repositories will show something like:
```bash
git remote -v
origin https://github.com/YOUR_USERNAME/typhon.git (fetch)
origin https://github.com/YOUR_USERNAME/typhon.git (push)
upstream https://github.com/atmtools/typhon.git (fetch)
upstream https://github.com/atmtools/typhon.git (push)
```

Make sure to pull (or fetch) the upstream master branch at regular intervals to
keep track of changes done to the project.
```bash
git pull upstream
```

## Work on your contribution
Now you can start with the development of your new feature (or bug fix).
During the work you can use git's commit and push mechanism to save and track
your changes.

You can push those changes to your personal fork.
```bash
git push origin/master
```

## Pull request
After pushing your changes to your fork navigate to the GitHub page of your
work.  Click on the Pull request button. Add the needed information to the web
form and submit your request.  The developer team will review your changes and
decide whether to accept your changes.

# Credits
This file is a brief summary of a
[blog post](http://blog.davidecoppola.com/2016/11/howto-contribute-to-open-source-project-on-github/)
by Davide Coppola
