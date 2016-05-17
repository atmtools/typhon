# -*- coding: utf-8 -*-


def get_version_info():
    VERSION = '0.3.0'
    # Add revision number for development versions
    DEVBUILD = False

    if DEVBUILD:
        from os import path
        if path.exists('.svn'):
            revision = get_svn_revision()
        else:
            revision = "dev"
        VERSION += '+' + revision

    return VERSION


def get_svn_revision():
    def _minimal_ext_cmd(cmd):
        from subprocess import Popen, PIPE
        from os import environ
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = Popen(cmd, stdout=PIPE, env=env).communicate()[0]
        return out

    try:
        from re import findall
        out = _minimal_ext_cmd(['svn', 'info'])
        revision = 'r' + findall('Revision: ([0-9]+)', out.decode('ascii'))[0]
    except (OSError, IndexError):
        revision = "dev"

    return revision


__version__ = get_version_info()
