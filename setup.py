#!/usr/bin/env python

"""Setup ASN installation."""

import re
from os import path as op

from setuptools import find_packages, setup


def _read(f): return open(
    op.join(op.dirname(__file__), f)).read() if op.exists(f) else ''


_meta = _read('asn/__init__.py')
_license = re.search(r'^__license__\s*=\s*"(.*)"', _meta, re.M).group(1)
_project = re.search(r'^__project__\s*=\s*"(.*)"', _meta, re.M).group(1)
_version = re.search(r'^__version__\s*=\s*"(.*)"', _meta, re.M).group(1)
_author = re.search(r'^__author__\s*=\s*"(.*)"', _meta, re.M).group(1)
_mail = re.search(r'^__email__\s*=\s*"(.*)"', _meta, re.M).group(1)

install_requires = [l for l in _read('requirements.txt').split('\n')
                    if l and not l.startswith('#') and not l.startswith('-')]

meta = dict(
    name=_project,
    version=_version,
    license=_license,
    description='Adversarial Skill Networks implementation in pytorch',
    platforms=('Any'),
    zip_safe=False,

    keywords='pytorch asn'.split(),

    author=_author,
    author_email=_mail,
    url=' https://github.com/mees/Adversarial-Skill-Networks',

    packages=find_packages(exclude=["tests"]),

    install_requires=install_requires,
)

if __name__ == "__main__":
    print("find_packag", find_packages(exclude=["tests"]))
    print()
    print()
    setup(**meta)
