#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='impro',
      version='0.9',
      description='Image processing package',
      author='Sebastian Reinhard',
      author_email='sebastian.reinhard@stud-mail.uni-wuerzburg.de',
      url='https://github.com/super-resolution',
      packages=find_packages(),
      package_data={'impro.visualisation': ['shaders/*','shaders/raycast/*']},
      include_package_data=True,
     )