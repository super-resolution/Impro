#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='impro',
      version='1.0',
      description='Image analysis package',
      author='Sebastian Reinhard',
      author_email='sebastian.reinhard@stud-mail.uni-wuerzburg.de',
      url='https://github.com/super-resolution',
      packages=find_packages(),
      package_data={'impro.render': ['shaders/*','shaders/raycast/*'], 'impro.analysis': ['cuda_files/*']},
      include_package_data=True,
     )