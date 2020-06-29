import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(name='tf_kde',
      version='0.1',
      description='A one-dimensional Kernel Density Estimation implemented in TensorFlow',
      url='http://github.com/AstroViking/tf-kde',
      author='Marc Steiner',
      author_email='astroviking@protonmail.com',
      license='WTFPL',
      packages=['tf_kde'],
      install_requires=requirements,
      zip_safe=False)
