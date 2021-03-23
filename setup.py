# Header
import os, sys
pj = os.path.join

name = 'Fruithunter'
description = 'Fruithunter pipeline.'
long_description= 'PyQGLViewer is a set of Python bindings for the libQGLViewer C++ library which extends the Qt framework with widgets and tools that eases the creation of OpenGL 3D viewers.'
authors = 'Simon Artzet, Juan Pablo Rojas Bustos, Frédéric Boudon'
authors_email = 'frederic.boudon@cirad.fr'
url= 'https://forgemia.inra.fr/openalea_phenotyping/fruithunter.git'
# LGPL compatible INRIA license
license = 'Cecill-C'

##############
# Setup script




version = '0.1.0'


from setuptools import setup


setup(
    name='fruithunter',
    version=version,
    description=description,
    long_description=long_description,
    author=authors,
    author_email=authors_email,
    url=url,
    license=license,

    # pure python  packages
    packages = [
        'pcl',
        'pcl.build',
        'randlanet',
        'randlanet.utils',
        'notebook'
    ],

    # python packages directory
    package_dir = { '' : '.'},

    package_data={
        "": ['*.pyd', '*.so', '*.dylib', '*.sh'],
    },

    include_package_data = True,
    zip_safe = False,

    entry_points = {
        'console_scripts': ['compute_fpfh = pcl.launcher:main',],
    }

)
