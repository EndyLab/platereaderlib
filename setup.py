from setuptools import setup

setup(
    name='platereaderlib',
    description='Functions for working with data from scientific plate readers.',
    author='Anton Jackson-Smith',
    author_email='acjs@stanford.edu',
    version='1',
    modules=['platereader.py'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
    ],
)
