from setuptools import setup, find_packages

setup(
    name='thor',
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torcheval',
        'torch',
        'numpy',
        'wandb',
        'pillow',
        'matplotlib',
        'tqdm',]
)