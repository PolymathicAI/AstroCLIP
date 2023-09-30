from setuptools import setup, find_packages

setup(
    name='AstroCLIP',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'lightning',
        'timm',
        'datasets',
        'wandb'
    ],
    author='EiffL',
    description='Cross-Modal Pretraining for Astronomical Images',
    url='https://github.com/yourusername/astroclip',
)
