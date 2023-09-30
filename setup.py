from setuptools import setup, find_packages

setup(
    name='AstroCLIP',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'astroclip=astroclip.cli:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your project',
    url='https://github.com/yourusername/astroclip',
)
