from setuptools import find_packages, setup

setup(
    name='reliance-lstm-trainer',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.10.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'google-cloud-storage>=2.10.0',
    ],
)
