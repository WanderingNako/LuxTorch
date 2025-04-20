from setuptools import setup, find_packages

setup(
    name='luxtorch',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    author='Lu Xin',
    description='A minimal deep learning framework',
    python_requires='>=3.7',
)
