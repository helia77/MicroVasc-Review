from setuptools import setup, find_packages

setup(
    name='manage_data',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pynrrd',
        'numpy',
        'simpleitk',
        'scikit-image'
    ],

    entry_points={
        'console_scripts': [
            'manage_data = manage_data.manage_data:main',
        ],
    },
    author='Helia Gohar',
    description='A package for managing and processing image and volume data.',
    license='MIT',
    url='https://github.com/helia77/MicroVasc-Review/manage_data',
)