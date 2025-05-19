from setuptools import setup, find_packages

setup(
    name='microvasc_review',
    version='0.1.0',
    description='Review of Segmentation and Skeletonization Methods for Large-Scale Microvascular Networks',
    author='Helia Goharbavang',
    license='MIT',
    packages=find_packages(include=[
        'implementations', 'implementations.*',
        'manage_data',    'manage_data.*',
        'metrics',        'metrics.*',
        'optimization',   'optimization.*',
    ]),
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'scikit-image>=0.19',
        'scikit-fmm>=2021.4',
        'opencv-python>=4.5',
        'pynrrd>=0.4',
        'SimpleITK>=2.0'
    ],
    entry_points={
        'console_scripts': [
            'binarize=implementations.Binarize:main',
            'skeleton=implementations.Skeleton:main'
        ]
    },
    python_requires='>=3.7',
)
