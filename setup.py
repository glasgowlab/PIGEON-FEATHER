from setuptools import setup, find_packages

setup(
    name="pigeon_feather",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'pyopenms',
        'MDAnalysis',
        'scikit-learn'
    ],
    
    package_data={
        'pigeon_feather': ['lib/*.*'],
    },

    entry_points={
        'console_scripts': [
            'pigeon=pigeon_feather.pigeon:main',
        ],
    },

)
