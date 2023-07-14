from setuptools import setup, find_packages

setup(
    name='sr4rs',
    version='0.0.1',
    url='https://github.com/aoyono/sr4rs',
    packages=find_packages(),
    install_requires=[
        "tensorflow>=1.15",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "sr4rs-lo2hi = sr4rs.sr:cli",
        ],
    },
)
