from setuptools import setup, find_packages

setup(
    name='sr4rs',
    version='0.0.1',
    url='https://github.com/aoyono/sr4rs',
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "git+ssh://git@github.com/remicres/otbtf.git#egg=otbtf",
    ],
    entry_points={
        "console_scripts": [
            "sr4rs-lo2hi = sr4rs.sr:cli",
        ],
    },
)
