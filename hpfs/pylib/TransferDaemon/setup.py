import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transfer-daemon-pkg-sigfredo",
    version="0.01.0001",
    author="Sigfredo Soto-Diaz",
    author_email="ssotodi1@jhmi.edu",
    description="Launches transfer for clincal specimen slides in the Astropath pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AstropathJHU/AstropathPipeline/tree/transfer_astroid/pylib",
    install_requires=[
        'pandas', 'joblib', 'traceback2',
        'argparse', 'numpy', 'pathlib', 'lxml'
    ],
    packages=setuptools.find_packages(),  # imports packages with '__init__.py' file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ",
        "Operating System :: MS Windows",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'transfer-daemon = TransferDaemon.Daemon:launch_transfer'
        ]
    }
)
