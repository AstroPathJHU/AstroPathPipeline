import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astropath-pkg-sigfredo",
    version="0.01.0001",
    author="Sigfredo Soto-Diaz",
    author_email="ssotodi1@jhmi.edu",
    description="Runs AstroID generation script",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AstroPathJHU/AstroPathPipeline/tree/hpf-edits/hpfs/ASTgen",
    install_requires=[
        'pandas', 'joblib', 'traceback2', 'openpyxl', 'argparse',
        'argparse', 'numpy', 'pathlib', 'lxml', 'setuptools-scm'
    ],
    packages=setuptools.find_packages(exclude=('setup', 'test')),  # imports packages with '__init__.py' file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ",
        "Operating System :: MS Windows",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'shared_tools = shared_tools.shared_tools:main'
        ]
    }
)
