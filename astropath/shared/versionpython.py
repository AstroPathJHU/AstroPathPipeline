import astropath.utilities.version

def version_py():
    """
    Check ability to use Python in Conda environment
    >> version_py()

    """
    
    print(astropath.utilities.version.astropathversion)

if __name__ == "__main__":
    version_py()
