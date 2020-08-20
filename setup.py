from setuptools import setup, find_packages
setup(
    name="Titanic_Binary_CLassifier",
    version="0.1",
    packages=find_packages(),
    scripts=["Model_Code/Binary_Network.py"],

    install_requires=["docutils>=0.3", "torch==1.5.0", "pandas==1.0.3", "matplotlib==3.2.1"],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"]
    },

    # metadata to display on PyPI
    author="Christopher Burton",
    author_email="chrisburton279@gmail.com",
    description="Simple binary classifier for Kaggle Titanic dataset"

)