import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE/"README.md").read_text()

setup(
    name="Titanicbc",
    version="0.0.2",
    packages=find_packages(include=['Titanic_Binary', 'Titanicbc']), #.* and init are interchangable
    #scripts=["Titanicbc/Binary_Network.py"],

    install_requires=["docutils>=0.3", "torch>=1.5.0", "pandas>=1.0.3", "matplotlib>=3.2.1", "psycopg2>=2.8.5"],

    package_data={
        "": ["*.txt", "*.yaml", "*.rst", "*.md", "*.pth"]
    },

    include_package_data = True,

    # metadata to display on PyPI
    author="Christopher Burton",
    author_email="chrisburton279@gmail.com",
    description= "Simple neural network interface including pre-trained model for the Kaggle Titanic dataset",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/chrisb27/Titanic_Binary"

)