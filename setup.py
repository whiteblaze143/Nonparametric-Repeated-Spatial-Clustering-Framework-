from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="proust-spatial",
    version="0.2.0",
    author="Jianing Yao, Mithun M, Rajitha Senanayake",
    author_email="jyao37@jhmi.edu",
    description="A framework for spatial domain detection in spatial multi-omics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yaojianing/proust",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 