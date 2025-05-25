#!/usr/bin/env python3
"""
SubGraphRAG+ Setup Configuration
"""

from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "SubGraphRAG+ - Advanced knowledge graph retrieval system"

setup(
    name="subgraphrag-plus",
    version="1.0.0",
    author="SubGraphRAG+ Team",
    author_email="contact@subgraphragplus.com",
    description="Advanced knowledge graph retrieval system with intelligent subgraph extraction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/clarkandrew/SubgraphRAGPlus",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies will be installed via requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
) 