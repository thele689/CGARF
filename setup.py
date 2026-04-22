from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="cgarf",
    version="0.1.0",
    author="CGARF Contributors",
    description="Causality-Guided Automated Program Repair Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-account>/CGARF",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "requests>=2.28.0",
        "openai>=1.0.0",
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
        "transformers>=4.25.0",
        "sentence-transformers>=2.2.0",
        "pydantic>=1.10.0",
        "dataclasses-json>=0.5.0",
        "tqdm>=4.64.0",
        "loguru>=0.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
