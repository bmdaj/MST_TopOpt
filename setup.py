from setuptools import setup, find_packages

setup(
    name="MST_TopOpt",                # Package name
    version="0.1.0",                  # Version
    author="<Beñat Martinez de Aguirre Jokisch",      # Author's name
    author_email="bmdaj13@gmail.com", # Author's email
    description="A package to solve topology optimization problems that target optica forcesl based on the Maxwell Stress Tensor formalism.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bmdaj/MST_TopOpt",  # Repository URL
    packages=find_packages(),         # Automatically find sub-packages
    install_requires=[],              # List dependencies here or in requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",          # Minimum Python version
)