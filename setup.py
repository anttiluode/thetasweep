from setuptools import setup, find_packages

setup(
    name="thetasweep",
    version="0.1.0",
    author="Antti Luode",
    description="Biologically-grounded sequence processing via directional reservoir sweeps",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "llm": ["llama-cpp-python"],
        "dev": ["pytest", "matplotlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
