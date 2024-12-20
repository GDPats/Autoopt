from setuptools import setup, find_packages

setup(
    name="autoopt",  # Name of your package
    version="0.0.1",  # Initial version
    author="Alexandros S. Kalafatelis, Gerasimos Patsourakis, Vasilis Nikolakakis",
    author_email="alkalafatelis@fourdotinfinity.com", " gpatsourakis@fourdotinfinity.com", "vnik@fourdotinfinity.com",
    description="A library for automated model tuning and data preprocessing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown", 
    url="https://github.com/GDPats/Autoopt",
    packages=find_packages(), 
    install_requires=[
        "scikit-learn>=0.24.0", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
