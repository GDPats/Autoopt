from setuptools import setup, find_packages

setup(
    name="autoopt",  # Name of your package
    version="0.0.1",  # Initial version
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for automated model tuning and data preprocessing.",
    long_description=open("README.md").read(),  # You can create a README.md file to add detailed info
    long_description_content_type="text/markdown",  # Markdown for the README
    url="https://github.com/yourusername/autoopt",  # Replace with your GitHub repo if available
    packages=find_packages(),  # Automatically finds sub-packages
    install_requires=[
        "scikit-learn>=0.24.0",  # Dependency for the functions we’ll be using
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
