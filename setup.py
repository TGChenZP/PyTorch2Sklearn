from setuptools import setup, find_packages

setup(
    name="PyTorch2Sklearn",  # Replace with your package name
    version="0.0.3",
    author="Lang Chen",
    author_email="ronchen6666@gmail.com",
    description="Refactoring PyTorch models into sklearn-like API",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    # Replace with your GitHub repo
    url="https://github.com/TGChenZP/PyTorch2Sklearn",
    packages=find_packages(),  # Automatically find all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
