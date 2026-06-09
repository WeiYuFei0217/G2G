from setuptools import setup, find_packages

setup(
    name="g2g",
    version="1.0.0",
    description="G2G: Exploiting Intra-Group Geometry for Inter-Group Pose Estimation",
    packages=find_packages(include=["g2g", "g2g.*"]),
    python_requires=">=3.10",
)
