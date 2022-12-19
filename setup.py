#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="hear_mn",
    description="MobileNet pretrained model for HEAR 2021 NeurIPS Competition",
    author="Florian Schmid",
    author_email="florian.schmid@jku.at",
    url="https://github.com/fschmid56/EfficientAT",
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    project_urls={
        "Source Code": "https://github.com/fschmid56/EfficientAT_HEAR",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[]
)
