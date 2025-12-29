from setuptools import setup, find_packages

setup(
    name="xplaincam",  
    version="0.1.0",  
    author="",
    author_email="",
    description="A package for CAM evaluation and visualization",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "drop_increase_eval=xplaincam.scripts.drop_increase_evaluation:main",
            "del_inser_eval=xplaincam.scripts.del_inser_evaluation:main",
        ],
    },
)
