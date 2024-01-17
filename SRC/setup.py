from setuptools import setup, find_packages

setup(
    name='CV-RCNN',
    version='1.0.0',
    author='Alber',
    description='A sample Python project',
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "torch",
        "torchvision",
        "matplotlib",
        "pillow",
        "icecream",
        "boto3",
        "opencv-python",
        # other dependencies
    ]
)