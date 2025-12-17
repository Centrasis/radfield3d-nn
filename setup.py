from setuptools import setup, find_packages, Distribution
import os


if os.environ.get("CI_COMMIT_TAG") is not None: # Gitlab CI
    version = os.environ.get("CI_COMMIT_TAG")
elif os.environ.get("CI_COMMIT_REF_NAME") is not None: # Gitlab CI
    version = os.environ.get("CI_COMMIT_REF_NAME")
elif os.environ.get("GITHUB_REF") is not None: # Github Actions
    version = os.environ.get("GITHUB_REF").split("/")[-1]
else:
    version = "0.0.0"


setup(
   name='radfield3d-nn',
   version=version,
   install_requires=[
        "rich>=14.1.0",
        "RadFiled3D>=1.2.3",
        "torch>=2.9.1",
        "torchvision",
        "numpy>=2.0.0",
        "lightning>=2.5.6",
        "pandas>=2.3.3"
   ],
   packages=find_packages(),
   author="Felix Lehner",
   author_email="felix.lehner@ptb.de",
   python_requires='>=3.12',
   license=open(os.path.join(os.path.dirname(__file__), "LICENSE")).read(),
   description="Implementation of neural networks for estimating spatially resolved radiation fields. The training pipeline is based on RadFiled3D datasets.",
   long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
   long_description_content_type="text/markdown"
)
