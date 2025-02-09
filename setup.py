import setuptools
setuptools.setup(
name="multilingua_ocr",
use_scm_version=True,
author="Mossab Ibrahim",
author_email="ventura@cs.byu.edu",
description="Framework for Spanish-Indigenous historical document OCR",
long_description=open("README.md").read(),
long_description_content_type="text/markdown",
url="https://github.com/multilingua-ocr/framework",
packages=setuptools.find_packages(),
classifiers=[
"Development Status :: 4 - Beta",
"Intended Audience :: Science/Research",
"License :: OSI Approved :: MIT License",
"Programming Language :: Python :: 3",
"Programming Language :: Python :: 3.8",
"Programming Language :: Python :: 3.9",
"Topic :: Scientific/Engineering :: Artificial Intelligence",
],
python_requires=">=3.8",
install_requires=[
"torch>=2.0.0",
"torchvision>=0.15.0",
"transformers>=4.28.0",
"Pillow>=9.5.0",
"opencv-python>=4.7.0.72",
"numpy>=1.24.3",
"pandas>=2.0.1",
"PyYAML>=6.0",
],
extras_require={
"dev": [
"pytest>=7.3.1",
"pytest-cov>=4.0.0",
"black>=22.3.0",
"isort>=5.10.1",
"flake8>=4.0.1",
"mypy>=0.981",
]
},
)
