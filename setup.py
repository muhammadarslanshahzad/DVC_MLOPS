from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__  = "0.0.1"

REPO_NAME = "bg_removal"
AUTHOR_USER_NAME = "muhammadarslanshahzad"
SRC_REPO = 'bgremove'
AUTHOR_EMAIL = 'EMAIL HERE'

setup(
    name=SRC_REPO, 
    version=__version__, 
    author=AUTHOR_USER_NAME, 
    author_email=AUTHOR_EMAIL, 
    description="A small python app for removing the BG",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}", 
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues", 
    }, 
    package_dir={"": "src"}, 
    packages=find_packages(where="src")
)

