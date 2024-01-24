from setuptools import find_packages, setup

def get_requirements(path):
    requirements = []
    with open(path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirements.replace("\n","")]

        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements

setup(
    name='SMS_SPAM_DETECTION',
    version='0.0.1',
    author='SJR',
    author_email='shnvklm2004@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)