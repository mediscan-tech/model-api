from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from urllib.request import urlretrieve
import os

MODEL_URL = 'https://mediscan.nyc3.cdn.digitaloceanspaces.com/skin_diseases_model.h5'
MODEL_PATH = os.path.join('models', 'skin_diseases_model.h5')

def download_model():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}")
        urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")

class PostDevelopCommand(develop):
    def run(self):
        download_model()
        develop.run(self)

class PostInstallCommand(install):
    def run(self):
        download_model()
        install.run(self)

setup(
    name='your_project_name',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    # Add other setup parameters as needed
)