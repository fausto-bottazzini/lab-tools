from setuptools import setup, find_packages

setup(
    name='settings',
    version='0.1.4',
    description='Librería personal con funciones matemáticas y utilitarias',
    author='Bottazzini',
    packages=find_packages(),  # encuentra automáticamente el paquete "settings"
    include_package_data=True,
    install_requires=[],
)