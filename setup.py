from setuptools import setup, find_packages

setup(
    name='settings',
    version='1.0.0',
    description='Librería personal con funciones matemáticas y utilitarias',
    author='Bottazzini',
    packages=find_packages(),  # encuentra automáticamente el paquete "settings"
    include_package_data=True,
    install_requires=[],
)