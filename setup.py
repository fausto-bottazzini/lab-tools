from setuptools import setup, find_packages

setup(
    name = 'settings',
    version = '1.0.2',
    description = 'Librería personal con funciones matemáticas y utilitarias',
    author = 'Bottazzini',
    packages = find_packages(),  # encuentra automáticamente el paquete "settings"
    include_package_data = True,
    install_requires = [
        "numpy>=1.26.4",
        "matplotlib>=3.9.2",
        "scipy>=1.13.1",
        "sympy>=1.13.2",
    ],
    python_requires = ">=3.8"
)
