from setuptools import setup

setup(
    name="optix", # The name of your package
    version="0.1", # The version of your package
    description="A training speedup package for Stable Diffusion models", # A short description of your package
    install_requires=["torch"], # A list of packages that your package depends on
    packages=["optix"], # A list of subpackages that your package contains
    # You can add more parameters as needed, such as license, author, url, etc.
)