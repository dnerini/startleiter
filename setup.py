from setuptools import setup, find_packages

setup(
    name="startleiter",
    version="0.0.1",
    url="https://github.com/dnerini/startleiter.git",
    author="Daniele Nerini",
    author_email="daniele.nerini@gmail.com",
    description="",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4", "matplotlib", "metpy", "numpy", "netcdf4", "pandas>=1.4.2", "psutil", "psycopg2-binary", "requests", "selenium", "sqlalchemy",
        "toml", "fastapi", "uvicorn", "bottleneck", "tensorflow-cpu", "shap", "xarray>=2022.3.0"
    ],
)
