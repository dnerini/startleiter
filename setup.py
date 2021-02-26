from setuptools import setup, find_packages

setup(
    name="bombcast",
    version="0.0.1",
    url="https://github.com/dnerini/bombcast.git",
    author="Daniele Nerini",
    author_email="daniele.nerini@gmail.com",
    description="Forecast so-called 'giorni bomba' for paragliding",
    packages=find_packages(),
    install_requires=["beautifulsoup4", "pandas", "psycopg2", "selenium", "sqlalchemy", "toml"],
)