from pathlib import Path

import toml

config = toml.load(Path(__file__).parents[0] / "config" / "config.toml")
credentials = toml.load(Path(__file__).parents[0] / "config" / "credentials.toml")

# build postgresql URI
username = credentials["postgresql"]["username"]
password = credentials["postgresql"]["password"]
ip_address = credentials["postgresql"]["ip_address"]
port = credentials["postgresql"]["port"]
database = credentials["postgresql"]["database"]
postgresql_uri = (
    f"postgresql+psycopg2://{username}:{password}@{ip_address}:{port}/{database}"
)

config["postgresql"] = {
    "uri": postgresql_uri,
}
config["netcdf"] = {
    "repo": credentials["netcdf"]["repo"],
}
