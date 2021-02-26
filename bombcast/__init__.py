from pathlib import Path

import toml

# Read local `credentials.toml` file.
credentials = toml.load(Path(__file__).parents[0] / "config" / "credentials.toml")

# build postgresql URI
username = credentials["postgresql"]["username"]
password = credentials["postgresql"]["password"]
ip_address = credentials["postgresql"]["ip_address"]
port = credentials["postgresql"]["port"]
database = credentials["postgresql"]["database"]
postgresql_uri = f"postgres+psycopg2://{username}:{password}@{ip_address}:{port}/{database}"

config = {}
config["postgresql"] = {
    "uri": postgresql_uri,
}
