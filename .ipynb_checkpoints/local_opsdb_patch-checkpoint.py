import sqlite3
from pathlib import Path

class OpsimDatabase:
    def __init__(self, database_filename):
        if not Path(database_filename).exists():
            raise FileNotFoundError(f"Could not find database: {database_filename}")
        self.conn = sqlite3.connect(database_filename)

    def get_connection(self):
        return self.conn
