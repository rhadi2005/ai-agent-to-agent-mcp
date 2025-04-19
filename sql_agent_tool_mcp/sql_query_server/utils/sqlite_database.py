import sqlite3
import os
import requests
from typing import List, Dict, Any, Optional

_DB_INSTANCE = None


class SQLiteDatabase:
    """A simple wrapper for SQLite databases that mimics the interface used in the agent."""

    def __init__(self, db_path: str):
        """Initialize with the path to the SQLite database file."""
        self.db_path = db_path
        self.conn = None

    def _get_connection(self):
        """Get a connection to the database, creating it if necessary."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            # Enable column access by name
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close the database connection if open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get_usable_table_names(self) -> List[str]:
        """Return a list of table names in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def get_table_info_no_throw(self, table_name: str) -> str:
        """Return schema information for the specified table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Check if table exists
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if not cursor.fetchone():
                return f"Error: table {table_name} not found in database"

            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            # Format column information
            column_info = []
            for col in columns:
                col_dict = dict(col)
                column_info.append(f"{col_dict['name']} ({col_dict['type']})")

            return "\n".join(column_info)
        except Exception as e:
            return f"Error getting schema for {table_name}: {str(e)}"

    def run_no_throw(self, query: str) -> str:
        """Execute the query and return the results as a string."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)

            # For non-SELECT queries (INSERT, UPDATE, DELETE)
            if not cursor.description:
                conn.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"

            results = cursor.fetchall()

            if not results:
                return ""

            # Format results as a string table
            # Get column names
            columns = [description[0] for description in cursor.description]

            # Calculate column widths for better formatting
            col_widths = [len(col) for col in columns]
            for row in results:
                row_dict = dict(row)
                for i, col in enumerate(columns):
                    col_widths[i] = max(col_widths[i], len(str(row_dict[col])))

            # Format header
            header = " | ".join(f"{col:{width}}" for col,
                                width in zip(columns, col_widths))
            separator = "-" * len(header)
            output = [header, separator]

            # Format rows
            for row in results:
                row_dict = dict(row)
                formatted_row = [
                    f"{str(row_dict[col]):{width}}" for col, width in zip(columns, col_widths)]
                output.append(" | ".join(formatted_row))

            return "\n".join(output)
        except Exception as e:
            return f"Error: {str(e)}"


def download_database(url: str, file_path: str) -> bool:
    """Download a database file from a URL if it doesn't exist locally."""
    if os.path.exists(file_path):
        print(f"Database file already exists: {file_path}")
        return True

    try:
        print(f"Downloading database from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Database file downloaded: {file_path}")
            return True
        else:
            print(
                f"Failed to download database. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading database: {str(e)}")
        return False


def setup_database(db_file: str = "Chinook.db") -> Optional[SQLiteDatabase]:
    """Download and setup the Chinook SQLite database."""
    global _DB_INSTANCE

    # Return the existing instance if already set up
    if _DB_INSTANCE is not None:
        return _DB_INSTANCE

    try:
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
        if download_database(url, db_file):
            _DB_INSTANCE = SQLiteDatabase(db_file)
            print(f"Database instance created successfully: {db_file}")
            return _DB_INSTANCE
        return None
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return None


# Initialize the global database instance when module is imported
_DB_INSTANCE = setup_database()


def get_db_instance() -> Optional[SQLiteDatabase]:
    """Get the global database instance, initializing it if necessary."""
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = setup_database()
    return _DB_INSTANCE
