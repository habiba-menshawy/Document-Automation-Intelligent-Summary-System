"""
SQLite database implementation.
Follows Single Responsibility Principle - handles only SQLite operations.
"""

import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path
from config import SQLITE_DB_PATH
from logger.logger_config import Logger
log = Logger.get_logger(__name__)


class SQLiteDB():
    """
    SQLite database implementation with connection pooling and error handling.
    Implements relational database interface.
    """
    
    def __init__(self, db_path: str = "documents.db"):
        """
        Initialize SQLite database.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = SQLITE_DB_PATH or db_path
        self._connection = None
        self._in_transaction = False
        
        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        if db_dir != Path('.'):
            db_dir.mkdir(parents=True, exist_ok=True)
            self.create_tables()

        log.info(f"SQLite database initialized at: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        Ensures proper connection handling.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except sqlite3.Error as e:
            log.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn and not self._in_transaction:
                conn.close()
    
    def _get_cursor(self):
        """Get a cursor for transaction operations."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection.cursor()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """
        Execute a SELECT query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            
        Returns:
            List of result rows.
            
        Raises:
            sqlite3.Error: If query execution fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                results = cursor.fetchall()
                
                log.debug(f"Query executed: {query[:100]}... returned {len(results)} rows")
                return results
                
        except sqlite3.Error as e:
            log.error(f"Error executing query: {e}")
            log.error(f"Query: {query}")
            raise
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            
        Returns:
            Number of affected rows.
            
        Raises:
            sqlite3.Error: If query execution fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                affected = cursor.rowcount
                
                log.debug(f"Update executed: {query[:100]}... affected {affected} rows")
                return affected
                
        except sqlite3.Error as e:
            log.error(f"Error executing update: {e}")
            log.error(f"Query: {query}")
            raise
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query string.
            params_list: List of parameter tuples.
            
        Returns:
            Number of affected rows.
            
        Raises:
            sqlite3.Error: If execution fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                affected = cursor.rowcount
                
                log.debug(f"Batch executed: {query[:100]}... affected {affected} rows")
                return affected
                
        except sqlite3.Error as e:
            log.error(f"Error executing batch: {e}")
            log.error(f"Query: {query}")
            raise
    
    def begin_transaction(self) -> None:
        """Begin a transaction."""
        if self._in_transaction:
            log.warning("Transaction already in progress")
            return
        
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._in_transaction = True
        log.debug("Transaction started")
    
    def commit(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            log.warning("No transaction to commit")
            return
        
        try:
            if self._connection:
                self._connection.commit()
                log.debug("Transaction committed")
        except sqlite3.Error as e:
            log.error(f"Error committing transaction: {e}")
            raise
        finally:
            self._cleanup_transaction()
    
    def rollback(self) -> None:
        """Rollback the current transaction."""
        if not self._in_transaction:
            log.warning("No transaction to rollback")
            return
        
        try:
            if self._connection:
                self._connection.rollback()
                log.debug("Transaction rolled back")
        except sqlite3.Error as e:
            log.error(f"Error rolling back transaction: {e}")
            raise
        finally:
            self._cleanup_transaction()
    
    def _cleanup_transaction(self) -> None:
        """Clean up transaction resources."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._in_transaction = False
    
    def create_tables(self) -> None:
        """
        Create database tables with proper schema.
        
        Raises:
            sqlite3.Error: If table creation fails.
        """
        try:
            # Documents table
            # where primary key here is the primary key from milvus
            self.execute_update("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,   
                    filename TEXT NOT NULL,
                    summary TEXT,
                    date TEXT,
                    classification TEXT,
                    anomalies TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Entities table
            self.execute_update("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_pos INTEGER,
                    end_pos INTEGER,
                    confidence REAL,
                    method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            log.info("Database tables created successfully")
            
        except sqlite3.Error as e:
            log.error(f"Error creating tables: {e}")
            raise
    
    def get_table_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get information about all tables.
        
        Returns:
            Dictionary with table information.
        """
        try:
            # Get all tables
            tables_query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            tables = self.execute_query(tables_query)
            
            table_info = {}
            for table_row in tables:
                table_name = table_row[0]
                
                # Get column info
                columns_query = f"PRAGMA table_info({table_name})"
                columns = self.execute_query(columns_query)
                
                # Get row count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                count = self.execute_query(count_query)[0][0]
                
                table_info[table_name] = {
                    'columns': [dict(col) for col in columns],
                    'row_count': count
                }
            
            log.info(f"Retrieved info for {len(table_info)} tables")
            return table_info
            
        except sqlite3.Error as e:
            log.error(f"Error getting table info: {e}")
            raise
    
    def vacuum(self) -> None:
        """Optimize database by reclaiming space."""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                log.info("Database vacuumed successfully")
        except sqlite3.Error as e:
            log.error(f"Error vacuuming database: {e}")
            raise
    
    def get_database_size(self) -> int:
        """
        Get database file size in bytes.
        
        Returns:
            Database size in bytes.
        """
        try:
            db_path = Path(self.db_path)
            if db_path.exists():
                size = db_path.stat().st_size
                log.debug(f"Database size: {size} bytes")
                return size
            return 0
        except Exception as e:
            log.error(f"Error getting database size: {e}")
            raise

if __name__ == "__main__":
    # Initialize database
    db = SQLiteDB()

    # Ensure tables are created
    db.create_tables()
    
    # Insert dummy document
    doc_query = """
        INSERT INTO documents (id, filename, summary, date, classification, anomalies)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    doc_data = (1, "example_doc.txt", "This is a test summary.", "2025-10-31", "Report", "None")
    db.execute_update(doc_query, doc_data)
    print("Inserted dummy document")

    # Insert dummy entities linked to the document
    entity_query = """
        INSERT INTO entities (document_id, label, text, start_pos, end_pos, confidence, method)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    entity_data = [
        (1, "Person", "John Doe", 0, 7, 0.99, "OCR"),
        (1, "Date", "2025-10-31", 10, 20, 0.95, "OCR")
    ]
    db.execute_many(entity_query, entity_data)
    print("Inserted dummy entities")

    # Fetch and print table info
    info = db.get_table_info()
    for table, details in info.items():
        print(f"Table: {table}")
        print(f"Columns: {details['columns']}")
        print(f"Row count: {details['row_count']}")
