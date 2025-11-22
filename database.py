import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def setup_database():
    database = "activity.db"

    sql_create_logs_table = """ CREATE TABLE IF NOT EXISTS logs (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                        active_app_name text NOT NULL,
                                        active_window_title text NOT NULL,
                                        screenshot_path text,
                                        ocr_text text,
                                        embedding_json text
                                    ); """

    sql_create_summaries_table = """ CREATE TABLE IF NOT EXISTS summaries (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                        summary_text text NOT NULL
                                    ); """

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create logs table
        create_table(conn, sql_create_logs_table)
        # create summaries table
        create_table(conn, sql_create_summaries_table)
        conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    setup_database()
