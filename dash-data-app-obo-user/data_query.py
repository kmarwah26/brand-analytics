import pandas as pd
from databricks import sql
from databricks.sdk import WorkspaceClient
from config import DATABRICKS_CONFIG

# Create a workspace client for authentication
workspace = WorkspaceClient(
    host=DATABRICKS_CONFIG['host'],
    client_id=DATABRICKS_CONFIG['client_id'],
    client_secret=DATABRICKS_CONFIG['client_secret'],
    auth_type='oauth'  # Explicitly set auth type to OAuth
)

def sql_query_with_service_principal(query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=DATABRICKS_CONFIG['host'],
        http_path=f"/sql/1.0/warehouses/{DATABRICKS_CONFIG['warehouse_id']}",
        credentials_strategy=workspace  # Use the workspace client for authentication
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

def sql_query_with_user_token(query: str, user_token: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=DATABRICKS_CONFIG['host'],
        http_path=f"/sql/1.0/warehouses/{DATABRICKS_CONFIG['warehouse_id']}",
        access_token=user_token  # Pass the user token into the SQL connect to query on behalf of user
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas() 