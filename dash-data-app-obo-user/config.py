import os
import yaml

def load_yaml_config():
    """Load configuration from app.yaml file."""
    try:
        with open('app.yaml', 'r') as file:
            config = yaml.safe_load(file)
            print("Loaded YAML config:", config)  # Debug print
            env_vars = {item['name']: item['value'] for item in config.get('env', [])}
            print("Extracted env vars:", env_vars)  # Debug print
            return env_vars
    except Exception as e:
        print(f"Error loading app.yaml: {str(e)}")
        return {}

# Load configuration from app.yaml
yaml_config = load_yaml_config()
print("YAML config loaded:", yaml_config)  # Debug print

# Clear ALL existing Databricks environment variables
for key in list(os.environ.keys()):
    if key.startswith('DATABRICKS_'):
        del os.environ[key]

# Set only OAuth-related environment variables
os.environ['DATABRICKS_HOST'] = yaml_config.get('DATABRICKS_HOST', '').rstrip('/')
os.environ['DATABRICKS_CLIENT_ID'] = yaml_config.get('DATABRICKS_CLIENT_ID', '')
os.environ['DATABRICKS_CLIENT_SECRET'] = yaml_config.get('DATABRICKS_CLIENT_SECRET', '')
os.environ['DATABRICKS_WAREHOUSE_ID'] = yaml_config.get('DATABRICKS_WAREHOUSE_ID', '')

# Databricks configuration
DATABRICKS_CONFIG = {
    'host': os.environ['DATABRICKS_HOST'],
    'client_id': os.environ['DATABRICKS_CLIENT_ID'],
    'client_secret': os.environ['DATABRICKS_CLIENT_SECRET'],
    'warehouse_id': os.environ['DATABRICKS_WAREHOUSE_ID']
}

print("Final DATABRICKS_CONFIG:", DATABRICKS_CONFIG)  # Debug print
print("Environment variables:", {k: v for k, v in os.environ.items() if k.startswith('DATABRICKS_')})  # Debug print

# Validate required configuration
def validate_config():
    required_vars = ['host', 'client_id', 'client_secret', 'warehouse_id']
    missing_vars = [var for var in required_vars if not DATABRICKS_CONFIG[var]]
    if missing_vars:
        print("Current configuration:", DATABRICKS_CONFIG)  # Debug print
        raise ValueError(f"Missing required environment variables in app.yaml: {', '.join(missing_vars)}") 