import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import flask  # for request context
import base64
from wordcloud import WordCloud
import io
from databricks import sql
from databricks.sdk.core import Config
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
#assert os.getenv('DATABRICKS_WAREHOUSE_ID')
#assert os.getenv('SERVING_ENDPOINT')

# Load data from SQL
def sqlQuery(query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    logger.info(f"Executing SQL query: {query[:100]}...")  # Log first 100 chars of query
    start_time = time.time()
    try:
        cfg = Config()  # Pull environment variables for auth
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
            credentials_provider=lambda: cfg.authenticate
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall_arrow().to_pandas()
                logger.info(f"SQL query completed in {time.time() - start_time:.2f} seconds. Returned {len(result)} rows.")
                return result
    except Exception as e:
        logger.error(f"SQL query failed: {str(e)}")
        raise

try:
    #data = sqlQuery("SELECT * FROM retail_cpg_demo.brand_manager.vw_brand_insights_toys")
    #sales_data = sqlQuery("SELECT * FROM retail_cpg_demo.brand_manager.monthly_brand_metrics WHERE category = 'Toys & Games'")

    # Load data from CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'app_data', 'brand_insights_data.csv')
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")

    # Load sales from CSV
    sales_path = os.path.join(current_dir, 'app_data', 'monthly_sales_toys.csv')
    sales_data = pd.read_csv(sales_path)
    print(f"Data shape: {sales_data.shape}")
    print(f"Data columns: {sales_data.columns}")

    
    # Convert the date column to a datetime object
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Clean up sentiment column
    # Standardize sentiment values
    sentiment_mapping = {
        'positive': 'Positive',
        'POSITIVE': 'Positive',
        'Positive': 'Positive',
        'negative': 'Negative',
        'NEGATIVE': 'Negative',
        'Negative': 'Negative',
        'neutral': 'Neutral',
        'NEUTRAL': 'Neutral',
        'Neutral': 'Neutral',
        'love': 'Love',
        'LOVE': 'Love',
        'Love': 'Love',
        'great': 'Great',
        'GREAT': 'Great',
        'Great': 'Great',
        'fine': 'Fine',
        'FINE': 'Fine',
        'Fine': 'Fine',
        'disappointed': 'Disappointed',
        'DISAPPOINTED': 'Disappointed',
        'Disappointed': 'Disappointed',
        'bad': 'Bad',
        'BAD': 'Bad',
        'Bad': 'Bad'
    }
    
    # Apply sentiment mapping and fill any missing values based on sentiment_score
    data['sentiment'] = data['sentiment'].map(sentiment_mapping)
    #data['sentiment'] = data['sentiment'].str.capitalize()
 

    data.loc[data['sentiment'].isna(), 'sentiment'] = data.loc[data['sentiment'].isna(), 'sentiment_score'].apply(
        lambda x: 'Love' if x == 5 else 'Great' if x == 4 else 'Fine' if x == 3 else 'Disappointed' if x == 2 else 'Bad' if x == 1 else 'Neutral'
    )
    
    
    # Add negative reviews for LEGO across different months
    april_2023 = pd.Timestamp('2023-04-15')
    may_2023 = pd.Timestamp('2023-05-15')
    june_2023 = pd.Timestamp('2023-06-15')
    july_2023 = pd.Timestamp('2023-07-15')
    
    # April 2023 - Random Issues
    random_issues = [
        'Missing pieces in the set',
        'Instructions unclear',
        'Pieces don\'t fit properly',
        'Box damaged on arrival',
        'Wrong pieces included'
    ]
    lego_april_reviews = pd.DataFrame({
        'date': [april_2023] * 1000,
        'category': ['Toys & Games'] * 1000,
        'brand': ['LEGO'] * 1000,
        'product': ['LEGO Set'] * 1000,
        'rating': [2.0] * 1000,
        'review_text': np.random.choice(random_issues, 1000),
        'sentiment': ['Negative'] * 1000,
        'sentiment_score': [1.5] * 1000,
        'positive_feature_list': [''] * 1000,
        'negative_feature_list': ['quality issues, missing parts'] * 1000,
        'avg_brand_price': [data[data['brand'] == 'LEGO']['avg_brand_price'].iloc[0]] * 1000
    })
    
    # May 2023 - Poor Quality
    lego_may_reviews = pd.DataFrame({
        'date': [may_2023] * 1500,
        'category': ['Toys & Games'] * 1500,
        'brand': ['LEGO'] * 1500,
        'product': ['LEGO Set'] * 1500,
        'rating': [2.0] * 1500,
        'review_text': ['Poor quality of pieces'] * 1500,
        'sentiment': ['Negative'] * 1500,
        'sentiment_score': [1.5] * 1500,
        'positive_feature_list': [''] * 1500,
        'negative_feature_list': ['poor quality, durability issues'] * 1500,
        'avg_brand_price': [data[data['brand'] == 'LEGO']['avg_brand_price'].iloc[0]] * 1500
    })
    
    # June 2023 - High Prices
    lego_june_reviews = pd.DataFrame({
        'date': [june_2023] * 3000,
        'category': ['Toys & Games'] * 3000,
        'brand': ['LEGO'] * 3000,
        'product': ['LEGO Set'] * 3000,
        'rating': [2.0] * 3000,
        'review_text': ['Too expensive for the value'] * 3000,
        'sentiment': ['Negative'] * 3000,
        'sentiment_score': [1.5] * 3000,
        'positive_feature_list': [''] * 3000,
        'negative_feature_list': ['high price, poor value'] * 3000,
        'avg_brand_price': [data[data['brand'] == 'LEGO']['avg_brand_price'].iloc[0]] * 3000
    })
    
    # July 2023 - Control Missing
    lego_july_reviews = pd.DataFrame({
        'date': [july_2023] * 5000,
        'category': ['Toys & Games'] * 5000,
        'brand': ['LEGO'] * 5000,
        'product': ['LEGO Set'] * 5000,
        'rating': [2.0] * 5000,
        'review_text': ['Missing control pieces'] * 5000,
        'sentiment': ['Negative'] * 5000,
        'sentiment_score': [1.5] * 5000,
        'positive_feature_list': [''] * 5000,
        'negative_feature_list': ['missing control, incomplete set'] * 5000,
        'avg_brand_price': [data[data['brand'] == 'LEGO']['avg_brand_price'].iloc[0]] * 5000
    })
    
    # Concatenate all new reviews with the existing data
    data = pd.concat([data, lego_april_reviews, lego_may_reviews, lego_june_reviews, lego_july_reviews], ignore_index=True)
    
    # Ensure required columns exist
    required_columns = ['date', 'category', 'brand', 'product', 'rating', 'review_text', 'sentiment', 'sentiment_score', 'positive_feature_list', 'negative_feature_list', 'avg_brand_price']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
    # Add positive reviews for Barbie across different months
    may_2023 = pd.Timestamp('2023-05-15')
    june_2023 = pd.Timestamp('2023-06-15')
    july_2023 = pd.Timestamp('2023-07-15')
    
    # May 2023 - Positive Reviews
    barbie_may_reviews = pd.DataFrame({
        'date': [may_2023] * 500,
        'category': ['Toys & Games'] * 500,
        'brand': ['Barbie'] * 500,
        'product': ['Barbie Doll'] * 500,
        'rating': [5.0] * 500,
        'review_text': ['Excellent quality and design'] * 500,
        'sentiment': ['Positive'] * 500,
        'sentiment_score': [4.5] * 500,
        'positive_feature_list': ['quality, design, durability'] * 500,
        'negative_feature_list': [''] * 500,
        'avg_brand_price': [data[data['brand'] == 'Barbie']['avg_brand_price'].iloc[0]] * 500
    })
    
    # June 2023 - Positive Reviews
    barbie_june_reviews = pd.DataFrame({
        'date': [june_2023] * 630,
        'category': ['Toys & Games'] * 630,
        'brand': ['Barbie'] * 630,
        'product': ['Barbie Doll'] * 630,
        'rating': [5.0] * 630,
        'review_text': ['Great customer experience'] * 630,
        'sentiment': ['Positive'] * 630,
        'sentiment_score': [4.5] * 630,
        'positive_feature_list': ['customer service, packaging, value'] * 630,
        'negative_feature_list': [''] * 630,
        'avg_brand_price': [data[data['brand'] == 'Barbie']['avg_brand_price'].iloc[0]] * 630
    })
    
    # July 2023 - Positive Reviews
    barbie_july_reviews = pd.DataFrame({
        'date': [july_2023] * 650,
        'category': ['Toys & Games'] * 650,
        'brand': ['Barbie'] * 650,
        'product': ['Barbie Doll'] * 650,
        'rating': [5.0] * 650,
        'review_text': ['Amazing product quality'] * 650,
        'sentiment': ['Positive'] * 650,
        'sentiment_score': [4.5] * 650,
        'positive_feature_list': ['quality, innovation, brand value'] * 650,
        'negative_feature_list': [''] * 650,
        'avg_brand_price': [data[data['brand'] == 'Barbie']['avg_brand_price'].iloc[0]] * 650
    })
    
    # Concatenate all new reviews with the existing data
    data = pd.concat([data, barbie_may_reviews, barbie_june_reviews, barbie_july_reviews], ignore_index=True)
    
    # Ensure required columns exist
    required_columns = ['date', 'category', 'brand', 'product', 'rating', 'review_text', 'sentiment', 'sentiment_score', 'positive_feature_list', 'negative_feature_list', 'avg_brand_price']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
except Exception as e:
    print(f"An error occurred loading data: {str(e)}")
    data = pd.DataFrame()

# Generate synthetic orders data
def generate_synthetic_orders(data):
    # Get unique dates and brands
    dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
    brands = data['brand'].unique()
    
    # Create empty DataFrame for orders
    orders_data = []
    
    # Generate orders for each brand
    for brand in brands:
        # Get brand's review data to use as base
        brand_data = data[data['brand'] == brand]
        
        # Generate base orders (using review count as a rough proxy)
        base_orders = len(brand_data) * np.random.uniform(0.5, 2.0)  # Random multiplier
        
        # Generate daily orders
        for date in dates:
            # Add some randomness and seasonality
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)  # Yearly seasonality
            weekly_pattern = 1 + 0.2 * np.sin(2 * np.pi * date.dayofweek / 7)  # Weekly pattern
            random_factor = np.random.normal(1, 0.1)  # Random daily variation
            
            # Calculate daily orders
            daily_orders = int(base_orders * seasonality * weekly_pattern * random_factor / len(dates))
            
            # Special handling for LEGO brand to show decline
            if brand == 'LEGO':
                # Define decline factors for different months
                decline_factors = {
                    '2023-04': 1.0,    # April: Normal
                    '2023-05': 0.7,    # May: 30% decline
                    '2023-06': 0.4,    # June: 60% decline
                    '2023-07': 0.2     # July: 80% decline
                }
                
                # Get the month key
                month_key = date.strftime('%Y-%m')
                
                # Apply decline factor if in the specified months
                if month_key in decline_factors:
                    daily_orders = int(daily_orders * decline_factors[month_key])
            
            # Calculate returns (typically 5-15% of orders, higher for LEGO during decline)
            if brand == 'LEGO' and month_key in decline_factors:
                # Higher return rate during decline period
                return_rate = 0.15 + (1 - decline_factors[month_key]) * 0.2  # Increases as orders decline
            else:
                return_rate = np.random.uniform(0.05, 0.15)  # Normal return rate
            
            daily_returns = int(daily_orders * return_rate)
            
            orders_data.append({
                'date': date,
                'brand': brand,
                'orders': max(0, daily_orders),  # Ensure non-negative
                'returns': max(0, daily_returns)  # Ensure non-negative
            })
    
    return pd.DataFrame(orders_data)

# Generate the synthetic orders data
orders_df = generate_synthetic_orders(data)

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                assets_folder='assets')  # Specify assets folder

# Custom styles
CARD_STYLE = {
    'backgroundColor': '#2d3436',  # Lighter dark gray
    'border': '1px solid #636e72',  # Lighter border
    'borderRadius': '5px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
}

CARD_HEADER_STYLE = {
    'backgroundColor': '#636e72',  # Lighter header
    'color': 'white',
    'borderBottom': '1px solid #b2bec3'  # Lighter border
}

COUNTER_BOX_STYLE = {
    'padding': '20px',
    'borderRadius': '10px',
    'textAlign': 'center',
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.2)'
}

# Add loading component styles
LOADING_STYLE = {
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',
    'justifyContent': 'center',
    'padding': '20px',
    'backgroundColor': '#2d3436',
    'borderRadius': '5px',
    'marginTop': '20px'
}

SPINNER_STYLE = {
    'width': '3rem',
    'height': '3rem',
    'color': '#3498db'
}



# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read())
        return f'data:image/png;base64,{encoded.decode()}'
    except:
        return None

# Function to generate wordcloud
def generate_wordcloud(texts, max_words=50, background_color='#2d3436'):
    if not texts or all(pd.isna(texts)):
        return None
    try:
        # Clean and convert texts to strings
        cleaned_texts = []
        for text in texts:
            # Handle NaN/None values
            if pd.isna(text):
                continue
            # Convert to string and clean
            try:
                text_str = str(text).strip()
                if text_str:  # Only add non-empty strings
                    cleaned_texts.append(text_str)
            except:
                continue
        
        if not cleaned_texts:
            return None
            
        text = ' '.join(cleaned_texts)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color=background_color,
            colormap='viridis',
            max_words=50,  # Limited to 50 words
            contour_width=1,
            contour_color='#636e72',
            min_font_size=10,  # Added to ensure readability
            max_font_size=100  # Added to control maximum word size
        ).generate(text)
        
        # Convert to base64 for display
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        return f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
    except Exception as e:
        print(f"Error generating wordcloud: {str(e)}")
        return None

# Layout
app.layout = dbc.Container([
    dcc.Store(id='page-load-trigger', data=0),
    dbc.Row([
        dbc.Col([
            html.H1("Brand Analyzer", className="text-center my-4 text-light"),
            html.Div([
                html.Span("BUILT ON", style={
                    'color': 'white',
                    'fontSize': '14px',
                    'marginRight': '10px',
                    'verticalAlign': 'middle'
                }),
                html.Img(
                    src=app.get_asset_url('img/small-scale-lockup-full-color-white-rgb.svg'),
                    style={
                        'width': '150px',
                        'display': 'inline-block',
                        'verticalAlign': 'middle'
                    }
                )
            ], style={
                'textAlign': 'center',
                'marginBottom': '20px'
            })
        ], width=12)
    ]),
    
    # Filter Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters", style=CARD_HEADER_STYLE),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Retailer", className="text-light"),
                            dcc.Dropdown(
                                id='retailer-dummy',
                                options=[{'label': 'Amazon', 'value': 'Amazon'}],
                                value='Amazon',
                                className="mb-3",
                                style={'color': 'black'}
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Category", className="text-light"),
                            dcc.Dropdown(
                                id='category-filter',
                                options=[{'label': cat, 'value': cat} for cat in sorted(data['category'].unique())],
                                value='Toys & Games',
                                className="mb-3",
                                style={'color': 'black'}
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Brand", className="text-light"),
                            dcc.Dropdown(
                                id='brand-filter',
                                options=[],
                                value=None,
                                className="mb-3",
                                style={'color': 'black'}
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Date Range", className="text-light"),
                            dcc.RangeSlider(
                                id='date-range-slider',
                                min=pd.to_datetime(data['date'].min()).timestamp(),
                                max=pd.to_datetime('2023-07-31').timestamp(),
                                step=30*24*60*60,  # 30 days in seconds
                                value=[
                                    pd.to_datetime(data['date'].min()).timestamp(),
                                    pd.to_datetime('2023-07-31').timestamp()
                                ],
                                marks={
                                    int(pd.to_datetime(date).timestamp()): date.strftime('%b %Y')
                                    for date in pd.date_range(
                                        start=data['date'].min(),
                                        end='2023-07-31',
                                        freq='3M'
                                    )
                                },
                                className="mb-3"
                            )
                        ], width=4)
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ], className="mb-4"),
    
    # Tabs
    dbc.Tabs([
        # Brand Health Tab
        dbc.Tab([
            # Brand Health Counters Row
            dbc.Row([
                # Sales Share Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Sales Share", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='brand-health-score', className="text-center mb-2", style={'color': '#2ecc71', 'fontSize': '2.5rem', 'cursor': 'pointer'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#2d4a3e'}),  # Lighter green
                    dbc.Tooltip(
                        html.Div([
                            html.H6("Sales Share Summary", style={
                                'color': 'white',
                                'marginBottom': '15px',
                                'fontSize': '1.1rem',
                                'fontWeight': 'bold',
                                'borderBottom': '2px solid #636e72',
                                'paddingBottom': '5px'
                            }),
                            html.Div([
                                html.Span("Brand:", style={'color': '#b2bec3'}),
                                html.Span(id='sales-brand-name', style={'color': '#2ecc71', 'fontWeight': 'bold'})
                            ], style={
                                'color': '#b2bec3',
                                'marginBottom': '8px',
                                'fontSize': '0.9rem',
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'alignItems': 'center'
                            }),
                            html.Div([
                                html.P("Sales Share represents the brand's market share based on total sales volume. It's calculated by comparing the brand's sales to the total category sales. A higher score indicates stronger market presence and sales performance.", 
                                    style={
                                        'color': '#b2bec3',
                                        'fontSize': '0.85rem',
                                        'marginTop': '15px',
                                        'paddingTop': '15px',
                                        'borderTop': '1px solid #636e72',
                                        'lineHeight': '1.4'
                                    }
                                )
                            ])
                        ], style={
                            'backgroundColor': '#1e272e',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.2)',
                            'border': '1px solid #636e72',
                            'minWidth': '250px'
                        }),
                        target="brand-health-score",
                        placement="top"
                    )
                ], width=3),
                # Units Share Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Units Share", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='competitive-score', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2.5rem', 'cursor': 'pointer'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#2d3e4a'}),  # Lighter blue
                    dbc.Tooltip(
                        html.Div([
                            html.H6("Units Share Summary", style={
                                'color': 'white',
                                'marginBottom': '15px',
                                'fontSize': '1.1rem',
                                'fontWeight': 'bold',
                                'borderBottom': '2px solid #636e72',
                                'paddingBottom': '5px'
                            }),
                            html.Div([
                                html.Span("Brand:", style={'color': '#b2bec3'}),
                                html.Span(id='units-brand-name', style={'color': '#3498db', 'fontWeight': 'bold'})
                            ], style={
                                'color': '#b2bec3',
                                'marginBottom': '8px',
                                'fontSize': '0.9rem',
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'alignItems': 'center'
                            }),
                            html.Div([
                                html.P("Units Share represents the brand's market share based on the number of units sold. It's calculated by comparing the brand's unit sales to the total category unit sales. A higher score indicates stronger market penetration and volume performance.", 
                                    style={
                                        'color': '#b2bec3',
                                        'fontSize': '0.85rem',
                                        'marginTop': '15px',
                                        'paddingTop': '15px',
                                        'borderTop': '1px solid #636e72',
                                        'lineHeight': '1.4'
                                    }
                                )
                            ])
                        ], style={
                            'backgroundColor': '#1e272e',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.2)',
                            'border': '1px solid #636e72',
                            'minWidth': '250px'
                        }),
                        target="competitive-score",
                        placement="top"
                    )
                ], width=3),
                # Reviews Share Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Reviews Share", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='product-score', className="text-center mb-2", style={'color': '#9b59b6', 'fontSize': '2.5rem', 'cursor': 'pointer'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#3e2d4a'}),  # Lighter purple
                    dbc.Tooltip(
                        html.Div([
                            html.H6("Reviews Share Summary", style={
                                'color': 'white',
                                'marginBottom': '15px',
                                'fontSize': '1.1rem',
                                'fontWeight': 'bold',
                                'borderBottom': '2px solid #636e72',
                                'paddingBottom': '5px'
                            }),
                            html.Div([
                                html.Span("Brand:", style={'color': '#b2bec3'}),
                                html.Span(id='reviews-brand-name', style={'color': '#9b59b6', 'fontWeight': 'bold'})
                            ], style={
                                'color': '#b2bec3',
                                'marginBottom': '8px',
                                'fontSize': '0.9rem',
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'alignItems': 'center'
                            }),
                            html.Div([
                                html.P("Reviews Share represents the brand's share of customer reviews in the category. It's calculated by comparing the brand's review count to the total category reviews. A higher score indicates stronger customer engagement and feedback volume.", 
                                    style={
                                        'color': '#b2bec3',
                                        'fontSize': '0.85rem',
                                        'marginTop': '15px',
                                        'paddingTop': '15px',
                                        'borderTop': '1px solid #636e72',
                                        'lineHeight': '1.4'
                                    }
                                )
                            ])
                        ], style={
                            'backgroundColor': '#1e272e',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.2)',
                            'border': '1px solid #636e72',
                            'minWidth': '250px'
                        }),
                        target="product-score",
                        placement="top"
                    )
                ], width=3),
                # Average Customer Rating Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Avg. Customer Rating", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='average-rating', className="text-center mb-2", style={'color': '#f1c40f', 'fontSize': '2.5rem', 'cursor': 'pointer'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#4a3e2d'}),  # Lighter yellow
                    dbc.Tooltip(
                        html.Div([
                            html.H6("Customer Rating Summary", style={
                                'color': 'white',
                                'marginBottom': '15px',
                                'fontSize': '1.1rem',
                                'fontWeight': 'bold',
                                'borderBottom': '2px solid #636e72',
                                'paddingBottom': '5px'
                            }),
                            html.Div([
                                html.Span("Brand:", style={'color': '#b2bec3'}),
                                html.Span(id='rating-brand-name', style={'color': '#f1c40f', 'fontWeight': 'bold'})
                            ], style={
                                'color': '#b2bec3',
                                'marginBottom': '8px',
                                'fontSize': '0.9rem',
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'alignItems': 'center'
                            }),
                            html.Div([
                                html.P("Average Customer Rating represents the mean rating given by customers to the brand's products. It's calculated from verified purchase reviews on a scale of 1-5 stars. A higher rating indicates better customer satisfaction and product quality.", 
                                    style={
                                        'color': '#b2bec3',
                                        'fontSize': '0.85rem',
                                        'marginTop': '15px',
                                        'paddingTop': '15px',
                                        'borderTop': '1px solid #636e72',
                                        'lineHeight': '1.4'
                                    }
                                )
                            ])
                        ], style={
                            'backgroundColor': '#1e272e',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.2)',
                            'border': '1px solid #636e72',
                            'minWidth': '250px'
                        }),
                        target="average-rating",
                        placement="top"
                    )
                ], width=3)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Reviews At A Glance", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            # Add counters row
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Total Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='total-reviews-counter', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2rem', 'cursor': 'pointer'})
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#2d3e4a'}),  # Lighter blue
                                    dbc.Tooltip(id="total-reviews-tooltip", target="total-reviews-counter", placement="top")
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Positive Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='positive-reviews-counter', className="text-center mb-2", style={'color': '#2ecc71', 'fontSize': '2rem', 'cursor': 'pointer'})
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#2d4a3e'}),  # Lighter green
                                    dbc.Tooltip(id="positive-reviews-tooltip", target="positive-reviews-counter", placement="top")
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Neutral Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='neutral-reviews-counter', className="text-center mb-2", style={'color': '#f1c40f', 'fontSize': '2rem', 'cursor': 'pointer'})
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#4a3e2d'}),  # Lighter yellow
                                    dbc.Tooltip(id="neutral-reviews-tooltip", target="neutral-reviews-counter", placement="top")
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Negative Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='negative-reviews-counter', className="text-center mb-2", style={'color': '#e74c3c', 'fontSize': '2rem', 'cursor': 'pointer'})
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#4a2d2d'}),  # Lighter red
                                    dbc.Tooltip(id="negative-reviews-tooltip", target="negative-reviews-counter", placement="top")
                                ], width=3)
                            ], className="mb-4")
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            
            # Sentiment Analysis Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("LLM Extracted Sentiment Analysis", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Markdown("Key words extracted from consumer reviews using Databricks [_ai_analyze_sentiment_](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_analyze_sentiment) function.", style={'color': 'white'}),
                            dcc.Graph(id='sentiment-treemap', style={'height': '600px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Overall", tab_id="tab-brand-health"),
        
        # Product Attribute Tab
        dbc.Tab([
            dbc.Row([
                # Monthly Orders Chart (now first)
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.Span("Monthly Orders & Returns", style={'flex': '1'}),
                                html.Span(id='orders-range-display', style={'color': '#b2bec3', 'fontSize': '0.9rem'})
                            ], style={'display': 'flex', 'alignItems': 'center'})
                        ], style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(
                                id='monthly-orders-chart',
                                style={'height': '400px'},
                                config={'displayModeBar': True}
                            )
                        ])
                    ], style=CARD_STYLE)
                ], width=6),
                # Monthly Reviews Chart (now second)
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.Span("Monthly Review Trends", style={'flex': '1'}),
                                html.Span(id='selected-range-display', style={'color': '#b2bec3', 'fontSize': '0.9rem'})
                            ], style={'display': 'flex', 'alignItems': 'center'})
                        ], style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(
                                id='monthly-reviews-chart',
                                style={'height': '400px'},
                                config={'displayModeBar': True}
                            )
                        ])
                    ], style=CARD_STYLE)
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                # Brand Review Word Clouds
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.Span(id="brand-review-header", style={'flex': '1'}),
                                html.Span(id='wordcloud-range-display', style={'color': '#b2bec3', 'fontSize': '0.9rem'})
                            ], style={'display': 'flex', 'alignItems': 'center'})
                        ], style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Positive", className="text-center mb-3"),
                                    html.Img(id='brand-positive-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6),
                                dbc.Col([
                                    html.H4("Negative", className="text-center mb-3"),
                                    html.Img(id='brand-negative-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6)
                            ])
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Details", tab_id="tab-product"),
        
        # Competitive Positioning Tab
        dbc.Tab([
            # Existing Market Share Trends
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Share of Voice", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='market-share-trend', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),

            # Pricing comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Pricing Comparison", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='pricing-comparison', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            
            # Existing Competitive Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="category-review-header", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Positive", className="text-center mb-3"),
                                    html.Img(id='positive-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6),
                                dbc.Col([
                                    html.H4("Negative", className="text-center mb-3"),
                                    html.Img(id='negative-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6)
                            ])
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Competitive", tab_id="tab-competitive")
    ], id="tabs", active_tab="tab-brand-health", className="mb-4"),
    
], fluid=True, style={'backgroundColor': '#2d3436', 'minHeight': '100vh', 'padding': '20px'})  # Lighter background

# Update the callback to dynamically filter brands based on category
@app.callback(
    Output('brand-filter', 'options'),
    Output('brand-filter', 'value'),
    Input('category-filter', 'value')
)
def update_brand_options(selected_category):
    logger.info(f"update_brand_options triggered with category: {selected_category}")
    filtered_data = data[data['category'] == selected_category]
    brands = sorted(filtered_data['brand'].unique())
    options = [{'label': brd, 'value': brd} for brd in brands]
    value = brands[0] if brands else None
    logger.info(f"Returning {len(options)} brand options")
    return options, value


# Update the callback with better error handling and data validation
@app.callback(
    Output('sentiment-treemap', 'figure'),
    Output('monthly-reviews-chart', 'figure'),
    Output('monthly-orders-chart', 'figure'),
    Output('brand-health-score', 'children'),
    Output('competitive-score', 'children'),
    Output('product-score', 'children'),
    Output('average-rating', 'children'),
    Output('market-share-trend', 'figure'),
    Output('pricing-comparison', 'figure'),
    Output('positive-wordcloud', 'src'),
    Output('negative-wordcloud', 'src'),
    Output('brand-positive-wordcloud', 'src'),
    Output('brand-negative-wordcloud', 'src'),
    Output('total-reviews-counter', 'children'),
    Output('positive-reviews-counter', 'children'),
    Output('neutral-reviews-counter', 'children'),
    Output('negative-reviews-counter', 'children'),
    Output('selected-range-display', 'children'),
    Output('wordcloud-range-display', 'children'),
    Output('sales-brand-name', 'children'),
    Output('units-brand-name', 'children'),
    Output('reviews-brand-name', 'children'),
    Output('rating-brand-name', 'children'),
    Input('page-load-trigger', 'data'),
    Input('category-filter', 'value'),
    Input('brand-filter', 'value'),
    Input('retailer-dummy', 'value'),
    Input('date-range-slider', 'value'),
    Input('monthly-reviews-chart', 'relayoutData'),
    Input('monthly-orders-chart', 'relayoutData'),
    State('monthly-reviews-chart', 'figure'),
    State('monthly-orders-chart', 'figure'),
    prevent_initial_call=True
)
def update_visuals(n_clicks, category, brand, retailer, date_range, reviews_relayout, orders_relayout, reviews_figure, orders_figure):
    import copy
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    def create_empty_fig():
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    if category is None or brand is None:
        empty_fig = create_empty_fig()
        return (
            empty_fig, empty_fig, empty_fig, "0%", "0%", "0%", "0.0", empty_fig, empty_fig,
            None, None, None, None, "0", "0", "0", "0", "", "", "", "", "", ""
        )

    # Convert timestamp to datetime
    start_date = pd.to_datetime(date_range[0], unit='s')
    end_date = pd.to_datetime(date_range[1], unit='s')

    # Apply filters
    category_data = data[(data['category'] == category) & (data['date'] >= start_date) & (data['date'] <= end_date)]
    filtered_data = category_data[category_data['brand'] == brand]

    if len(filtered_data) == 0:
        empty_fig = create_empty_fig()
        return (
            empty_fig, empty_fig, empty_fig, "0%", "0%", "0%", "0.0", empty_fig, empty_fig,
            None, None, None, None, "0", "0", "0", "0", "", "", "", "", "", ""
        )

    # --- Cross-filtering and zoom preservation logic ---
    xaxis_range = None
    range_display = "Zoom in on the chart to filter wordclouds by date range"
    wordcloud_range_display = "Showing all data"
    range_data = filtered_data

    # 1. If relayout event, use its range
    if trigger_id == 'monthly-reviews-chart' and reviews_relayout:
        if 'xaxis.range[0]' in reviews_relayout and 'xaxis.range[1]' in reviews_relayout:
            xaxis_range = [reviews_relayout['xaxis.range[0]'], reviews_relayout['xaxis.range[1]']]
        elif any(k in reviews_relayout for k in ['autosize', 'xaxis.autorange', 'yaxis.autorange']):
            xaxis_range = None
    elif trigger_id == 'monthly-orders-chart' and orders_relayout:
        if 'xaxis.range[0]' in orders_relayout and 'xaxis.range[1]' in orders_relayout:
            xaxis_range = [orders_relayout['xaxis.range[0]'], orders_relayout['xaxis.range[1]']]
        elif any(k in orders_relayout for k in ['autosize', 'xaxis.autorange', 'yaxis.autorange']):
            xaxis_range = None
    # 2. If not a relayout event, but previous figure had a zoom, preserve it
    elif reviews_figure and 'layout' in reviews_figure and 'xaxis' in reviews_figure['layout']:
        prev_range = reviews_figure['layout']['xaxis'].get('range')
        if prev_range:
            xaxis_range = prev_range

    # Now, filter your data for wordclouds and set xaxis for both charts
    if xaxis_range:
        start_zoom = pd.to_datetime(xaxis_range[0])
        end_zoom = pd.to_datetime(xaxis_range[1])
        range_data = filtered_data[(filtered_data['date'] >= start_zoom) & (filtered_data['date'] <= end_zoom)]
        range_display = f"Showing data from {start_zoom.strftime('%B %Y')} to {end_zoom.strftime('%B %Y')}"
        wordcloud_range_display = range_display
    else:
        range_data = filtered_data
        range_display = "Zoom in on the chart to filter wordclouds by date range"
        wordcloud_range_display = "Showing all data"

    # --- Wordclouds ---
    brand_positive_wordcloud = generate_wordcloud(
        range_data[range_data['sentiment_score'] > 3]['positive_feature_list'].dropna().tolist(),
        background_color='white'
    )
    brand_negative_wordcloud = generate_wordcloud(
        range_data[range_data['sentiment_score'] < 3]['negative_feature_list'].dropna().tolist(),
        background_color='#636e72'
    )
    # Competitor wordclouds (always full range for category)
    other_brands_data = category_data[category_data['brand'] != brand]
    positive_wordcloud = generate_wordcloud(other_brands_data['positive_feature_list'].dropna().tolist(), background_color='white')
    negative_wordcloud = generate_wordcloud(other_brands_data['negative_feature_list'].dropna().tolist(), background_color='#636e72')

    # --- Sentiment Treemap ---
    sentiment_counts = range_data['sentiment'].value_counts()
    sentiment_colors = {
        'Love': '#10e380', 'Great': '#26c77b', 'Positive': '#30b375', 'Fine': '#4f6159',
        'Disappointed': '#c45b1d', 'Bad': '#c4281d', 'Negative': '#fc1505'
    }
    sentiment_fig = go.Figure(go.Treemap(
        ids=sentiment_counts.index,
        labels=sentiment_counts.index,
        parents=[''] * len(sentiment_counts),
        values=sentiment_counts.values,
        marker=dict(colors=[sentiment_colors.get(s, '#808080') for s in sentiment_counts.index], line=dict(width=2, color='#2d3436')),
        textinfo="label+value+percent parent",
        textfont=dict(color='white', size=14),
        hovertemplate='<span style="color: #2d3436">%{label}</span><br>Count: %{value}<br>Percentage: %{percentParent:.1%}<br><extra></extra>'
    ))
    sentiment_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=25, r=25, b=25), font=dict(color='white'),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, font=dict(size=12, color='white'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )

    # --- Monthly Reviews Chart ---
    monthly_sentiment = range_data.groupby([
        pd.to_datetime(range_data['date']).dt.to_period('M'),
        pd.cut(range_data['sentiment_score'], bins=[0, 2.9, 3.1, 5], labels=['Negative (<3)', 'Neutral (=3)', 'Positive (>3)'])
    ]).size().unstack(fill_value=0)
    monthly_sentiment.index = monthly_sentiment.index.astype(str)
    sentiment_stats = {}
    for sentiment in monthly_sentiment.columns:
        mean = monthly_sentiment[sentiment].mean()
        std = monthly_sentiment[sentiment].std()
        sentiment_stats[sentiment] = {'mean': mean, 'std': std, 'threshold': mean + (1.5 * std)}
    sentiment_colors_line = {'Positive (>3)': '#10e380', 'Neutral (=3)': '#4f6159', 'Negative (<3)': '#c4281d'}
    monthly_reviews_chart = go.Figure()
    for sentiment in monthly_sentiment.columns:
        hover_text = []
        for month, value in monthly_sentiment[sentiment].items():
            stats = sentiment_stats[sentiment]
            if value > stats['threshold']:
                hover_text.append(f"<span style='color: #2d3436'>Month: {month}<br>Count: {value}<br>⚠️ Spike Detected</span>")
            else:
                hover_text.append(f"<span style='color: #2d3436'>Month: {month}<br>Count: {value}</span>")
        monthly_reviews_chart.add_trace(go.Scatter(
            name=sentiment, x=monthly_sentiment.index, y=monthly_sentiment[sentiment], mode='lines+markers+text',
            line=dict(color=sentiment_colors_line.get(sentiment, '#808080'), width=3),
            marker=dict(size=8, color=sentiment_colors_line.get(sentiment, '#808080'), line=dict(width=2, color='#1a1a1a')),
            text=monthly_sentiment[sentiment].apply(lambda x: f'{x:,}'), textposition='top center', textfont=dict(color='white', size=11),
            hovertext=hover_text, hoverinfo='text'))
    if xaxis_range:
        monthly_reviews_chart.update_layout(xaxis={'range': xaxis_range, 'autorange': False})
    else:
        monthly_reviews_chart.update_layout(xaxis={'autorange': True})
    monthly_reviews_chart.update_layout(
        xaxis_title='Month', yaxis_title='Number of Reviews', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(gridcolor='#404040', tickangle=45), yaxis=dict(gridcolor='#404040', showgrid=True), hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='white'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'), margin=dict(t=40, l=25, r=25, b=50)
    )

    # --- Monthly Orders Chart ---
    sales_data['month'] = pd.to_datetime(sales_data['month'])
    brand_sales = sales_data[(sales_data['brand'] == brand) & (sales_data['month'] >= start_date) & (sales_data['month'] <= end_date)]
    monthly_sales = brand_sales.groupby(pd.to_datetime(brand_sales['month']).dt.to_period('M')).agg({'units': 'sum', 'returns': 'sum'}).reset_index()
    monthly_sales['date'] = monthly_sales['month'].astype(str)
    orders_fig = go.Figure()
    orders_fig.add_trace(go.Scatter(
        x=monthly_sales['date'], y=monthly_sales['units'], mode='lines+markers+text', name='Units Sold',
        line=dict(color='#FFA500', width=3), marker=dict(size=8, color='#FFA500', line=dict(width=2, color='#1a1a1a')),
        text=[f'{int(x):,}' for x in monthly_sales['units']], textposition='top center', textfont=dict(color='white', size=11),
        hovertemplate='<span style="color: #2d3436">Month: %{x}</span><br><span style="color: #2d3436">Units Sold: %{y:,.0f}</span><br><extra></extra>'
    ))
    orders_fig.add_trace(go.Scatter(
        x=monthly_sales['date'], y=monthly_sales['returns'], mode='lines+markers+text', name='Returns',
        line=dict(color='#e74c3c', width=3), marker=dict(size=8, color='#e74c3c', line=dict(width=2, color='#1a1a1a')),
        text=[f'{int(x):,}' for x in monthly_sales['returns']], textposition='bottom center', textfont=dict(color='white', size=11),
        hovertemplate='<span style="color: #2d3436">Month: %{x}</span><br><span style="color: #2d3436">Returns: %{y:,.0f}</span><br><span style="color: #2d3436">Return Rate: %{customdata:.1%}</span><br><extra></extra>',
        customdata=[x/y if y else 0 for x, y in zip(monthly_sales['returns'], monthly_sales['units'])]
    ))
    if xaxis_range:
        orders_fig.update_layout(xaxis={'range': xaxis_range, 'autorange': False})
    else:
        orders_fig.update_layout(xaxis={'autorange': True})
    orders_fig.update_layout(
        xaxis_title='Month', yaxis_title='Number of Units/Returns', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(gridcolor='#404040', tickangle=45), yaxis=dict(gridcolor='#404040', showgrid=True), hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='white'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'), margin=dict(t=40, l=25, r=25, b=50)
    )

    # --- Market Share Trend ---
    monthly_brand_counts = category_data.groupby([
        pd.to_datetime(category_data['date']).dt.to_period('M'), 'brand']
    ).size().reset_index(name='review_count')
    monthly_brand_counts['date'] = monthly_brand_counts['date'].dt.to_timestamp()
    market_share_fig = go.Figure()
    for brand_name in category_data['brand'].unique():
        brand_data = monthly_brand_counts[monthly_brand_counts['brand'] == brand_name]
        market_share_fig.add_trace(go.Scatter(
            x=brand_data['date'], y=brand_data['review_count'], name=brand_name, stackgroup='one', fill='tonexty', line=dict(width=0.5),
            text=brand_data['review_count'].apply(lambda x: f'{x:,}'), textposition='top center', textfont=dict(color='#2d3436', size=11),
            hovertemplate='<span style="color: #2d3436">%{x|%B %Y}</span><br><span style="color: #2d3436">Brand: %{fullData.name}</span><br><span style="color: #2d3436">Reviews: %{y:,}</span><br><extra></extra>',
            hoverlabel=dict(bgcolor='white', font=dict(family='Arial', size=14, color='#2d3436'), bordercolor='#636e72')
        ))
    market_share_fig.update_layout(
        xaxis_title='Month', yaxis_title='Number of Reviews', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(gridcolor='#404040', tickformat='%B %Y'), yaxis=dict(gridcolor='#404040', showgrid=True), hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='white'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )

    # --- Pricing Comparison ---
    price_fig = go.Figure()
    brands = category_data['brand'].unique()
    colors = ['#FFA500' if b == brand else '#3498db' for b in brands]
    price_fig.add_trace(go.Bar(
        x=brands, y=category_data.groupby('brand')['avg_brand_price'].mean(), name='Brand Average Price', marker_color=colors
    ))
    category_avg = category_data['avg_brand_price'].mean()
    price_fig.add_trace(go.Scatter(
        x=brands, y=[category_avg] * len(brands), name='Category Average', line=dict(color='#e74c3c', width=2, dash='dash'), mode='lines'
    ))
    price_fig.update_layout(
        xaxis_title='Brand', yaxis_title='Average Price', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(gridcolor='#404040'), yaxis=dict(gridcolor='#404040'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='white'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)')
    )

    # --- Metrics ---
    total_reviews = len(range_data)
    positive_reviews_count = len(range_data[range_data['sentiment_score'] >= 3])
    neutral_reviews_count = len(range_data[(range_data['sentiment_score'] >= 2) & (range_data['sentiment_score'] < 3)])
    negative_reviews_count = len(range_data[range_data['sentiment_score'] < 2])
    brand_health = (range_data['sentiment_score'] >= 3).mean() * 100
    competitive_position = np.random.uniform(75, 95)
    product_score = np.random.uniform(80, 98)
    if brand_health >= 80:
        health_color = '#2ecc71'
    elif brand_health >= 70:
        health_color = '#f1c40f'
    else:
        health_color = '#e74c3c'

    return [
        sentiment_fig.to_dict(),
        monthly_reviews_chart.to_dict(),
        orders_fig.to_dict(),
        html.Div(f"{brand_health:.1f}%", style={'color': health_color}),
        html.Div(f"{competitive_position:.1f}%"),
        html.Div(f"{product_score:.1f}%"),
        html.Div(f"{range_data['rating'].mean():.1f}/5.0"),
        market_share_fig.to_dict(),
        price_fig.to_dict(),
        positive_wordcloud,
        negative_wordcloud,
        brand_positive_wordcloud,
        brand_negative_wordcloud,
        html.Div(f"{total_reviews:,}"),
        html.Div(f"{positive_reviews_count:,}"),
        html.Div(f"{neutral_reviews_count:,}"),
        html.Div(f"{negative_reviews_count:,}"),
        html.Div(range_display),
        html.Div(wordcloud_range_display),
        brand, brand, brand, brand
    ]

# Update the callback to handle all tab switching
@app.callback(
    Output('tabs', 'active_tab'),
    Input('page-load-trigger', 'data'),
    prevent_initial_call=True
)
def switch_to_product_tab(n_clicks):
    logger.info("switch_to_product_tab triggered")
    return 'tab-market'

# Update the tooltip callback to show multiple top features
@app.callback(
    [Output("total-reviews-tooltip", "children"),
     Output("positive-reviews-tooltip", "children"),
     Output("neutral-reviews-tooltip", "children"),
     Output("negative-reviews-tooltip", "children")],
    [Input('category-filter', 'value'),
     Input('brand-filter', 'value')]
)
def update_review_summaries(category, brand):
    logger.info(f"update_review_summaries triggered with category={category}, brand={brand}")
    try:
        # Filter data for the selected brand and category
        filtered_data = data[(data['category'] == category) & (data['brand'] == brand)]
        logger.info(f"Filtered data shape for tooltips: {filtered_data.shape}")
        
        # Calculate metrics
        total_reviews = len(filtered_data)
        positive_reviews = len(filtered_data[filtered_data['sentiment_score'] > 3])
        neutral_reviews = len(filtered_data[filtered_data['sentiment_score'] == 3])
        negative_reviews = len(filtered_data[filtered_data['sentiment_score'] < 3])
        
        # Common tooltip styles
        tooltip_header_style = {
            'color': 'white',
            'marginBottom': '15px',
            'fontSize': '1.1rem',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #636e72',
            'paddingBottom': '5px'
        }
        
        tooltip_text_style = {
            'color': '#b2bec3',
            'marginBottom': '8px',
            'fontSize': '0.9rem',
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center'
        }
        
        tooltip_container_style = {
            'backgroundColor': '#1e272e',
            'padding': '15px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.2)',
            'border': '1px solid #636e72',
            'minWidth': '250px'
        }

        # Function to get top features
        def get_top_features(data, feature_list, n=3):
            features = data[feature_list].str.split(',').explode().str.strip()
            return features.value_counts().head(n).index.tolist()
        
        # Create summary tooltips
        total_summary = html.Div([
            html.H6("Total Reviews Summary", style=tooltip_header_style),
            html.Div([
                html.Span("Total Reviews:", style={'color': '#b2bec3'}),
                html.Span(f"{total_reviews:,}", style={'color': '#3498db', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Average Rating:", style={'color': '#b2bec3'}),
                html.Span(f"{filtered_data['rating'].mean():.1f}/5.0", style={'color': '#3498db', 'fontWeight': 'bold'})
            ], style=tooltip_text_style)
        ], style=tooltip_container_style)
        
        positive_summary = html.Div([
            html.H6("Positive Reviews Summary", style=tooltip_header_style),
            html.Div([
                html.Span("Positive Reviews:", style={'color': '#b2bec3'}),
                html.Span(f"{positive_reviews:,}", style={'color': '#2ecc71', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Positive Rate:", style={'color': '#b2bec3'}),
                html.Span(f"{positive_reviews/total_reviews*100:.1f}%", style={'color': '#2ecc71', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Top Features:", style={'color': '#b2bec3'}),
                html.Span(
                    ", ".join(get_top_features(filtered_data[filtered_data['sentiment_score'] > 3], 'positive_feature_list')),
                    style={'color': '#2ecc71', 'fontWeight': 'bold', 'textAlign': 'right', 'maxWidth': '150px'}
                )
            ], style=tooltip_text_style)
        ], style=tooltip_container_style)
        
        neutral_summary = html.Div([
            html.H6("Neutral Reviews Summary", style=tooltip_header_style),
            html.Div([
                html.Span("Neutral Reviews:", style={'color': '#b2bec3'}),
                html.Span(f"{neutral_reviews:,}", style={'color': '#f1c40f', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Neutral Rate:", style={'color': '#b2bec3'}),
                html.Span(f"{neutral_reviews/total_reviews*100:.1f}%", style={'color': '#f1c40f', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Top Features:", style={'color': '#b2bec3'}),
                html.Span(
                    ", ".join(get_top_features(filtered_data[filtered_data['sentiment_score'] == 3], 'positive_feature_list')),
                    style={'color': '#f1c40f', 'fontWeight': 'bold', 'textAlign': 'right', 'maxWidth': '150px'}
                )
            ], style=tooltip_text_style)
        ], style=tooltip_container_style)
        
        negative_summary = html.Div([
            html.H6("Negative Reviews Summary", style=tooltip_header_style),
            html.Div([
                html.Span("Negative Reviews:", style={'color': '#b2bec3'}),
                html.Span(f"{negative_reviews:,}", style={'color': '#e74c3c', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Negative Rate:", style={'color': '#b2bec3'}),
                html.Span(f"{negative_reviews/total_reviews*100:.1f}%", style={'color': '#e74c3c', 'fontWeight': 'bold'})
            ], style=tooltip_text_style),
            html.Div([
                html.Span("Top Issues:", style={'color': '#b2bec3'}),
                html.Span(
                    ", ".join(get_top_features(filtered_data[filtered_data['sentiment_score'] < 3], 'negative_feature_list')),
                    style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'right', 'maxWidth': '150px'}
                )
            ], style=tooltip_text_style)
        ], style=tooltip_container_style)
        
        return total_summary, positive_summary, neutral_summary, negative_summary
        
    except Exception as e:
        logger.error(f"Error in update_review_summaries callback: {str(e)}", exc_info=True)
        error_msg = html.P(f"Error generating summary: {str(e)}", style={'color': 'white'})
        return error_msg, error_msg, error_msg, error_msg

# Add new callback for the brand review header
@app.callback(
    Output("brand-review-header", "children"),
    Input('brand-filter', 'value')
)
def update_brand_review_header(selected_brand):
    logger.info(f"update_brand_review_header triggered with brand={selected_brand}")
    if selected_brand:
        return f"What people are saying about {selected_brand} products..."
    return "What people are saying about selected brand..."

# Add callback for the category review header
@app.callback(
    Output("category-review-header", "children"),
    Input('category-filter', 'value')
)
def update_category_review_header(selected_category):
    logger.info(f"update_category_review_header triggered with category={selected_category}")
    if selected_category:
        return f"What people are saying about other products in the {selected_category} category..."
    return "What people are saying about other products in the selected category..."

# Update the cross-filtering callback to handle all interactions
# @app.callback(
#     [Output('brand-positive-wordcloud', 'src', allow_duplicate=True),
#      Output('brand-negative-wordcloud', 'src', allow_duplicate=True),
#      Output('wordcloud-range-display', 'children', allow_duplicate=True),
#      Output('selected-range-display', 'children', allow_duplicate=True),
#      Output('monthly-reviews-chart', 'figure', allow_duplicate=True),
#      Output('monthly-orders-chart', 'figure', allow_duplicate=True),
#      Output('market-share-trend', 'figure', allow_duplicate=True)],
#     [Input('monthly-orders-chart', 'relayoutData'),
#      Input('monthly-reviews-chart', 'relayoutData'),
#      Input('market-share-trend', 'relayoutData')],
#     [State('category-filter', 'value'),
#      State('brand-filter', 'value'),
#      State('monthly-reviews-chart', 'figure'),
#      State('monthly-orders-chart', 'figure'),
#      State('market-share-trend', 'figure')],
#     prevent_initial_call=True
# )
# def update_details_tab_figures(orders_relayout, reviews_relayout, market_share_relayout, category, brand, reviews_figure, orders_figure, market_share_figure):
#     # Get the trigger that caused the callback
#     ctx = dash.callback_context
#     trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
#     logger.info(f"update_details_tab_figures triggered by {trigger_id} with category={category}, brand={brand}")
    
#     if not category or not brand:
#         logger.info("No category or brand selected, returning current figures")
#         return None, None, "", "", reviews_figure, orders_figure, market_share_figure
        
#     try:
#         filtered_data = data[(data['category'] == category) & (data['brand'] == brand)]
#         logger.info(f"Filtered data shape: {filtered_data.shape}")
        
#         # Determine which chart triggered the callback and get the date range
#         relayout_data = None
#         if trigger_id == 'monthly-orders-chart':
#             relayout_data = orders_relayout
#         elif trigger_id == 'monthly-reviews-chart':
#             relayout_data = reviews_relayout
#         else:
#             relayout_data = market_share_relayout
        
#         if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
#             # Get the zoom range
#             start_date = pd.to_datetime(relayout_data['xaxis.range[0]'])
#             end_date = pd.to_datetime(relayout_data['xaxis.range[1]'])
            
#             # Filter data for the selected range
#             range_data = filtered_data[
#                 (filtered_data['date'] >= start_date) & 
#                 (filtered_data['date'] <= end_date)
#             ]
            
#             # Generate wordclouds for the selected range
#             # Positive wordcloud: sentiment_score > 3
#             positive_data = range_data[range_data['sentiment_score'] > 3]
#             brand_positive_wordcloud = generate_wordcloud(
#                 positive_data['positive_feature_list'].dropna().tolist(),
#                 background_color='white'
#             )
            
#             # Negative wordcloud: sentiment_score < 3
#             negative_data = range_data[range_data['sentiment_score'] < 3]
#             brand_negative_wordcloud = generate_wordcloud(
#                 negative_data['negative_feature_list'].dropna().tolist(),
#                 background_color='#636e72'
#             )
            
#             # Update the range displays
#             range_display = f"Showing data from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
#             wordcloud_range_display = range_display
            
#             # Update all charts to show the same range
#             for figure in [reviews_figure, orders_figure, market_share_figure]:
#                 if figure and 'layout' in figure:
#                     if 'xaxis' not in figure['layout']:
#                         figure['layout']['xaxis'] = {}
#                     figure['layout']['xaxis'].update({
#                         'range': [start_date, end_date],
#                         'autorange': False
#                     })
            
#         elif relayout_data and any(key in relayout_data for key in ['autosize', 'xaxis.autorange', 'yaxis.autorange']):
#             # Reset zoom on all charts
#             for figure in [reviews_figure, orders_figure, market_share_figure]:
#                 if figure and 'layout' in figure:
#                     if 'xaxis' not in figure['layout']:
#                         figure['layout']['xaxis'] = {}
#                     figure['layout']['xaxis'].update({
#                         'autorange': True
#                     })
            
#             # Show all data in wordclouds
#             # Positive wordcloud: sentiment_score > 3
#             positive_data = filtered_data[filtered_data['sentiment_score'] > 3]
#             brand_positive_wordcloud = generate_wordcloud(
#                 positive_data['positive_feature_list'].dropna().tolist(),
#                 background_color='white'
#             )
            
#             # Negative wordcloud: sentiment_score < 3
#             negative_data = filtered_data[filtered_data['sentiment_score'] < 3]
#             brand_negative_wordcloud = generate_wordcloud(
#                 negative_data['negative_feature_list'].dropna().tolist(),
#                 background_color='#636e72'
#             )
#             range_display = "Showing all data"
#             wordcloud_range_display = range_display
#         else:
#             # Keep the current state of the figures
#             brand_positive_wordcloud = generate_wordcloud(
#                 filtered_data[filtered_data['sentiment_score'] > 3]['positive_feature_list'].dropna().tolist(),
#                 background_color='white'
#             )
#             brand_negative_wordcloud = generate_wordcloud(
#                 filtered_data[filtered_data['sentiment_score'] < 3]['negative_feature_list'].dropna().tolist(),
#                 background_color='#636e72'
#             )
#             range_display = "Showing all data"
#             wordcloud_range_display = range_display
            
#         return brand_positive_wordcloud, brand_negative_wordcloud, wordcloud_range_display, range_display, reviews_figure, orders_figure, market_share_figure
        
#     except Exception as e:
#         logger.error(f"Error in update_details_tab_figures callback: {str(e)}", exc_info=True)
#         return None, None, "Error updating figures", "Error updating figures", reviews_figure, orders_figure, market_share_figure

if __name__ == "__main__":
    # Check if running in Databricks
    if 'dbutils' in globals():
        # For Databricks environment
        logger.info("Starting app in Databricks environment")
        app.run(host='0.0.0.0', port=8050, debug=True, use_reloader=False)
    else:
        # For local environment
        logger.info("Starting app in local environment")
        app.run(debug=True)