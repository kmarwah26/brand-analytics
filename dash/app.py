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

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

def sqlQuery(query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    cfg = Config()  # Pull environment variables for auth
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


# Generate Sample Data
def generate_sample_data() -> pd.DataFrame:
    """Generate comprehensive sample category review data."""
    np.random.seed(42)
    n_samples = 2000  # Increased sample size
    
    # Generate dates for the past year with more recent dates having higher frequency
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=12)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate categories with different characteristics
    categories = ['Apple Products', 'Industrial & Scientific', 'Health & Personal Care', 'Amazon Devices']
    category_weights = [0.4, 0.3, 0.2, 0.1]  # Weights sum to 1.0
    
    # Generate random data with more realistic patterns
    data = pd.DataFrame({
        'date': np.random.choice(dates, n_samples),
        'category': np.random.choice(categories, n_samples, p=category_weights),
        'source': np.random.choice(['Website', 'Social Media', 'App Store', 'Email'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'rating': np.random.randint(1, 6, n_samples),
        'review_text': [
            f"Sample review text {i} about the category experience and product quality."
            for i in range(n_samples)
        ]
    })
    
    # Add sentiment based on rating with some randomness
    def get_sentiment(rating):
        if rating >= 4:
            return np.random.choice(['Positive', 'Positive', 'Positive', 'Neutral'], p=[0.8, 0.1, 0.05, 0.05])
        elif rating <= 2:
            return np.random.choice(['Negative', 'Negative', 'Negative', 'Neutral'], p=[0.8, 0.1, 0.05, 0.05])
        else:
            return np.random.choice(['Neutral', 'Positive', 'Negative'], p=[0.6, 0.2, 0.2])
    
    data['sentiment'] = data['rating'].apply(get_sentiment)
    
    # Add product attributes
    attributes = ['Quality', 'Price', 'Service', 'Innovation', 'Design']
    for attr in attributes:
        data[f'{attr.lower()}_score'] = np.random.uniform(1, 10, n_samples)
    
    # Add market share data
    data['market_share'] = np.random.uniform(15, 35, n_samples)
    
    # Add competitive metrics
    data['competitive_position'] = np.random.uniform(60, 95, n_samples)
    
    # Add time-based patterns
    data['month'] = data['date'].dt.month
    data['quarter'] = data['date'].dt.quarter
    
    # Add seasonal patterns
    def add_seasonal_pattern(row):
        month = row['month']
        if month in [12, 1, 2]:  # Winter
            return row['rating'] * 1.1
        elif month in [3, 4, 5]:  # Spring
            return row['rating'] * 1.05
        elif month in [6, 7, 8]:  # Summer
            return row['rating'] * 0.95
        else:  # Fall
            return row['rating'] * 1.0
    
    data['seasonal_rating'] = data.apply(add_seasonal_pattern, axis=1)
    
    # Add category-specific patterns
    category_patterns = {
        'Apple Products': {'rating_boost': 1.2, 'sentiment_boost': 0.9},
        'Industrial & Scientific': {'rating_boost': 1.1, 'sentiment_boost': 0.8},
        'Health & Personal Care': {'rating_boost': 0.9, 'sentiment_boost': 0.7},
        'Amazon Devices': {'rating_boost': 0.8, 'sentiment_boost': 0.6}
    }
    
    for category, pattern in category_patterns.items():
        mask = data['category'] == category
        data.loc[mask, 'rating'] = data.loc[mask, 'rating'] * pattern['rating_boost']
        data.loc[mask, 'market_share'] = data.loc[mask, 'market_share'] * pattern['sentiment_boost']
    
    # Ensure ratings stay within 1-5 range
    data['rating'] = data['rating'].clip(1, 5)
    
    # Add review categories
    review_categories = ['Product', 'Service', 'Price', 'Quality', 'Experience']
    data['review_category'] = np.random.choice(review_categories, n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    
    # Add review length
    data['review_length'] = np.random.randint(50, 500, n_samples)
    
    # Add response time (in hours)
    data['response_time'] = np.random.exponential(24, n_samples)  # Mean response time of 24 hours
    
    # Add resolution time (in hours)
    data['resolution_time'] = data['response_time'] + np.random.exponential(48, n_samples)
    
    # Add brand column
    brands_by_category = {
        'Apple Products': ['iPhone', 'iPad', 'MacBook', 'Apple Watch'],
        'Industrial & Scientific': ['Bosch', '3M', 'Honeywell', 'Siemens'],
        'Health & Personal Care': ['Oral-B', 'Philips', 'Braun', 'Panasonic'],
        'Amazon Devices': ['Echo', 'Fire TV', 'Kindle', 'Ring']
    }
    data['brand'] = data['category'].apply(lambda cat: np.random.choice(brands_by_category[cat]))
    
    return data



def load_dummy_data() -> pd.DataFrame:
    """Load sample data."""
    try:
        return generate_sample_data()
    except Exception as e:
        print(f"Data generation failed: {str(e)}")
        return pd.DataFrame()
    


# Fetch the data
try:
    # This example query depends on the nyctaxi data set in Unity Catalog, see https://docs.databricks.com/en/discover/databricks-datasets.html for details
    data = sqlQuery("SELECT * FROM retail_cpg_demo.brand_manager.vw_brand_insights")
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
except Exception as e:
    print(f"An error occurred in querying data: {str(e)}")
    data = pd.DataFrame()

# Convert the date column to a datetime object
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                assets_folder='assets')  # Specify assets folder

# Create assets folder if it doesn't exist
assets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)

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

# Add custom CSS for dropdowns
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dark theme for dropdowns */
            .Select-control {
                background-color: #2d3436 !important;
                border-color: #636e72 !important;
            }
            .Select-menu-outer {
                background-color: #2d3436 !important;
                border-color: #636e72 !important;
            }
            .Select-option {
                background-color: #2d3436 !important;
                color: white !important;
            }
            .Select-option:hover {
                background-color: #636e72 !important;
            }
            .Select-value-label {
                color: white !important;
            }
            .Select-placeholder {
                color: #b2bec3 !important;
            }
            .Select--single > .Select-control .Select-value {
                color: white !important;
            }
            .Select--multi .Select-value {
                background-color: #636e72 !important;
                border-color: #b2bec3 !important;
                color: white !important;
            }
            .Select--multi .Select-value-icon {
                border-color: #b2bec3 !important;
                color: white !important;
            }
            .Select--multi .Select-value-icon:hover {
                background-color: #b2bec3 !important;
                color: #2d3436 !important;
            }
            .Select-arrow {
                border-color: #b2bec3 transparent transparent !important;
            }
            .Select.is-open > .Select-control .Select-arrow {
                border-color: transparent transparent #b2bec3 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read())
        return f'data:image/png;base64,{encoded.decode()}'
    except:
        return None

# Layout
app.layout = dbc.Container([
    dcc.Store(id='page-load-trigger', data=0),
    dbc.Row([dbc.Col(html.H1("Brand Manager", className="text-center my-4 text-light"), width=12)]),
    
    # Filter Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters", style=CARD_HEADER_STYLE),
                dbc.CardBody([
    dbc.Row([
        dbc.Col([
                            dbc.Label("Category", className="text-light"),
                            dcc.Dropdown(
                                id='category-filter',
                                options=[{'label': cat, 'value': cat} for cat in sorted(data['category'].unique())],
                                value=data['category'].iloc[0],
                                className="mb-3"
                            )
                        ], width=4),
        dbc.Col([
                            dbc.Label("Brand", className="text-light"),
                            dcc.Dropdown(
                                id='brand-filter',
                                options=[],  # Will be populated by callback
                                value=None,  # Will be set by callback
                                className="mb-3"
                            )
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Minimum Sentiment Score", className="text-light"),
                            dcc.Slider(
                                id='sentiment-filter',
                                min=0,
                                max=100,
                                step=1,
                                value=0,
                                marks={
                                    0: '0%',
                                    25: '25%',
                                    50: '50%',
                                    75: '75%',
                                    100: '100%'
                                },
                                className="mb-3"
                            ),
                            html.Div(id='sentiment-value', className="text-light text-center")
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
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Brand Health Tracker", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='brand-health-score', className="text-center mb-2", style={'color': '#2ecc71', 'fontSize': '2.5rem'}),
                            html.P("Overall Brand Health Score", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#2d4a3e'})  # Lighter green
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Competitive Positioning", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='competitive-score', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2.5rem'}),
                            html.P("Market Share & Position", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#2d3e4a'})  # Lighter blue
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Product Attribute Analysis", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='product-score', className="text-center mb-2", style={'color': '#9b59b6', 'fontSize': '2.5rem'}),
                            html.P("Product Performance Score", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#3e2d4a'})  # Lighter purple
                ], width=4)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sentiment Distribution", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='sentiment-donut', style={'height': '300px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Key Metrics", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            # Add counters row
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Total Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='total-reviews-counter', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2rem'}),
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#2d3e4a'})  # Lighter blue
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Positive Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='positive-reviews-counter', className="text-center mb-2", style={'color': '#2ecc71', 'fontSize': '2rem'}),
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#2d4a3e'})  # Lighter green
                                ], width=6)
                            ], className="mb-4"),
                            html.Div(id='metrics-container', className="d-flex flex-column gap-3 text-light")
                        ])
                    ], style=CARD_STYLE)
                ], width=6)
            ], className="mb-4")
        ], label="Brand Health", tab_id="tab-brand-health"),
        
        # Product Attribute Tab
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Reviews", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dag.AgGrid(
                                id='reviews-table',
                                columnDefs=[
                                    {"headerName": "Date", "field": "date", "sortable": True, "width": 100},
                                    {"headerName": "Sentiment", "field": "sentiment", "sortable": True, "width": 100},
                                    {"headerName": "Rating", "field": "rating", "sortable": True, "width": 80},
                                    {"headerName": "Category", "field": "category", "sortable": True, "width": 150},
                                    {"headerName": "Source", "field": "source", "sortable": True, "width": 120},
                                    {
                                        "headerName": "Review",
                                        "field": "review_text",
                                        "sortable": True,
                                        "flex": 1,
                                        "minWidth": 300,
                                        "autoHeight": True,
                                        "wrapText": True,
                                        "cellStyle": {
                                            "whiteSpace": "normal",
                                            "lineHeight": "1.5"
                                        }
                                    }
                                ],
                                defaultColDef={
                                    "resizable": True,
                                    "sortable": True,
                                    "filter": True,
                                    "minWidth": 80
                                },
                                dashGridOptions={
                                    "rowSelection": "single",
                                    "theme": "ag-theme-alpine-dark",
                                    "rowHeight": "auto",
                                    "domLayout": "autoHeight"
                                },
                                style={"height": "400px", "width": "100%"}
                            )
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Attribute Analysis", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='attribute-analysis', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Product Attribute", tab_id="tab-product"),
        
        # Competitive Positioning Tab
        dbc.Tab([
            # Brand Comparison Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Brand Comparison", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select Brands to Compare", className="text-light"),
                                    dcc.Dropdown(
                                        id='category1-filter',
                                        options=[
                                            {'label': 'Apple Products', 'value': 'Apple Products'},
                                            {'label': 'Industrial & Scientific', 'value': 'Industrial & Scientific'},
                                            {'label': 'Health & Personal Care', 'value': 'Health & Personal Care'},
                                            {'label': 'Amazon Devices', 'value': 'Amazon Devices'}
                                        ],
                                        value='Apple Products',
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label(" ", className="text-light"),  # Empty label for alignment
                                    dcc.Dropdown(
                                        id='category2-filter',
                                        options=[
                                            {'label': 'Apple Products', 'value': 'Apple Products'},
                                            {'label': 'Industrial & Scientific', 'value': 'Industrial & Scientific'},
                                            {'label': 'Health & Personal Care', 'value': 'Health & Personal Care'},
                                            {'label': 'Amazon Devices', 'value': 'Amazon Devices'}
                                        ],
                                        value='Apple Products',
                                        className="mb-3"
                                    )
                                ], width=6)
                            ]),
                            dcc.Graph(id='brand-comparison-chart', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            
            # Existing Market Share Trends
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Share Trends", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='market-share-trend', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            
            # Existing Competitive Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Competitive Analysis", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Positive Feedback Word Cloud", className="text-center mb-3"),
                                    html.Img(id='positive-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6),
                                dbc.Col([
                                    html.H4("Negative Feedback Word Cloud", className="text-center mb-3"),
                                    html.Img(id='negative-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6)
                            ])
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Competitive Positioning", tab_id="tab-competitive")
    ], id="tabs", active_tab="tab-brand-health", className="mb-4"),
    
], fluid=True, style={'backgroundColor': '#2d3436', 'minHeight': '100vh', 'padding': '20px'})  # Lighter background

# Update the callback to dynamically filter brands based on category
@app.callback(
    Output('brand-filter', 'options'),
    Output('brand-filter', 'value'),
    Input('category-filter', 'value')
)
def update_brand_options(selected_category):
    filtered_data = data[data['category'] == selected_category]
    brands = sorted(filtered_data['brand'].unique())
    options = [{'label': brd, 'value': brd} for brd in brands]
    value = brands[0] if brands else None
    return options, value


# Update the callback to remove image-related code
@app.callback(
    Output('sentiment-donut', 'figure'),
    Output('brand-comparison-chart', 'figure'),
    Output('reviews-table', 'rowData'),
    Output('metrics-container', 'children'),
    Output('brand-health-score', 'children'),
    Output('competitive-score', 'children'),
    Output('product-score', 'children'),
    Output('market-share-trend', 'figure'),
    Output('positive-wordcloud', 'src'),
    Output('negative-wordcloud', 'src'),
    Output('attribute-analysis', 'figure'),
    Output('sentiment-value', 'children'),
    Output('total-reviews-counter', 'children'),
    Output('positive-reviews-counter', 'children'),
    Input('page-load-trigger', 'data'),
    Input('category-filter', 'value'),
    Input('brand-filter', 'value'),
    Input('sentiment-filter', 'value'),
    Input('category1-filter', 'value'),
    Input('category2-filter', 'value')
)
def update_visuals(n_clicks, category, brand, sentiment_threshold, category1, category2):
    #data = load_dummy_data()
    
    # Apply filters
    filtered_data = data[data['category'] == category]
    filtered_data = filtered_data[filtered_data['brand'] == brand]
    # Filter by sentiment threshold
    sentiment_scores = filtered_data['rating'] * 20  # Convert 1-5 rating to 0-100 scale
    filtered_data = filtered_data[sentiment_scores >= sentiment_threshold]
    
    # Generate word clouds
    def generate_wordcloud(texts, max_words=100):
        if not texts:
            return None
        text = ' '.join(texts)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#2d3436',
            colormap='viridis',
            max_words=max_words,
            contour_width=1,
            contour_color='#636e72'
        ).generate(text)
        
        # Convert to base64 for display
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        return f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
    
    # Generate positive and negative word clouds
    positive_reviews = filtered_data[filtered_data['sentiment'] == 'Positive']['review_text'].tolist()
    negative_reviews = filtered_data[filtered_data['sentiment'] == 'Negative']['review_text'].tolist()
    
    positive_wordcloud = generate_wordcloud(positive_reviews)
    negative_wordcloud = generate_wordcloud(negative_reviews)
    
    # Create donut chart for sentiment distribution
    sentiment_counts = filtered_data['sentiment'].value_counts()
    donut_fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.6,
        marker_colors=['#2ecc71', '#e74c3c', '#95a5a6'],
        textinfo='percent+label',
        textposition='outside',
        pull=[0.05, 0, 0]
    )])
    donut_fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.2,
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(
            text="Sentiment Distribution",
            font=dict(size=20, color='white'),
            x=0.5,
            y=0.95
        ),
        margin=dict(t=50, b=50, l=50, r=150)
    )
    
    # Create category comparison chart
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=12)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Generate monthly review counts for selected categories
    def get_monthly_counts(data, category):
        # Filter data for the specific category
        category_data = filtered_data[filtered_data['category'] == category].copy()
        
        # Convert dates to month and count reviews
        category_data['month'] = category_data['date'].dt.to_period('M')
        monthly_counts = category_data.groupby('month').size()
        
        # Create a series with all months, filling missing months with 0
        full_counts = pd.Series(0, index=date_range.to_period('M'))
        full_counts.update(monthly_counts)
        return full_counts
    
    # Get data for both categories
    category1_data = get_monthly_counts(filtered_data, category1)
    category2_data = get_monthly_counts(filtered_data, category2)
    
    # Create the comparison chart
    comparison_fig = go.Figure()
    
    # Add bars for category 1
    comparison_fig.add_trace(go.Bar(
        x=[d.strftime('%Y-%m') for d in date_range],
        y=category1_data.values,
        name=category1,
        marker_color='#3498db',
        opacity=0.8
    ))
    
    # Add bars for category 2
    comparison_fig.add_trace(go.Bar(
        x=[d.strftime('%Y-%m') for d in date_range],
        y=category2_data.values,
        name=category2,
        marker_color='#e74c3c',
        opacity=0.8
    ))
    
    comparison_fig.update_layout(
        title='Monthly Review Comparison',
        barmode='stack',  # Changed to stacked bars
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            title='Month',
            gridcolor='#404040',
            tickangle=45
        ),
        yaxis=dict(
            title='Number of Reviews',
            gridcolor='#404040'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'  # Added unified hover mode
    )
    
    # Add hover template
    comparison_fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                     "Category: %{fullData.name}<br>" +
                     "Reviews: %{y}<br>" +
                     "<extra></extra>"
    )
    
    # Calculate key metrics
    total_reviews = len(filtered_data)
    positive_reviews = len(filtered_data[filtered_data['sentiment'] == 'Positive'])
    positive_pct = (filtered_data['sentiment'] == 'Positive').mean() * 100
    avg_rating = filtered_data['rating'].mean()
    response_rate = np.random.uniform(85, 95)  # Simulated response rate
    resolution_time = np.random.uniform(2, 4)  # Simulated average resolution time in hours
    
    # Get last 5 reviews for the table
    recent_reviews = filtered_data.sort_values('date', ascending=False).head(5)
    
    metrics = [
        html.H4(f"Total Reviews: {total_reviews:,}", style={'color': 'white'}),
        html.H4(f"Positive Sentiment: {positive_pct:.1f}%", style={'color': 'white'}),
        html.H4(f"Average Rating: {avg_rating:.1f}/5.0", style={'color': 'white'}),
        html.H4(f"Response Rate: {response_rate:.1f}%", style={'color': 'white'}),
        html.H4(f"Avg. Resolution Time: {resolution_time:.1f} hours", style={'color': 'white'})
    ]
    
    # Calculate brand health scores
    brand_health = (filtered_data['sentiment'] == 'Positive').mean() * 100
    competitive_position = np.random.uniform(75, 95)  # Simulated competitive position
    product_score = np.random.uniform(80, 98)  # Simulated product score
    
    # Format the scores
    brand_health_score = f"{brand_health:.1f}%"
    competitive_score = f"{competitive_position:.1f}%"
    product_score = f"{product_score:.1f}%"
    
    # Create market share trend
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    market_share = pd.DataFrame({
        'date': dates,
        'market_share': np.random.uniform(20, 30, len(dates))
    })
    market_share_fig = px.line(
        market_share,
        x='date',
        y='market_share',
        title='Market Share Trend'
    )
    market_share_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#404040'),
        yaxis=dict(gridcolor='#404040')
    )
    
    # Create attribute analysis
    attributes = ['Quality', 'Price', 'Service', 'Innovation', 'Design']
    importance = np.random.uniform(0.7, 1.0, len(attributes))
    satisfaction = np.random.uniform(0.6, 0.9, len(attributes))
    
    attribute_fig = go.Figure()
    attribute_fig.add_trace(go.Bar(
        x=attributes,
        y=importance,
        name='Importance'
    ))
    attribute_fig.add_trace(go.Bar(
        x=attributes,
        y=satisfaction,
        name='Satisfaction'
    ))
    attribute_fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#404040'),
        yaxis=dict(gridcolor='#404040')
    )
    
    # Add sentiment value display
    sentiment_value = f"Current Threshold: {sentiment_threshold}%"
    
    return (
        donut_fig,
        comparison_fig,
        recent_reviews.to_dict('records'),
        metrics,
        brand_health_score,
        competitive_score,
        product_score,
        market_share_fig,
        positive_wordcloud,
        negative_wordcloud,
        attribute_fig,
        sentiment_value,
        f"{total_reviews:,}",
        f"{positive_reviews:,}"
    )

if __name__ == "__main__":
    app.run(debug=True)