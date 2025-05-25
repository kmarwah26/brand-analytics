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

# # Ensure environment variable is set correctly
# assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# # Load data from SQL
# def sqlQuery(query: str) -> pd.DataFrame:
#     """Execute a SQL query and return the result as a pandas DataFrame."""
#     cfg = Config()  # Pull environment variables for auth
#     with sql.connect(
#         server_hostname=cfg.host,
#         http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
#         credentials_provider=lambda: cfg.authenticate
#     ) as connection:
#         with connection.cursor() as cursor:
#             cursor.execute(query)
#             return cursor.fetchall_arrow().to_pandas()

# data = sqlQuery("SELECT * FROM retail_cpg_demo.brand_manager.vw_brand_insights")

# Load data from CSV
try:
    data = pd.read_csv('app_data/brand_insights_data.csv')
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    
    # Convert the date column to a datetime object
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Filter out specific brands
    excluded_brands = [
        'Hasbro',
        'Hasbro Gaming',
        'Hot Wheels',
        'L.O.L. Surprise!',
        'Learning Resources',
        'Ravensburger',
        'Schleich',
        'VTech',
        'Crayola',
        'Fisher-Price',
        'PASOW',
        'TR Industrial',
        'GE',
        'Bolt Dropper',
        'Energizer',
        'Cambridge Resources',
        'Indoor Tactics',
        'JOTO',
        'D-Line',
        'Wrap-It Storage',
        'OHill',
        'iFixit',
        'Bayco',
        'Gorilla',
        'DEWALT'
    ]
    data = data[~data['brand'].isin(excluded_brands)]
    
    # Combine Mattel brands
    data['brand'] = data['brand'].replace('Mattel Games', 'Mattel')
    
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
        
except Exception as e:
    print(f"An error occurred loading data: {str(e)}")
    data = pd.DataFrame()

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
            
            /* Chatbot icon styles */
            .chatbot-icon {
                position: fixed;
                bottom: 30px;
                right: 30px;
                width: 60px;
                height: 60px;
                background-color: #3498db;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                cursor: pointer;
                transition: transform 0.2s ease-in-out;
                z-index: 1000;
            }
            
            .chatbot-icon:hover {
                transform: scale(1.1);
                background-color: #2980b9;
            }
            
            .chatbot-icon svg {
                width: 30px;
                height: 30px;
                fill: white;
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
        <a href="/chat" target="_blank" class="chatbot-icon" title="Open Chatbot">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
            </svg>
        </a>
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
    
    # AI Agent Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Brand AI Agent"), style={'backgroundColor': '#2d3436', 'color': 'white'}),
        dbc.ModalBody([
            html.Div([
                html.H5("Analyzing Most Recent Reviews", style={'color': 'white', 'marginBottom': '20px'}),
                html.Div(id='ai-loading-message', style={'color': 'white', 'marginBottom': '20px'}),
                html.Div(id='ai-loading-spinner', style={'marginBottom': '20px'}),
                html.Div(id='ai-analysis-content', style={'marginTop': '20px'})
            ])
        ], style={'backgroundColor': '#2d3436'}),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-ai-modal", className="ms-auto", n_clicks=0),
            style={'backgroundColor': '#2d3436'}
        ),
    ], id="ai-agent-modal", size="lg", is_open=False),
    
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
                                value='Toys & Games',
                                className="mb-3"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Brand", className="text-light"),
                            dcc.Dropdown(
                                id='brand-filter',
                                options=[],  # Will be populated by callback
                                value=None,  # Will be set by callback
                                className="mb-3"
                            )
                        ], width=6)
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
                        dbc.CardHeader("Key Metrics", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            # Add counters row
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Total Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='total-reviews-counter', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2rem'}),
                                            html.A(
                                                "Competitive Trends",
                                                href="#positive-wordcloud",
                                                id="total-reviews-link",
                                                className="text-center d-block",
                                                style={
                                                    'color': '#3498db',
                                                    'textDecoration': 'none',
                                                    'fontSize': '0.9rem',
                                                    'marginTop': '5px'
                                                }
                                            )
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#2d3e4a'})  # Lighter blue
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Positive Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='positive-reviews-counter', className="text-center mb-2", style={'color': '#2ecc71', 'fontSize': '2rem'}),
                                            html.A(
                                                "Trends",
                                                href="#tabs",
                                                id="positive-reviews-link",
                                                className="text-center d-block",
                                                style={
                                                    'color': '#2ecc71',
                                                    'textDecoration': 'none',
                                                    'fontSize': '0.9rem',
                                                    'marginTop': '5px'
                                                }
                                            )
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#2d4a3e'})  # Lighter green
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Neutral Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='neutral-reviews-counter', className="text-center mb-2", style={'color': '#f1c40f', 'fontSize': '2rem'}),
                                            html.A(
                                                "View Attributes",
                                                href="#tabs",
                                                id="neutral-reviews-link",
                                                className="text-center d-block",
                                                style={
                                                    'color': '#f1c40f',
                                                    'textDecoration': 'none',
                                                    'fontSize': '0.9rem',
                                                    'marginTop': '5px'
                                                }
                                            )
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#4a3e2d'})  # Lighter yellow
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Negative Reviews", className="text-center mb-2", style={'color': 'white'}),
                                            html.H2(id='negative-reviews-counter', className="text-center mb-2", style={'color': '#e74c3c', 'fontSize': '2rem'}),
                                            html.A(
                                                "Summary",
                                                href="#tabs",
                                                id="negative-reviews-link",
                                                className="text-center d-block",
                                                style={
                                                    'color': '#e74c3c',
                                                    'textDecoration': 'none',
                                                    'fontSize': '0.9rem',
                                                    'marginTop': '5px'
                                                }
                                            )
                                        ])
                                    ], style={**CARD_STYLE, 'backgroundColor': '#4a2d2d'})  # Lighter red
                                ], width=3)
                            ], className="mb-4"),
                            html.Div(id='metrics-container', className="d-flex flex-column gap-3 text-light")
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            
            # Sentiment Analysis Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sentiment Analysis", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='sentiment-treemap', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Brand Health", tab_id="tab-brand-health"),
        
        # Product Attribute Tab
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.Span("Monthly Review Trends", style={'flex': '1'}),
                                dbc.Button(
                                    "AI Agent",
                                    id="open-ai-modal",
                                    color="primary",
                                    className="ms-auto",
                                    style={'backgroundColor': '#3498db', 'borderColor': '#3498db'}
                                )
                            ], style={'display': 'flex', 'alignItems': 'center'})
                        ], style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='monthly-reviews-chart', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Brand Review Attributes", style=CARD_HEADER_STYLE),
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
        ], label="Product Attribute", tab_id="tab-product"),
        
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
                        dbc.CardHeader("Category Review Attributes", style=CARD_HEADER_STYLE),
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


# Update the callback with better error handling and data validation
@app.callback(
    Output('sentiment-treemap', 'figure'),
    Output('monthly-reviews-chart', 'figure'),
    Output('metrics-container', 'children'),
    Output('brand-health-score', 'children'),
    Output('competitive-score', 'children'),
    Output('product-score', 'children'),
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
    Input('page-load-trigger', 'data'),
    Input('category-filter', 'value'),
    Input('brand-filter', 'value')
)
def update_visuals(n_clicks, category, brand):
    # Create empty figure template
    def create_empty_fig():
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    # Return default values if filters are not set
    if category is None or brand is None:
        empty_fig = create_empty_fig()
        return (
            empty_fig,  # sentiment-treemap
            empty_fig,  # monthly-reviews-chart
            [],  # metrics-container
            "0%",  # brand-health-score
            "0%",  # competitive-score
            "0%",  # product-score
            empty_fig,  # market-share-trend
            empty_fig,  # pricing-comparison
            None,  # positive-wordcloud
            None,  # negative-wordcloud
            empty_fig,  # brand-positive-wordcloud
            empty_fig,  # brand-negative-wordcloud
            "0",  # total-reviews-counter
            "0",  # positive-reviews-counter
            "0",  # neutral-reviews-counter
            "0"   # negative-reviews-counter
        )

    try:
        # Apply filters
        category_data = data[data['category'] == category]
        filtered_data = category_data[category_data['brand'] == brand]
        
        if len(filtered_data) == 0:
            empty_fig = create_empty_fig()
            return (
                empty_fig,  # sentiment-treemap
                empty_fig,  # monthly-reviews-chart
                [],  # metrics-container
                "0%",  # brand-health-score
                "0%",  # competitive-score
                "0%",  # product-score
                empty_fig,  # market-share-trend
                empty_fig,  # pricing-comparison
                None,  # positive-wordcloud
                None,  # negative-wordcloud
                empty_fig,  # brand-positive-wordcloud
                empty_fig,  # brand-negative-wordcloud
                "0",  # total-reviews-counter
                "0",  # positive-reviews-counter
                "0",  # neutral-reviews-counter
                "0"   # negative-reviews-counter
            )

        # Create sentiment treemap
        sentiment_counts = filtered_data['sentiment'].value_counts()
        
        # Create treemap with sentiment data
        sentiment_treemap = go.Figure(go.Treemap(
            ids=sentiment_counts.index,
            labels=sentiment_counts.index,
            parents=[''] * len(sentiment_counts),
            values=sentiment_counts.values,
            textinfo="label+value+percent parent",
            marker=dict(
                colors=['#2ecc71' if filtered_data[filtered_data['sentiment'] == s]['sentiment_score'].mean() > 3 
                       else '#f1c40f' if filtered_data[filtered_data['sentiment'] == s]['sentiment_score'].mean() == 3 
                       else '#e74c3c' 
                       for s in sentiment_counts.index],
                line=dict(width=2, color='#2d3436')
            ),
            textfont=dict(color='white', size=14)
        ))
        
        sentiment_treemap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, l=25, r=25, b=25),
            font=dict(color='white'),
            title=dict(
                text="Sentiment Distribution",
                font=dict(size=20, color='white'),
                x=0.5,
                y=0.95
            )
        )
        
        # Generate word clouds
        def generate_wordcloud(texts, max_words=100, background_color='#2d3436'):
            if not texts:
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
                    max_words=max_words,
                    contour_width=1,
                    contour_color='#636e72'
                ).generate(text)
                
                # Convert to base64 for display
                img = io.BytesIO()
                wordcloud.to_image().save(img, format='PNG')
                img.seek(0)
                return f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
            except Exception as e:
                print(f"Error generating wordcloud: {str(e)}")
                return None
        
        # Get reviews from all other brands in the selected category
        category_data = data[data['category'] == category]  # Filter by selected category
        other_brands_data = category_data[category_data['brand'] != brand]  # Filter out selected brand
        
        # Generate word clouds for competitor analysis
        positive_reviews = other_brands_data['positive_feature_list'].str.split(',').explode().tolist()
        negative_reviews = other_brands_data['negative_feature_list'].str.split(',').explode().tolist()
        
        positive_wordcloud = generate_wordcloud(positive_reviews, background_color='white')  # White background
        negative_wordcloud = generate_wordcloud(negative_reviews, background_color='#636e72')  # Grey background
        
        # Generate word clouds for brand analysis
        brand_positive_reviews = filtered_data['positive_feature_list'].str.split(',').explode().tolist()
        brand_negative_reviews = filtered_data['negative_feature_list'].str.split(',').explode().tolist()
        
        brand_positive_wordcloud = generate_wordcloud(brand_positive_reviews, background_color='white')  # White background
        brand_negative_wordcloud = generate_wordcloud(brand_negative_reviews, background_color='#636e72')  # Grey background
        
        # Create market share trend
        # Create Share of Voice chart
        # Group by month and brand to get review counts
        monthly_brand_counts = category_data.groupby([
            pd.to_datetime(category_data['date']).dt.to_period('M'),
            'brand'
        ]).size().reset_index(name='review_count')
        
        # Convert period to timestamp for plotting
        monthly_brand_counts['date'] = monthly_brand_counts['date'].dt.to_timestamp()
        
        # Create stacked area chart
        market_share_fig = go.Figure()
        
        # Add area for each brand
        for brand_name in category_data['brand'].unique():
            brand_data = monthly_brand_counts[monthly_brand_counts['brand'] == brand_name]
            market_share_fig.add_trace(go.Scatter(
                x=brand_data['date'],
                y=brand_data['review_count'],
                name=brand_name,
                stackgroup='one',
                fill='tonexty',
                line=dict(width=0.5),
                hovertemplate='<b>%{x|%B %Y}</b><br>' +
                             'Brand: %{fullData.name}<br>' +
                             'Reviews: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        market_share_fig.update_layout(
            title='Share of Voice',
            xaxis_title='Month',
            yaxis_title='Number of Reviews',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(
                gridcolor='#404040',
                tickformat='%B %Y'
            ),
            yaxis=dict(
                gridcolor='#404040',
                showgrid=True
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            )
        )

        # Create price analysis chart
        price_fig = go.Figure()

        brands = category_data['brand'].unique()
        colors = ['#FFA500' if b == brand else '#3498db' for b in brands]
        
        price_fig.add_trace(go.Bar(
            x=brands,
            y=category_data.groupby('brand')['avg_brand_price'].mean(),
            name='Brand Average Price',
            marker_color=colors
        ))
        
        category_avg = category_data['avg_brand_price'].mean()
        price_fig.add_trace(go.Scatter(
            x=category_data['brand'].unique(),
            y=[category_avg] * len(category_data['brand'].unique()),
            name='Category Average',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            mode='lines'
        ))
        
        price_fig.update_layout(
            title='Brand Price Comparison',
            xaxis_title='Brand',
            yaxis_title='Average Price',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#404040'),
            yaxis=dict(gridcolor='#404040'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            )
        )

        # Create monthly sentiment bar chart
        monthly_sentiment = filtered_data.groupby([
            pd.to_datetime(filtered_data['date']).dt.to_period('M'),
            pd.cut(filtered_data['sentiment_score'], 
                  bins=[0, 2, 3, 5], 
                  labels=['Negative (<3)', 'Neutral (3)', 'Positive (>3)'])
        ]).size().unstack(fill_value=0)
        
        monthly_sentiment.index = monthly_sentiment.index.astype(str)
        
        monthly_reviews_chart = go.Figure()
        
        # Add bars for each sentiment score range
        colors = {
            'Positive (>3)': '#2ecc71',
            'Neutral (3)': '#f1c40f',
            'Negative (<3)': '#e74c3c'
        }
        
        for sentiment in monthly_sentiment.columns:
            monthly_reviews_chart.add_trace(go.Bar(
                name=sentiment,
                x=monthly_sentiment.index,
                y=monthly_sentiment[sentiment],
                marker_color=colors.get(sentiment, '#95a5a6'),
                text=monthly_sentiment[sentiment],  # Add data labels
                textposition='auto',  # Automatically position labels
                textfont=dict(
                    color='white',
                    size=14
                )
            ))
        
        monthly_reviews_chart.update_layout(
            title='Total Reviews',
            xaxis_title='Month',
            yaxis_title='Number of Reviews',
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(
                gridcolor='#404040',
                tickangle=45
            ),
            yaxis=dict(
                gridcolor='#404040',
                showgrid=True
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            ),
            uniformtext_minsize=8,  # Minimum font size for labels
            uniformtext_mode='hide'  # Hide labels if they don't fit
        )

        # Calculate metrics
        total_reviews = len(filtered_data)
        positive_reviews_count = len(filtered_data[filtered_data['sentiment_score'] >= 3])
        neutral_reviews_count = len(filtered_data[(filtered_data['sentiment_score'] >= 2) & (filtered_data['sentiment_score'] < 3)])
        negative_reviews_count = len(filtered_data[filtered_data['sentiment_score'] < 2])
        
        # Calculate scores
        brand_health = (filtered_data['sentiment_score'] >= 3).mean() * 100
        competitive_position = np.random.uniform(75, 95)
        product_score = np.random.uniform(80, 98)
        
        # Determine color based on brand health score
        if brand_health >= 80:
            health_color = '#2ecc71'  # Green
        elif brand_health >= 70:
            health_color = '#f1c40f'  # Yellow
        else:
            health_color = '#e74c3c'  # Red
            
        metrics = [
            html.H4(f"Total Reviews: {total_reviews:,}", style={'color': 'white'}),
            html.H4(f"Positive Sentiment: {positive_reviews_count / total_reviews * 100:.1f}%", style={'color': 'white'}),
            html.H4(f"Neutral Sentiment: {neutral_reviews_count / total_reviews * 100:.1f}%", style={'color': 'white'}),
            html.H4(f"Negative Sentiment: {negative_reviews_count / total_reviews * 100:.1f}%", style={'color': 'white'}),
            html.H4(f"Average Rating: {filtered_data['rating'].mean():.1f}/5.0", style={'color': 'white'}),
            html.H4(f"Average Sentiment Score: {filtered_data['sentiment_score'].mean():.1f}/5.0", style={'color': 'white'})
        ]
        
        return (
            sentiment_treemap,
            monthly_reviews_chart,
            metrics,
            html.Span(f"{brand_health:.1f}%", style={'color': health_color}),  # Updated brand health score with color
            f"{competitive_position:.1f}%",
            f"{product_score:.1f}%",
            market_share_fig,
            price_fig,
            positive_wordcloud,
            negative_wordcloud,
            brand_positive_wordcloud,
            brand_negative_wordcloud,
            f"{total_reviews:,}",
            f"{positive_reviews_count:,}",
            f"{neutral_reviews_count:,}",
            f"{negative_reviews_count:,}"
        )
        
    except Exception as e:
        print(f"Error in callback: {str(e)}")
        empty_fig = create_empty_fig()
        return (
            empty_fig,  # sentiment-treemap
            empty_fig,  # monthly-reviews-chart
            [],  # metrics-container
            "0%",  # brand-health-score
            "0%",  # competitive-score
            "0%",  # product-score
            empty_fig,  # market-share-trend
            empty_fig,  # pricing-comparison
            None,  # positive-wordcloud
            None,  # negative-wordcloud
            empty_fig,  # brand-positive-wordcloud
            empty_fig,  # brand-negative-wordcloud
            "0",  # total-reviews-counter
            "0",  # positive-reviews-counter
            "0",  # neutral-reviews-counter
            "0"   # negative-reviews-counter
        )

# Update the callback to handle all tab switching
@app.callback(
    Output('tabs', 'active_tab'),
    Input('positive-reviews-link', 'n_clicks'),
    Input('neutral-reviews-link', 'n_clicks'),
    Input('negative-reviews-link', 'n_clicks'),
    Input('total-reviews-link', 'n_clicks'),
    prevent_initial_call=True
)
def switch_to_product_tab(pos_clicks, neu_clicks, neg_clicks, total_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'total-reviews-link':
        return 'tab-competitive'
    elif button_id in ['positive-reviews-link', 'neutral-reviews-link', 'negative-reviews-link']:
        return 'tab-product'
    return dash.no_update

# Add the callback for the AI Agent modal
@app.callback(
    Output("ai-agent-modal", "is_open"),
    [Input("open-ai-modal", "n_clicks"),
     Input("close-ai-modal", "n_clicks")],
    [State("ai-agent-modal", "is_open")],
)
def toggle_ai_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Add callback to handle loading animation and messages
@app.callback(
    [Output("ai-loading-message", "children"),
     Output("ai-loading-spinner", "children"),
     Output("ai-analysis-content", "children")],
    [Input("ai-agent-modal", "is_open"),
     Input('category-filter', 'value'),
     Input('brand-filter', 'value')]
)
def update_ai_loading(is_open, category, brand):
    if not is_open:
        return "", "", []
    
    # Initial loading state
    if is_open:
        # Show loading spinner and initial message
        loading_spinner = dbc.Spinner(
            color="primary",
            size="lg",
            type="grow",
            fullscreen=False,
            spinner_style={"width": "3rem", "height": "3rem"}
        )
        
        # Return initial loading state
        return "Analyzing...", loading_spinner, []
    
    # After 5 seconds, show connecting message
    #time.sleep(5)
    loading_spinner = dbc.Spinner(
        color="primary",
        size="lg",
        type="grow",
        fullscreen=False,
        spinner_style={"width": "3rem", "height": "3rem"}
    )
    
    #time.sleep(3)
    try:
        # Filter data
        filtered_data = data[(data['category'] == category) & (data['brand'] == brand)]
        
        # Calculate metrics
        total_reviews = len(filtered_data)
        positive_reviews = len(filtered_data[filtered_data['sentiment_score'] > 3])
        neutral_reviews = len(filtered_data[filtered_data['sentiment_score'] == 3])
        negative_reviews = len(filtered_data[filtered_data['sentiment_score'] < 3])
        
        # Create analysis content
        analysis_content = [
            html.Div([
                html.H6("Key Insights:", style={'color': 'white', 'marginBottom': '15px'}),
                html.Ul([
                    html.Li(f"Total Reviews: {total_reviews:,}", style={'color': 'white', 'marginBottom': '10px'}),
                    html.Li(f"Positive Sentiment: {positive_reviews/total_reviews*100:.1f}%", style={'color': 'white', 'marginBottom': '10px'}),
                    html.Li(f"Neutral Sentiment: {neutral_reviews/total_reviews*100:.1f}%", style={'color': 'white', 'marginBottom': '10px'}),
                    html.Li(f"Negative Sentiment: {negative_reviews/total_reviews*100:.1f}%", style={'color': 'white', 'marginBottom': '10px'})
                ]),
                html.H6("Recommendations:", style={'color': 'white', 'marginTop': '20px', 'marginBottom': '15px'}),
                html.Ul([
                    html.Li("Monitor negative sentiment trends closely", style={'color': 'white', 'marginBottom': '10px'}),
                    html.Li("Focus on maintaining positive customer experiences", style={'color': 'white', 'marginBottom': '10px'}),
                    html.Li("Address any recurring issues in negative reviews", style={'color': 'white', 'marginBottom': '10px'})
                ])
            ])
        ]
        
        return "", "", analysis_content
        
    except Exception as e:
        return "", "", [html.P(f"Error generating analysis: {str(e)}", style={'color': 'white'})]

if __name__ == "__main__":
    app.run(debug=True)