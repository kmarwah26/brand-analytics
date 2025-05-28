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

# print(".....")
# print(os.getcwd())

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID') 


# Load data from SQL
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
try:
    #data = sqlQuery("SELECT * FROM retail_cpg_demo.brand_manager.vw_brand_insights_toys")

    # Load data from CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'app_data', 'brand_insights_data.csv')
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    
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
 

    data.loc[data['sentiment'].isna(), 'sentiment'] = data.loc[data['sentiment'].isna(), 'sentiment_score'].apply(
        lambda x: 'Love' if x == 5 else 'Great' if x == 4 else 'Fine' if x == 3 else 'Disappointed' if x == 2 else 'Bad' if x == 1 else 'Neutral'
    )
    
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
        'DEWALT',
        'Mattel',
        'Mattel Games'
    ]
    data = data[~data['brand'].isin(excluded_brands)]
    
    # Remove the Mattel brand combination since we're excluding Mattel
    # data['brand'] = data['brand'].replace('Mattel Games', 'Mattel')
    
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

# Add custom CSS for animationsee
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

            /* Email analysis animation */
            .email-analysis-container {
                opacity: 0;
                transform: translateY(20px);
                transition: all 0.3s ease-in-out;
            }

            .email-analysis-container.show {
                opacity: 1;
                transform: translateY(0);
            }

            .email-analysis-content {
                background-color: #2d3436;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease-in-out;
            }

            .email-analysis-content:hover {
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
                transform: translateY(-2px);
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
    dbc.Row([
        dbc.Col([
            html.H1("Brand Manager", className="text-center my-4 text-light"),
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
    
    # AI Agent Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Brand AI Agent"), style={'backgroundColor': '#2d3436', 'color': 'white'}),
        dbc.ModalBody([
            html.Div([
                html.H5("Insights", style={'color': 'white', 'marginBottom': '20px'}),
                html.Div(id='ai-loading-message', style={'color': 'white', 'marginBottom': '20px'}),
                html.Div(id='ai-loading-spinner', style={'marginBottom': '20px'}),
                html.Div(id='ai-analysis-content', style={'marginTop': '20px'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Quick Tip",
                            id="recommendations-button",
                            color="success",
                            className="mt-3 me-2",
                            style={'backgroundColor': '#2ecc71', 'borderColor': '#2ecc71'}
                        ),
                        dbc.Button(
                            "Next Steps",
                            id="analyze-button",
                            color="primary",
                            className="mt-3",
                            style={'backgroundColor': '#3498db', 'borderColor': '#3498db'}
                        )
                    ], width=12)
                ]),
                html.Div(
                    id='email-analysis',
                    style={
                        'marginTop': '20px',
                        'padding': '20px',
                        'backgroundColor': '#636e72',
                        'borderRadius': '5px',
                        'display': 'none'  # Hidden by default
                    }
                )
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
                                value= None,  # Set default value
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
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Competitive Positioning", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='competitive-score', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2.5rem'}),
                            html.P("Market Share & Position", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#2d3e4a'})  # Lighter blue
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Product Attributes", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='product-score', className="text-center mb-2", style={'color': '#9b59b6', 'fontSize': '2.5rem'}),
                            html.P("Product Performance Score", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#3e2d4a'})  # Lighter purple
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Average Rating", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='average-rating', className="text-center mb-2", style={'color': '#f1c40f', 'fontSize': '2.5rem'}),
                            html.P("Customer Satisfaction Score", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#4a3e2d'})  # Lighter yellow
                ], width=3)
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
                        dbc.CardHeader("Sentiment Analysis", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='sentiment-treemap', style={'height': '600px'})
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
                        dbc.CardHeader(id="brand-review-header", style=CARD_HEADER_STYLE),
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
            "0%",  # brand-health-score
            "0%",  # competitive-score
            "0%",  # product-score
            "0.0",  # average-rating
            empty_fig,  # market-share-trend
            empty_fig,  # pricing-comparison
            None,  # positive-wordcloud
            None,  # negative-wordcloud
            None,  # brand-positive-wordcloud
            None,  # brand-negative-wordcloud
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
                "0%",  # brand-health-score
                "0%",  # competitive-score
                "0%",  # product-score
                "0.0",  # average-rating
                empty_fig,  # market-share-trend
                empty_fig,  # pricing-comparison
                None,  # positive-wordcloud
                None,  # negative-wordcloud
                None,  # brand-positive-wordcloud
                None,  # brand-negative-wordcloud
                "0",  # total-reviews-counter
                "0",  # positive-reviews-counter
                "0",  # neutral-reviews-counter
                "0"   # negative-reviews-counter
            )

        # Create sentiment visualization
        sentiment_counts = filtered_data['sentiment'].value_counts()
        
        # Define color mapping for each sentiment with modern shades
        sentiment_colors = {
            'Love': '#10e380',      # Material Green
            'Great': '#26c77b',     # Light Green
            'Positive': '#30b375',     # Light Green
            'Fine': '#4f6159',      # Material Amber
            'Disappointed': '#c45b1d', # Material Orange
            'Bad': '#c4281d',        # Material Red
            'Negative': '#fc1505'        # Material Red
        }
        
        # Create treemap with sentiment data
        sentiment_fig = go.Figure(go.Treemap(
            ids=sentiment_counts.index,
            labels=sentiment_counts.index,
            parents=[''] * len(sentiment_counts),
            values=sentiment_counts.values,
            marker=dict(
                colors=[sentiment_colors.get(s, '#808080') for s in sentiment_counts.index],
                line=dict(width=2, color='#2d3436')
            ),
            textinfo="label+value+percent parent",
            textfont=dict(color='white', size=14),
            hovertemplate='<span style="color: #2d3436">%{label}</span><br>' +
                         'Count: %{value}<br>' +
                         'Percentage: %{percentParent:.1%}<br>' +
                         '<extra></extra>'
        ))
        
        sentiment_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=25, l=25, r=25, b=25),
            font=dict(color='white'),
            # title=dict(
            #     text="Sentiment Distribution",
            #     font=dict(size=20, color='white'),
            #     x=0.5,
            #     y=0.95
            # ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            )
        )

        # Generate word clouds
        def generate_wordcloud(texts, max_words=100, background_color='#2d3436'):
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
        positive_reviews = other_brands_data['positive_feature_list'].dropna().tolist()
        negative_reviews = other_brands_data['negative_feature_list'].dropna().tolist()
        
        positive_wordcloud = generate_wordcloud(positive_reviews, background_color='white')  # White background
        negative_wordcloud = generate_wordcloud(negative_reviews, background_color='#636e72')  # Grey background
        
        # Generate word clouds for brand analysis
        brand_positive_reviews = filtered_data['positive_feature_list'].dropna().tolist()
        brand_negative_reviews = filtered_data['negative_feature_list'].dropna().tolist()
        
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
                text=brand_data['review_count'].apply(lambda x: f'{x:,}'),
                textposition='top center',
                textfont=dict(
                    color='#2d3436',
                    size=11
                ),
                hovertemplate='<span style="color: #2d3436">%{x|%B %Y}</span><br>' +
                             '<span style="color: #2d3436">Brand: %{fullData.name}</span><br>' +
                             '<span style="color: #2d3436">Reviews: %{y:,}</span><br>' +
                             '<extra></extra>',
                hoverlabel=dict(
                    bgcolor='white',
                    font=dict(
                        family='Arial',
                        size=14,
                        color='#2d3436'
                    ),
                    bordercolor='#636e72'
                )
            ))
        
        market_share_fig.update_layout(
            #title='Share of Voice',
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
            #title='Brand Price Comparison',
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

        # Create monthly sentiment line chart
        monthly_sentiment = filtered_data.groupby([
            pd.to_datetime(filtered_data['date']).dt.to_period('M'),
            pd.cut(filtered_data['sentiment_score'], 
                  bins=[0, 2.9, 3.1, 5], 
                  labels=['Negative (<3)', 'Neutral (=3)', 'Positive (>3)'])
        ]).size().unstack(fill_value=0)
        
        monthly_sentiment.index = monthly_sentiment.index.astype(str)
        
        # Calculate mean and standard deviation for each sentiment category
        sentiment_stats = {}
        for sentiment in monthly_sentiment.columns:
            mean = monthly_sentiment[sentiment].mean()
            std = monthly_sentiment[sentiment].std()
            sentiment_stats[sentiment] = {
                'mean': mean,
                'std': std,
                'threshold': mean + (1.5 * std)
            }

        # Define colors for sentiment categories
        sentiment_colors = {
            'Positive (>3)': '#10e380',  # Bright Green
            'Neutral (=3)': '#4f6159',   # Dark Grey
            'Negative (<3)': '#c4281d'   # Red
        }
        
        monthly_reviews_chart = go.Figure()
        
        # Add lines for each sentiment category
        for sentiment in monthly_sentiment.columns:
            # Create custom hover text that includes statistical flags
            hover_text = []
            for month, value in monthly_sentiment[sentiment].items():
                stats = sentiment_stats[sentiment]
                if value > stats['threshold']:
                    hover_text.append(
                        f"<span style='color: #2d3436'>Month: {month}<br>" +
                        f"Count: {value}<br>" +
                        f"⚠️ Spike Detected</span>"
                    )
                else:
                    hover_text.append(
                        f"<span style='color: #2d3436'>Month: {month}<br>" +
                        f"Count: {value}</span>"
                    )
            
            monthly_reviews_chart.add_trace(go.Scatter(
                name=sentiment,
                x=monthly_sentiment.index,
                y=monthly_sentiment[sentiment],
                mode='lines+markers+text',
                line=dict(
                    color=sentiment_colors.get(sentiment, '#808080'),
                    width=3
                ),
                marker=dict(
                    size=8,
                    color=sentiment_colors.get(sentiment, '#808080'),
                    line=dict(width=2, color='#1a1a1a')
                ),
                text=monthly_sentiment[sentiment].apply(lambda x: f'{x:,}'),
                textposition='top center',
                textfont=dict(
                    color='white',
                    size=11
                ),
                hovertext=hover_text,
                hoverinfo='text'
            ))
        
        monthly_reviews_chart.update_layout(
            #title='Monthly Review Trends',
            xaxis_title='Month',
            yaxis_title='Number of Reviews',
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
            ),
            margin=dict(t=40, l=25, r=25, b=50)
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
            
        return (
            sentiment_fig,
            monthly_reviews_chart,
            html.Span(f"{brand_health:.1f}%", style={'color': health_color}),  # Updated brand health score with color
            f"{competitive_position:.1f}%",
            f"{product_score:.1f}%",
            f"{filtered_data['rating'].mean():.1f}/5.0",
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
            "0%",  # brand-health-score
            "0%",  # competitive-score
            "0%",  # product-score
            "0.0",  # average-rating
            empty_fig,  # market-share-trend
            empty_fig,  # pricing-comparison
            None,  # positive-wordcloud
            None,  # negative-wordcloud
            None,  # brand-positive-wordcloud
            None,  # brand-negative-wordcloud
            "0",  # total-reviews-counter
            "0",  # positive-reviews-counter
            "0",  # neutral-reviews-counter
            "0"   # negative-reviews-counter
        )

# Update the callback to handle all tab switching
@app.callback(
    Output('tabs', 'active_tab'),
    Input('open-ai-modal', 'n_clicks'),
    prevent_initial_call=True
)
def switch_to_product_tab(ai_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'open-ai-modal':
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

# Update the callback content for brand-specific tips and next steps
@app.callback(
    [Output("ai-analysis-content", "children"),
     Output("ai-loading-message", "children"),
     Output("ai-loading-spinner", "children"),
     Output("email-analysis", "children"),
     Output("email-analysis", "style"),
     Output("email-analysis", "className")],
    [Input("recommendations-button", "n_clicks"),
     Input("analyze-button", "n_clicks")],
    [State('category-filter', 'value'),
     State('brand-filter', 'value')],
    prevent_initial_call=True
)
def handle_ai_actions(recommendations_clicks, analyze_clicks, category, brand):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [], "", None, None, {'display': 'none'}, 'email-analysis-container'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        # Filter data
        filtered_data = data[(data['category'] == category) & (data['brand'] == brand)]
        
        if button_id == "recommendations-button":
            # Brand-specific quick tips
            if brand == "Barbie":
                recommendations_content = html.Div([
                    html.H6("Quick Tips", style={'color': 'white', 'marginBottom': '15px'}),
                    html.Ul([
                        html.Li("Positive customer experience trends indicate strong brand loyalty. Consider expanding the customer engagement program.", 
                               style={'color': 'white', 'marginBottom': '10px'})
                    ])
                ])
            elif brand == "LEGO":
                recommendations_content = html.Div([
                    html.H6("Quick Tips", style={'color': 'white', 'marginBottom': '15px'}),
                    html.Ul([
                        html.Li("Increased customer concerns about product quality. Review quality control processes at manufacturing facilities.", 
                               style={'color': 'white', 'marginBottom': '10px'})
                    ])
                ])
            else:
                # Default recommendations for other brands
                recommendations_content = html.Div([
                    html.H6("Quick Tips", style={'color': 'white', 'marginBottom': '15px'}),
                    html.Ul([
                        html.Li("There is an increase in negative sentiment. Review the product and the customer experience.", 
                               style={'color': 'white', 'marginBottom': '10px'})
                    ])
                ])
            
            return recommendations_content, "", None, None, {'display': 'none'}, 'email-analysis-container'
            
        elif button_id == "analyze-button":
            # Brand-specific next steps
            if brand == "Barbie":
                analysis_content = [
                    html.Div([
                        html.H6("Next Steps:", style={'color': 'white', 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li("Prepare for upcoming Barbie Movie merchandise sale - coordinate with marketing team", 
                                   style={'color': 'white', 'marginBottom': '10px'}),
                            html.Li("Review inventory levels for Barbie Movie collection", 
                                   style={'color': 'white', 'marginBottom': '10px'}),
                            html.Li("Update promotional materials for the sale", 
                                   style={'color': 'white', 'marginBottom': '10px'})
                        ])
                    ])
                ]
                
                email_content = html.Div([
                    html.H6("Draft Email: Next Steps", style={'color': 'white', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Subject: Barbie Movie Merchandise Sale Planning", style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                        html.P("Dear Brand Manager,", style={'color': 'white', 'marginBottom': '10px'}),
                        html.P("We need to prepare for the upcoming Barbie Movie merchandise sale. Please coordinate with the marketing team to ensure all promotional materials are ready and inventory levels are sufficient. This is a key opportunity to boost sales and engage with our Barbie fan base.", style={'color': 'white', 'marginBottom': '10px'}),
                        html.P("Best regards,", style={'color': 'white', 'marginBottom': '5px'}),
                        html.P("AI Brand Analyst", style={'color': 'white'})
                    ], className="email-analysis-content")
                ])
                
            elif brand == "LEGO":
                analysis_content = [
                    html.Div([
                        html.H6("Next Steps:", style={'color': 'white', 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li("Send urgent communication to all packing and distribution centers", 
                                   style={'color': 'white', 'marginBottom': '10px'}),
                            html.Li("Review current packaging standards", 
                                   style={'color': 'white', 'marginBottom': '10px'}),
                            html.Li("Implement additional quality checks", 
                                   style={'color': 'white', 'marginBottom': '10px'})
                        ])
                    ])
                ]
                
                email_content = html.Div([
                    html.H6("Email Analysis", style={'color': 'white', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Subject: Urgent: Packaging Quality Concerns", style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                        html.P("Dear Brand Manager,", style={'color': 'white', 'marginBottom': '10px'}),
                        html.P("Recent customer feedback indicates an increase in damaged boxes and packaging issues. Please send an immediate communication to all packing and distribution centers to address these concerns. We need to review our current packaging standards and implement additional quality checks to maintain our brand's reputation for quality.", style={'color': 'white', 'marginBottom': '10px'}),
                        html.P("Best regards,", style={'color': 'white', 'marginBottom': '5px'}),
                        html.P("AI Brand Analyst", style={'color': 'white'})
                    ], className="email-analysis-content")
                ])
                
            else:
                # Default content for other brands
                analysis_content = [
                    html.Div([
                        html.H6("Next Steps:", style={'color': 'white', 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li("Monitor customer feedback trends", style={'color': 'white', 'marginBottom': '10px'}),
                            html.Li("Review current marketing strategies", style={'color': 'white', 'marginBottom': '10px'}),
                            html.Li("Analyze competitor activities", style={'color': 'white', 'marginBottom': '10px'})
                        ])
                    ])
                ]
                
                email_content = html.Div([
                    html.H6("Email Analysis", style={'color': 'white', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Subject: General Brand Update", style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                        html.P("Dear Brand Manager,", style={'color': 'white', 'marginBottom': '10px'}),
                        html.P("Please review the current brand performance metrics and consider implementing the suggested next steps to improve customer satisfaction and market position.", style={'color': 'white', 'marginBottom': '10px'}),
                        html.P("Best regards,", style={'color': 'white', 'marginBottom': '5px'}),
                        html.P("AI Brand Analyst", style={'color': 'white'})
                    ], className="email-analysis-content")
                ])
            
            return analysis_content, "", None, email_content, {
                'marginTop': '20px',
                'padding': '20px',
                'backgroundColor': '#636e72',
                'borderRadius': '5px',
                'display': 'block'
            }, 'email-analysis-container show'
            
    except Exception as e:
        return html.P(f"Error: {str(e)}", style={'color': 'white'}), "", None, None, {'display': 'none'}, 'email-analysis-container'

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
    try:
        # Filter data for the selected brand and category
        filtered_data = data[(data['category'] == category) & (data['brand'] == brand)]
        
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
            ], style=tooltip_text_style),
            html.Div([
                html.Span("This Month:", style={'color': '#b2bec3'}),
                html.Span(f"{len(filtered_data[filtered_data['date'] >= pd.Timestamp.now() - pd.DateOffset(months=1)]):,}", style={'color': '#3498db', 'fontWeight': 'bold'})
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
        error_msg = html.P(f"Error generating summary: {str(e)}", style={'color': 'white'})
        return error_msg, error_msg, error_msg, error_msg

# Add new callback for the brand review header
@app.callback(
    Output("brand-review-header", "children"),
    Input('brand-filter', 'value')
)
def update_brand_review_header(selected_brand):
    if selected_brand:
        return f"What people are saying about {selected_brand} products..."
    return "What people are saying about selected brand..."

# Add callback for the category review header
@app.callback(
    Output("category-review-header", "children"),
    Input('category-filter', 'value')
)
def update_category_review_header(selected_category):
    if selected_category:
        return f"What people are saying about other products in the {selected_category} category..."
    return "What people are saying about other products in the selected category..."

if __name__ == "__main__":
    # Check if running in Databricks
    if 'dbutils' in globals():
        # For Databricks environment
        app.run(host='0.0.0.0', port=8050, debug=True, use_reloader=False)
    else:
        # For local environment
        app.run(debug=True)