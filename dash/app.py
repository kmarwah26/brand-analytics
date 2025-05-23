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

# Databricks config
cfg = Config()

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

        
def sql_query_with_user_token(query: str, user_token: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
        access_token=user_token  # Pass the user token into the SQL connect to query on behalf of user
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
        

def load_data(local: bool, query: str) -> pd.DataFrame:
    """Load data from Databricks SQL warehouse using either user or app authentication."""
    try:
        if local:
            # Extract user access token from the request headers
            user_token = flask.request.headers.get('X-Forwarded-Access-Token')
            if not user_token:
                raise Exception("Missing access token in headers.")
            # Query the SQL data with the user credentials
            return sql_query_with_user_token(query, user_token=user_token)
        else:
            return sqlQuery(query)
    except Exception as e:
        print(f"Data load failed: {str(e)}")
        return pd.DataFrame()


# Fetch the data. When debugging locally, set local = True to authenticate with SP creds. For Databricks app deployment, set local = False
queryText = "SELECT * FROM retail_cpg_demo.brand_manager.vw_brand_insights"
data = load_data(local=False, query=queryText)

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
    Output('reviews-table', 'rowData'),
    Output('metrics-container', 'children'),
    Output('brand-health-score', 'children'),
    Output('competitive-score', 'children'),
    Output('product-score', 'children'),
    Output('market-share-trend', 'figure'),
    Output('pricing-comparison', 'figure'),
    Output('positive-wordcloud', 'src'),
    Output('negative-wordcloud', 'src'),
    Output('attribute-analysis', 'figure'),
    Output('sentiment-value', 'children'),
    Output('total-reviews-counter', 'children'),
    Output('positive-reviews-counter', 'children'),
    Input('page-load-trigger', 'data'),
    Input('category-filter', 'value'),
    Input('brand-filter', 'value'),
    Input('sentiment-filter', 'value')
)
def update_visuals(n_clicks, category, brand, sentiment_threshold):
    #data = load_dummy_data()
    
    # Apply filters
    category_data = data[data['category'] == category]
    filtered_data = category_data[category_data['brand'] == brand]
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
    
    # Create market share trend with monthly review counts
    # Group by month and count reviews
    monthly_counts = filtered_data.groupby(filtered_data['date'].dt.to_period('M')).size().reset_index(name='review_count')
    monthly_counts['date'] = monthly_counts['date'].dt.to_timestamp()

    # Create the line plot
    market_share_fig = px.line(
        monthly_counts,
        x='date',
        y='review_count',
        title='Market Share'
    )
    market_share_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#404040'),
        yaxis=dict(gridcolor='#404040')
    )

    # Create price analysis chart
    price_fig = go.Figure()

    brands = category_data['brand'].unique()
    # Assign colors: orange for selected, blue for others
    colors = ['#FFA500' if b == brand else '#3498db' for b in brands]
    
    # Add bar chart for brand prices
    price_fig.add_trace(go.Bar(
        x=brands,
        y=category_data.groupby('brand')['avg_brand_price'].mean(),
        name='Brand Average Price',
        marker_color=colors
    ))
    
    # Add line for category average
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

    # Calculate key metrics
    total_reviews = len(filtered_data)
    positive_reviews = len(filtered_data[filtered_data['sentiment'] == 'Positive'])
    positive_pct = (filtered_data['sentiment'] == 'Positive').mean() * 100
    avg_rating = filtered_data['rating'].mean()
    avg_sentiment_score = filtered_data['sentiment_score'].mean()
    
    # Get last 5 reviews for the table
    recent_reviews = filtered_data.sort_values('date', ascending=False).head(8)
    
    metrics = [
        html.H4(f"Total Reviews: {total_reviews:,}", style={'color': 'white'}),
        html.H4(f"Positive Sentiment: {positive_pct:.1f}%", style={'color': 'white'}),
        html.H4(f"Average Rating: {avg_rating:.1f}/5.0", style={'color': 'white'}),
        html.H4(f"Average Sentiment Score: {avg_sentiment_score:.1f}/5.0", style={'color': 'white'})
    ]
    
    # Calculate brand health scores
    brand_health = (filtered_data['sentiment'] == 'Positive').mean() * 100
    competitive_position = np.random.uniform(75, 95)  # Simulated competitive position
    product_score = np.random.uniform(80, 98)  # Simulated product score
    
    # Format the scores
    brand_health_score = f"{brand_health:.1f}%"
    competitive_score = f"{competitive_position:.1f}%"
    product_score = f"{product_score:.1f}%"
    
    
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
        recent_reviews.to_dict('records'),
        metrics,
        brand_health_score,
        competitive_score,
        product_score,
        market_share_fig,
        price_fig,
        positive_wordcloud,
        negative_wordcloud,
        attribute_fig,
        sentiment_value,
        f"{total_reviews:,}",
        f"{positive_reviews:,}"
    )

if __name__ == "__main__":
    app.run(debug=True)