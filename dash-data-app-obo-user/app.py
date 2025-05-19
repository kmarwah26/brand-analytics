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
# from config import DATABRICKS_CONFIG, validate_config
# from data_query import sql_query_with_service_principal, sql_query_with_user_token

# Validate configuration
# validate_config()

def generate_sample_data() -> pd.DataFrame:
    """Generate comprehensive sample brand review data."""
    np.random.seed(42)
    n_samples = 2000  # Increased sample size
    
    # Generate dates for the past year
    dates = pd.date_range(end=pd.Timestamp.now(), periods=365, freq='D')
    
    # Generate brands with different characteristics
    brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D']
    brand_weights = [0.4, 0.3, 0.2, 0.1]  # Different weights for each brand
    
    # Generate random data with more realistic patterns
    data = pd.DataFrame({
        'date': np.random.choice(dates, n_samples),
        'brand': np.random.choice(brands, n_samples, p=brand_weights),
        'source': np.random.choice(['Website', 'Social Media', 'App Store', 'Email'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'rating': np.random.randint(1, 6, n_samples),
        'review_text': [
            f"Sample review text {i} about the brand experience and product quality."
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
    
    # Add brand-specific patterns
    brand_patterns = {
        'Brand A': {'rating_boost': 1.2, 'sentiment_boost': 0.9},
        'Brand B': {'rating_boost': 1.1, 'sentiment_boost': 0.8},
        'Brand C': {'rating_boost': 0.9, 'sentiment_boost': 0.7},
        'Brand D': {'rating_boost': 0.8, 'sentiment_boost': 0.6}
    }
    
    for brand, pattern in brand_patterns.items():
        mask = data['brand'] == brand
        data.loc[mask, 'rating'] = data.loc[mask, 'rating'] * pattern['rating_boost']
        data.loc[mask, 'market_share'] = data.loc[mask, 'market_share'] * pattern['sentiment_boost']
    
    # Ensure ratings stay within 1-5 range
    data['rating'] = data['rating'].clip(1, 5)
    
    # Add review categories
    categories = ['Product', 'Service', 'Price', 'Quality', 'Experience']
    data['category'] = np.random.choice(categories, n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    
    # Add review length
    data['review_length'] = np.random.randint(50, 500, n_samples)
    
    # Add response time (in hours)
    data['response_time'] = np.random.exponential(24, n_samples)  # Mean response time of 24 hours
    
    # Add resolution time (in hours)
    data['resolution_time'] = data['response_time'] + np.random.exponential(48, n_samples)
    
    return data

def load_data() -> pd.DataFrame:
    """Load sample data."""
    try:
        return generate_sample_data()
    except Exception as e:
        print(f"Data generation failed: {str(e)}")
        return pd.DataFrame()

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

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
                            dbc.Label("Brand", className="text-light"),
                            dcc.Dropdown(
                                id='brand-filter',
                                options=[
                                    {'label': 'All Brands', 'value': 'all'},
                                    {'label': 'Brand A', 'value': 'Brand A'},
                                    {'label': 'Brand B', 'value': 'Brand B'},
                                    {'label': 'Brand C', 'value': 'Brand C'},
                                    {'label': 'Brand D', 'value': 'Brand D'}
                                ],
                                value='all',
                                className="mb-3"
                            )
                        ], width=6),
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
                                    {"headerName": "Date", "field": "date", "sortable": True, "width": 120},
                                    {"headerName": "Sentiment", "field": "sentiment", "sortable": True, "width": 120},
                                    {"headerName": "Rating", "field": "rating", "sortable": True, "width": 100},
                                    {"headerName": "Category", "field": "category", "sortable": True, "width": 120},
                                    {"headerName": "Source", "field": "source", "sortable": True, "width": 120},
                                    {
                                        "headerName": "Review",
                                        "field": "review_text",
                                        "sortable": True,
                                        "flex": 1,
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
                                    "minWidth": 100
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
                                        id='brand1-filter',
                                        options=[
                                            {'label': 'Brand A', 'value': 'Brand A'},
                                            {'label': 'Brand B', 'value': 'Brand B'},
                                            {'label': 'Brand C', 'value': 'Brand C'},
                                            {'label': 'Brand D', 'value': 'Brand D'}
                                        ],
                                        value='Brand A',
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label(" ", className="text-light"),  # Empty label for alignment
                                    dcc.Dropdown(
                                        id='brand2-filter',
                                        options=[
                                            {'label': 'Brand A', 'value': 'Brand A'},
                                            {'label': 'Brand B', 'value': 'Brand B'},
                                            {'label': 'Brand C', 'value': 'Brand C'},
                                            {'label': 'Brand D', 'value': 'Brand D'}
                                        ],
                                        value='Brand B',
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
                            dcc.Graph(id='competitive-radar', style={'height': '400px'})
                        ])
                    ], style=CARD_STYLE)
                ], width=12)
            ], className="mb-4")
        ], label="Competitive Positioning", tab_id="tab-competitive")
    ], id="tabs", active_tab="tab-brand-health", className="mb-4"),
    
], fluid=True, style={'backgroundColor': '#2d3436', 'minHeight': '100vh', 'padding': '20px'})  # Lighter background

# Update the callback to remove date range inputs
@app.callback(
    Output('sentiment-donut', 'figure'),
    Output('brand-comparison-chart', 'figure'),
    Output('reviews-table', 'rowData'),
    Output('metrics-container', 'children'),
    Output('brand-health-score', 'children'),
    Output('competitive-score', 'children'),
    Output('product-score', 'children'),
    Output('market-share-trend', 'figure'),
    Output('competitive-radar', 'figure'),
    Output('attribute-analysis', 'figure'),
    Output('sentiment-value', 'children'),
    Output('total-reviews-counter', 'children'),
    Output('positive-reviews-counter', 'children'),
    Input('page-load-trigger', 'data'),
    Input('brand-filter', 'value'),
    Input('sentiment-filter', 'value'),
    Input('brand1-filter', 'value'),
    Input('brand2-filter', 'value')
)
def update_visuals(n_clicks, brand, sentiment_threshold, brand1, brand2):
    data = load_data()
    
    # Apply filters
    if brand != 'all':
        data = data[data['brand'] == brand]
    
    # Filter by sentiment threshold
    sentiment_scores = data['rating'] * 20  # Convert 1-5 rating to 0-100 scale
    data = data[sentiment_scores >= sentiment_threshold]
    
    # Create donut chart for sentiment distribution
    sentiment_counts = data['sentiment'].value_counts()
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
    
    # Create brand comparison chart
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='M')
    
    # Generate monthly review counts for selected brands
    brand1_data = data[data['brand'] == brand1].groupby(data['date'].dt.to_period('M')).size()
    brand2_data = data[data['brand'] == brand2].groupby(data['date'].dt.to_period('M')).size()
    
    # Create the comparison chart
    comparison_fig = go.Figure()
    
    # Add bars for brand 1
    comparison_fig.add_trace(go.Bar(
        x=[str(d) for d in date_range],
        y=[brand1_data.get(pd.Period(d, freq='M'), 0) for d in date_range],
        name=brand1,
        marker_color='#3498db'
    ))
    
    # Add bars for brand 2
    comparison_fig.add_trace(go.Bar(
        x=[str(d) for d in date_range],
        y=[brand2_data.get(pd.Period(d, freq='M'), 0) for d in date_range],
        name=brand2,
        marker_color='#e74c3c'
    ))
    
    comparison_fig.update_layout(
        title='Monthly Review Comparison',
        barmode='group',
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
        )
    )
    
    # Calculate key metrics
    total_reviews = len(data)
    positive_reviews = len(data[data['sentiment'] == 'Positive'])
    positive_pct = (data['sentiment'] == 'Positive').mean() * 100
    avg_rating = data['rating'].mean()
    response_rate = np.random.uniform(85, 95)  # Simulated response rate
    resolution_time = np.random.uniform(2, 4)  # Simulated average resolution time in hours
    
    # Get last 5 reviews for the table
    recent_reviews = data.sort_values('date', ascending=False).head(5)
    
    metrics = [
        html.H4(f"Total Reviews: {total_reviews:,}", style={'color': 'white'}),
        html.H4(f"Positive Sentiment: {positive_pct:.1f}%", style={'color': 'white'}),
        html.H4(f"Average Rating: {avg_rating:.1f}/5.0", style={'color': 'white'}),
        html.H4(f"Response Rate: {response_rate:.1f}%", style={'color': 'white'}),
        html.H4(f"Avg. Resolution Time: {resolution_time:.1f} hours", style={'color': 'white'})
    ]
    
    # Calculate brand health scores
    brand_health = (data['sentiment'] == 'Positive').mean() * 100
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
    
    # Create competitive radar chart
    categories = ['Price', 'Quality', 'Service', 'Innovation', 'Brand Value']
    our_scores = np.random.uniform(7, 9, len(categories))
    competitor_scores = np.random.uniform(6, 8, len(categories))
    
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=our_scores,
        theta=categories,
        fill='toself',
        name='Our Brand'
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=competitor_scores,
        theta=categories,
        fill='toself',
        name='Competitor'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
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
        radar_fig,
        attribute_fig,
        sentiment_value,
        f"{total_reviews:,}",
        f"{positive_reviews:,}"
    )

if __name__ == "__main__":
    app.run(debug=True)