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
    """Generate sample brand review data."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate dates for the past year
    dates = pd.date_range(end=pd.Timestamp.now(), periods=365, freq='D')
    
    # Generate random data
    data = pd.DataFrame({
        'date': np.random.choice(dates, n_samples),
        'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], n_samples, p=[0.6, 0.3, 0.1]),
        'rating': np.random.randint(1, 6, n_samples),
        'review_text': [
            f"Sample review text {i} about the brand experience and product quality."
            for i in range(n_samples)
        ],
        'brand': np.random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D'], n_samples),
        'source': np.random.choice(['Website', 'Social Media', 'App Store', 'Email'], n_samples)
    })
    
    # Add some correlation between sentiment and rating
    data.loc[data['sentiment'] == 'Positive', 'rating'] = np.random.randint(4, 6, size=len(data[data['sentiment'] == 'Positive']))
    data.loc[data['sentiment'] == 'Negative', 'rating'] = np.random.randint(1, 3, size=len(data[data['sentiment'] == 'Negative']))
    data.loc[data['sentiment'] == 'Neutral', 'rating'] = np.random.randint(3, 5, size=len(data[data['sentiment'] == 'Neutral']))
    
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
    'backgroundColor': '#2b2b2b',
    'border': '1px solid #404040',
    'borderRadius': '5px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
}

CARD_HEADER_STYLE = {
    'backgroundColor': '#404040',
    'color': 'white',
    'borderBottom': '1px solid #505050'
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
    dbc.Row([dbc.Col(html.H1("Brand Insights Dashboard", className="text-center my-4 text-light"), width=12)]),
    
    # Filter Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters", style=CARD_HEADER_STYLE),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Date Range", className="text-light"),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date=pd.Timestamp.now() - pd.Timedelta(days=30),
                                end_date=pd.Timestamp.now(),
                                className="mb-3"
                            )
                        ], width=4),
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
                    ], style={**CARD_STYLE, 'backgroundColor': '#1e3a2e'})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Competitive Positioning", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='competitive-score', className="text-center mb-2", style={'color': '#3498db', 'fontSize': '2.5rem'}),
                            html.P("Market Share & Position", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#1e2e3a'})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Product Attribute Analysis", className="text-center mb-3", style={'color': 'white'}),
                            html.H2(id='product-score', className="text-center mb-2", style={'color': '#9b59b6', 'fontSize': '2.5rem'}),
                            html.P("Product Performance Score", className="text-center text-light")
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#2e1e3a'})
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
                            html.Div(id='metrics-container', className="d-flex flex-column gap-3 text-light")
                        ])
                    ], style=CARD_STYLE)
                ], width=6)
            ], className="mb-4"),
            
            # Reviews Table Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Reviews", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dag.AgGrid(
                                id='reviews-table',
                                columnDefs=[
                                    {"headerName": "Date", "field": "date", "sortable": True},
                                    {"headerName": "Sentiment", "field": "sentiment", "sortable": True},
                                    {"headerName": "Rating", "field": "rating", "sortable": True},
                                    {"headerName": "Category", "field": "category", "sortable": True},
                                    {"headerName": "Source", "field": "source", "sortable": True},
                                    {"headerName": "Review", "field": "review_text", "sortable": True}
                                ],
                                rowData=[],
                                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                                dashGridOptions={
                                    "rowSelection": "single",
                                    "theme": "ag-theme-alpine-dark"
                                },
                                style={"height": "400px"}
                            )
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
                        dbc.CardHeader("Product Performance Metrics", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dcc.Graph(id='product-metrics', style={'height': '400px'})
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
    
], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})

# Update the callback to include brand comparison
@app.callback(
    Output('sentiment-donut', 'figure'),
    Output('sentiment-trend', 'figure'),
    Output('reviews-table', 'rowData'),
    Output('metrics-container', 'children'),
    Output('brand-health-score', 'children'),
    Output('competitive-score', 'children'),
    Output('product-score', 'children'),
    Output('market-share-trend', 'figure'),
    Output('competitive-radar', 'figure'),
    Output('product-metrics', 'figure'),
    Output('attribute-analysis', 'figure'),
    Output('sentiment-value', 'children'),
    Output('brand-comparison-chart', 'figure'),
    Input('page-load-trigger', 'data'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('brand-filter', 'value'),
    Input('sentiment-filter', 'value'),
    Input('brand1-filter', 'value'),
    Input('brand2-filter', 'value')
)
def update_visuals(n_clicks, start_date, end_date, brand, sentiment_threshold, brand1, brand2):
    data = load_data()
    
    # Apply filters
    if start_date and end_date:
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
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
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c', '#95a5a6']
    )])
    donut_fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(
            text="Sentiment Distribution",
            font=dict(size=20, color='white'),
            x=0.5,
            y=0.95
        )
    )
    
    # Create sentiment trend chart
    daily_sentiment = data.groupby(['date', 'sentiment']).size().reset_index(name='count')
    trend_fig = px.bar(
        daily_sentiment,
        x='date',
        y='count',
        color='sentiment',
        barmode='group',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6'
        }
    )
    trend_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reviews",
        legend_title="Sentiment",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#404040'),
        yaxis=dict(gridcolor='#404040'),
        title=dict(
            text="Sentiment Trends Over Time",
            font=dict(size=20, color='white'),
            x=0.5,
            y=0.95
        )
    )
    
    # Calculate key metrics
    total_reviews = len(data)
    positive_pct = (data['sentiment'] == 'Positive').mean() * 100
    avg_rating = data['rating'].mean()
    response_rate = np.random.uniform(85, 95)  # Simulated response rate
    resolution_time = np.random.uniform(2, 4)  # Simulated average resolution time in hours
    
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
    
    # Create product metrics
    metrics = ['Performance', 'Reliability', 'Design', 'Features', 'Value']
    scores = np.random.uniform(8, 9.5, len(metrics))
    product_metrics_fig = px.bar(
        x=metrics,
        y=scores,
        title='Product Performance Metrics'
    )
    product_metrics_fig.update_layout(
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
    
    # Create brand comparison chart
    if start_date and end_date:
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    else:
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
    
    return (
        donut_fig,
        trend_fig,
        data.to_dict('records'),
        metrics,
        brand_health_score,
        competitive_score,
        product_score,
        market_share_fig,
        radar_fig,
        product_metrics_fig,
        attribute_fig,
        sentiment_value,
        comparison_fig
    )

if __name__ == "__main__":
    app.run(debug=True)