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

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

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

# Load data from CSV
try:
    data = pd.read_csv('app_data/brand_insights_data.csv')
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    
    # Convert the date column to a datetime object
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Ensure required columns exist
    required_columns = ['date', 'category', 'brand', 'rating', 'review_text', 'sentiment', 'sentiment_score', 'avg_brand_price']
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
                        dbc.CardHeader("Brand Analysis", style=CARD_HEADER_STYLE),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Positive Reviews", className="text-center mb-3"),
                                    html.Img(id='brand-positive-wordcloud', style={'width': '100%', 'height': 'auto'})
                                ], width=6),
                                dbc.Col([
                                    html.H4("Negative Reviews", className="text-center mb-3"),
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
                        dbc.CardHeader("Attributes", style=CARD_HEADER_STYLE),
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


# Update the callback with better error handling and data validation
@app.callback(
    Output('sentiment-treemap', 'figure'),
    Output('reviews-table', 'rowData'),
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
            [],  # reviews-table
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
            "0"   # positive-reviews-counter
        )

    try:
        # Apply filters
        category_data = data[data['category'] == category]
        filtered_data = category_data[category_data['brand'] == brand]
        
        if len(filtered_data) == 0:
            empty_fig = create_empty_fig()
            return (
                empty_fig,  # sentiment-treemap
                [],  # reviews-table
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
                "0"   # positive-reviews-counter
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
                colors=['#2ecc71' if s == 'Positive' else '#e74c3c' if s == 'Negative' else '#95a5a6' 
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
        def generate_wordcloud(texts, max_words=100):
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
            except Exception as e:
                print(f"Error generating wordcloud: {str(e)}")
                return None
        
        # Get reviews from all other brands in the selected category
        category_data = data[data['category'] == category]  # Filter by selected category
        other_brands_data = category_data[category_data['brand'] != brand]  # Filter out selected brand
        
        # Generate word clouds for competitor analysis
        positive_reviews = other_brands_data[other_brands_data['sentiment_score'] >= 3]['review_text'].tolist()
        negative_reviews = other_brands_data[other_brands_data['sentiment_score'] < 3]['review_text'].tolist()
        
        positive_wordcloud = generate_wordcloud(positive_reviews)
        negative_wordcloud = generate_wordcloud(negative_reviews)
        
        # Generate word clouds for brand analysis
        brand_positive_reviews = filtered_data[filtered_data['sentiment_score'] >= 3]['review_text'].tolist()
        brand_negative_reviews = filtered_data[filtered_data['sentiment_score'] < 3]['review_text'].tolist()
        
        brand_positive_wordcloud = generate_wordcloud(brand_positive_reviews)
        brand_negative_wordcloud = generate_wordcloud(brand_negative_reviews)
        
        # Create market share trend
        # Create Share of Voice chart
        # Group by month and brand to get review counts
        monthly_brand_counts = category_data.groupby([
            category_data['date'].dt.to_period('M'),
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

        # Calculate metrics
        total_reviews = len(filtered_data)
        positive_reviews_count = len(filtered_data[filtered_data['sentiment_score'] >= 3])
        positive_pct = (filtered_data['sentiment_score'] >= 3).mean() * 100
        avg_rating = filtered_data['rating'].mean()
        avg_sentiment_score = filtered_data['sentiment_score'].mean()
        
        # Get recent reviews
        recent_reviews = filtered_data.sort_values('date', ascending=False).head(8)
        
        metrics = [
            html.H4(f"Total Reviews: {total_reviews:,}", style={'color': 'white'}),
            html.H4(f"Positive Sentiment: {positive_pct:.1f}%", style={'color': 'white'}),
            html.H4(f"Average Rating: {avg_rating:.1f}/5.0", style={'color': 'white'}),
            html.H4(f"Average Sentiment Score: {avg_sentiment_score:.1f}/5.0", style={'color': 'white'})
        ]
        
        # Calculate scores
        brand_health = (filtered_data['sentiment_score'] >= 3).mean() * 100
        competitive_position = np.random.uniform(75, 95)
        product_score = np.random.uniform(80, 98)
        
        return (
            sentiment_treemap,
            recent_reviews.to_dict('records'),
            metrics,
            f"{brand_health:.1f}%",
            f"{competitive_position:.1f}%",
            f"{product_score:.1f}%",
            market_share_fig,
            price_fig,
            positive_wordcloud,
            negative_wordcloud,
            brand_positive_wordcloud,
            brand_negative_wordcloud,
            f"{total_reviews:,}",
            f"{positive_reviews_count:,}"
        )
        
    except Exception as e:
        print(f"Error in callback: {str(e)}")
        empty_fig = create_empty_fig()
        return (
            empty_fig,  # sentiment-treemap
            [],  # reviews-table
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
            "0"   # positive-reviews-counter
        )

if __name__ == "__main__":
    app.run(debug=True)