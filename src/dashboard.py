import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>E-Commerce Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                background-color: #f5f5f5;
            }
            .app-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .metric-card {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 30%;
                text-align: center;
            }
            .chart-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 30px;
            }
            h2 {
                color: #3498db;
                margin: 5px 0;
            }
            h4 {
                color: #7f8c8d;
                margin: 5px 0;
            }
            .filters-container {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
            }
            .recommendation-container {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .recommendation-item {
                padding: 10px;
                border-bottom: 1px solid #eee;
            }
            .recommendation-item:last-child {
                border-bottom: none;
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

# Load the data
df = pd.read_csv('../data/ecommerce_data.csv')

# Convert purchase_date to datetime with explicit format to avoid warnings
df['purchase_date'] = pd.to_datetime(df['purchase_date'], format='%Y-%m-%d')

# Create the recommendation system
def create_recommendation_system(df):
    # Combine product features
    df['product_features'] = df['product_name'] + ' ' + df['category'] + ' ' + df['review']
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['product_features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create mapping of product IDs to indices
    product_indices = pd.Series(df.index, index=df['product_id']).drop_duplicates()
    
    return cosine_sim, product_indices

# Get product recommendations
def get_recommendations(product_id, cosine_sim, product_indices, df):
    # Get the index of the product
    idx = product_indices[product_id]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort products based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar products (excluding the product itself)
    sim_scores = sim_scores[1:6]
    
    # Get product indices
    product_indices_list = [i[0] for i in sim_scores]
    
    # Return recommended products
    return df.iloc[product_indices_list][['product_id', 'product_name', 'category', 'price', 'rating']].drop_duplicates(subset=['product_id'])

# Create recommendation system
cosine_sim, product_indices = create_recommendation_system(df)

# App layout
app.layout = html.Div(className='app-container', children=[
    html.H1('E-Commerce Analytics Dashboard', style={'textAlign': 'center'}),
    
    # Filters
    html.Div([
        html.Div([
            html.H4('Date Range'),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['purchase_date'].min(),
                end_date=df['purchase_date'].max()
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.H4('Category'),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': x, 'value': x} for x in df['category'].unique()],
                value=None,
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], className='filters-container'),
    
    # Metrics Row
    html.Div([
        html.Div([
            html.H4('Total Sales'),
            html.H2(id='total-sales')
        ], className='metric-card'),
        html.Div([
            html.H4('Average Order Value'),
            html.H2(id='avg-order-value')
        ], className='metric-card'),
        html.Div([
            html.H4('Total Customers'),
            html.H2(id='total-customers')
        ], className='metric-card')
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}),
    
    # Charts Row 1
    html.Div([
        html.Div([
            dcc.Graph(id='sales-trend')
        ], className='chart-container', style={'width': '48%'}),
        html.Div([
            dcc.Graph(id='category-distribution')
        ], className='chart-container', style={'width': '48%'})
    ], style={'display': 'flex'}),
    
    # Charts Row 2
    html.Div([
        html.Div([
            dcc.Graph(id='customer-behavior')
        ], className='chart-container', style={'width': '48%'}),
        html.Div([
            dcc.Graph(id='product-performance')
        ], className='chart-container', style={'width': '48%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Product Recommendation Section
    html.Div([
        html.H3('Product Recommendation System'),
        html.Div([
            html.Div([
                html.Label('Select a Product:'),
                dcc.Dropdown(
                    id='product-dropdown',
                    options=[{'label': f"{row['product_name']} ({row['product_id']})", 'value': row['product_id']} 
                             for _, row in df.drop_duplicates('product_id').iterrows()],
                    value=df['product_id'].iloc[0]
                )
            ], style={'width': '50%', 'marginBottom': '20px'})
        ]),
        html.Div(id='recommendations-output', className='recommendation-container')
    ])
])

# Callback for updating metrics
@app.callback(
    [Output('total-sales', 'children'),
     Output('avg-order-value', 'children'),
     Output('total-customers', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('category-filter', 'value')]
)
def update_metrics(start_date, end_date, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['purchase_date'] >= start_date) &
            (filtered_df['purchase_date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    total_sales = f"${filtered_df['price'].sum():,.2f}"
    avg_order = f"${filtered_df['price'].mean():,.2f}"
    total_customers = len(filtered_df['user_id'].unique())
    
    return total_sales, avg_order, total_customers

# Callback for sales trend
@app.callback(
    Output('sales-trend', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('category-filter', 'value')]
)
def update_sales_trend(start_date, end_date, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['purchase_date'] >= start_date) &
            (filtered_df['purchase_date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    daily_sales = filtered_df.groupby(filtered_df['purchase_date'].dt.date)['price'].sum().reset_index()
    
    fig = px.line(daily_sales, x='purchase_date', y='price',
                  title='Daily Sales Trend')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Total Sales ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.8)')
    )
    return fig

# Callback for category distribution
@app.callback(
    Output('category-distribution', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_category_distribution(start_date, end_date):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['purchase_date'] >= start_date) &
            (filtered_df['purchase_date'] <= end_date)
        ]
    
    category_sales = filtered_df.groupby('category')['price'].sum().reset_index()
    
    fig = px.pie(category_sales, values='price', names='category',
                 title='Sales Distribution by Category',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Callback for customer behavior
@app.callback(
    Output('customer-behavior', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('category-filter', 'value')]
)
def update_customer_behavior(start_date, end_date, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['purchase_date'] >= start_date) &
            (filtered_df['purchase_date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    # Analyze purchase frequency by user
    user_purchase_counts = filtered_df.groupby('user_id').size().reset_index(name='purchase_count')
    user_purchase_counts = user_purchase_counts.sort_values('purchase_count', ascending=False).head(10)
    
    fig = px.bar(user_purchase_counts, x='user_id', y='purchase_count',
                 title='Top 10 Users by Purchase Frequency',
                 labels={'user_id': 'User ID', 'purchase_count': 'Number of Purchases'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.8)')
    )
    return fig

# Callback for product performance
@app.callback(
    Output('product-performance', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('category-filter', 'value')]
)
def update_product_performance(start_date, end_date, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['purchase_date'] >= start_date) &
            (filtered_df['purchase_date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    # Analyze product performance by sales and rating
    product_performance = filtered_df.groupby('product_name').agg(
        total_sales=('price', 'sum'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # Get top 10 products by sales
    top_products = product_performance.sort_values('total_sales', ascending=False).head(10)
    
    fig = px.scatter(top_products, x='total_sales', y='avg_rating', 
                     size='total_sales', color='avg_rating',
                     hover_name='product_name', 
                     title='Top 10 Products: Sales vs Rating',
                     color_continuous_scale='Viridis',
                     labels={'total_sales': 'Total Sales ($)', 'avg_rating': 'Average Rating'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.8)')
    )
    return fig

# Callback for product recommendations
@app.callback(
    Output('recommendations-output', 'children'),
    [Input('product-dropdown', 'value')]
)
def update_recommendations(product_id):
    if not product_id:
        return html.Div("Please select a product to get recommendations.")
    
    # Get recommendations for the selected product
    recommendations = get_recommendations(product_id, cosine_sim, product_indices, df)
    
    # Display selected product info
    selected_product = df[df['product_id'] == product_id].iloc[0]
    selected_product_info = html.Div([
        html.H4(f"Selected Product: {selected_product['product_name']}"),
        html.P(f"Category: {selected_product['category']} | Price: ${selected_product['price']:.2f} | Rating: {selected_product['rating']}")
    ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f0f8ff', 'borderRadius': '5px'})
    
    # Display recommendations
    recommendation_items = []
    for i, (_, row) in enumerate(recommendations.iterrows()):
        recommendation_items.append(html.Div([
            html.H5(f"{i+1}. {row['product_name']}"),
            html.P(f"Category: {row['category']} | Price: ${row['price']:.2f} | Rating: {row['rating']}")
        ], className='recommendation-item'))
    
    return html.Div([
        selected_product_info,
        html.H4("Recommended Products:"),
        html.Div(recommendation_items)
    ])

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
