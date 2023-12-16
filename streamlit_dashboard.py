import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import datetime
import seaborn as sns
import numpy as np
from PIL import Image



df = pd.read_csv("styles.csv")
df.dropna(inplace=True)
print(df.head())
apparel_df = df[df['masterCategory'] == 'Apparel']
footwear_df = df[df['masterCategory'] == 'Footwear']  # Assuming there's a 'Footwear' category in your dataset
accessories_df = df[df['masterCategory'] == 'Accessories']

# Group by 'year' and count occurrences for each category
apparel_count = apparel_df.groupby('year').size().reset_index(name='Apparel Count')
footwear_count = footwear_df.groupby('year').size().reset_index(name='Footwear Count')
accessories_count = accessories_df.groupby('year').size().reset_index(name='Accessories Count')

# Merge counts into a single DataFrame based on 'year'
result_df = pd.merge(apparel_count, footwear_count, on='year', how='outer')
result_df = pd.merge(result_df, accessories_count, on='year', how='outer').fillna(0)

result_df['Apparel Count'] = result_df['Apparel Count'].astype(int)
result_df['Footwear Count'] = result_df['Footwear Count'].astype(int)
result_df['Accessories Count'] = result_df['Accessories Count'].astype(int)

result_df = result_df[result_df['year'] != 2007]
result_df['Apparel Growth Rate %'] = result_df['Apparel Count'].pct_change() * 100
result_df['Accessories Growth Rate %'] = result_df['Accessories Count'].pct_change() * 100
result_df['Footwear Growth Rate %'] = result_df['Footwear Count'].pct_change() * 100

result_df_melted = result_df.melt(id_vars='year', value_vars=['Apparel Growth Rate %', 'Accessories Growth Rate %', 'Footwear Growth Rate %'], var_name='Category', value_name='Growth Rate')
# Display the resulting DataFrame
print(result_df_melted)
category_counts = df['masterCategory'].value_counts()
category_counts_df = pd.DataFrame({'Master Category': category_counts.index, 'Total Count': category_counts.values})

st.set_page_config(page_title="Fashion Trend Analysis",page_icon=":bar_chart:",layout="wide")
st.title(":bar_chart: Fashion Trend Analysis")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown(
    """
    <style>
    body {
        background-color: #222; /* Change this hex code to the desired dark color */
        color: white; /* Text color for better visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("Filter By:")
category = st.sidebar.multiselect("Filter By Category:",
                                  options=df["masterCategory"].unique(),
                                  default=df["masterCategory"].unique())
gender = st.sidebar.multiselect("Filter By Gender:",
                                options=df["gender"].unique(),
                                default=df["gender"].unique())
season = st.sidebar.multiselect("Filter By Season:",
                                options=df["season"].unique(),
                                default=df["season"].unique())
usage = st.sidebar.multiselect("Filter By Usage:",
                               options=df["usage"].unique(),
                               default=df["usage"].unique())

filtered_data = df[
    df["masterCategory"].isin(category) & df["gender"].isin(gender) & df["season"].isin(season) & df["usage"].isin(
        usage)]

# Display filtered data
st.write(filtered_data)

col1, col2= st.columns(2)


with col1:
    # Calculate counts based on filtered data
    filtered_category_counts = filtered_data.groupby('masterCategory').size().reset_index(name='Total Count')
    fig = px.bar(filtered_category_counts, x='masterCategory', y='Total Count',
                 labels={'masterCategory': 'Master Category', 'Total Count': 'Total Count'},
                 title="Total Count of Items in Each Master Category",
                 hover_data={'Total Count': True})
    st.plotly_chart(fig, use_container_width=True)

_, view1, dwn1, view2, dwn2 = st.columns([0.15, 0.20, 0.20, 0.20, 0.20])

with view1:
    expander = st.expander("Master Category Wise Sales")
    expander.write(filtered_category_counts)

with dwn1:
    st.download_button("Get Data", data=filtered_category_counts.to_csv().encode("utf-8"),
                       file_name="MasterCategoryElementsCount.csv", mime="text/csv")

with col2:
    # Pie chart for Season distribution
    st.subheader('Season Distribution:')
    season_counts = filtered_data['season'].value_counts()
    fig_season = px.pie(season_counts, values=season_counts.values, names=season_counts.index,
                        title='Season Distribution')
    st.plotly_chart(fig_season)

with view2:
    expander = st.expander("Season Distribution")
    expander.write(season_counts)

with dwn2:
    st.download_button("Get Data", data=season_counts.to_csv().encode("utf-8"),
                       file_name="SeasonCounts.csv", mime="text/csv")

st.divider()


subcategory_sales = filtered_data.groupby('subCategory').size().reset_index(name='Total Sales')

# Bar chart for total sales by subcategory
fig_bar = px.bar(subcategory_sales, x='subCategory', y='Total Sales',
                 labels={'subCategory': 'Subcategory', 'Total Sales': 'Total Sales'},
                 title='Total Sales by Subcategory')

# Line chart overlayed on the bar chart
fig_line = px.line(subcategory_sales, x='subCategory', y='Total Sales', labels={'Total Sales': 'Total Sales'},
                   title='Total Sales by Subcategory (Line Chart Overlayed)',
                   line_shape='linear', color_discrete_sequence=['orange'])

fig_bar.update_traces(marker_color='rgba(50, 171, 96, 0.6)', marker_line_color='rgb(0,0,0)',
                      marker_line_width=1.5, opacity=0.6)

fig_bar.update_layout(showlegend=True)
fig_line.update_layout(showlegend=True)

fig = px.bar(subcategory_sales, x='subCategory', y='Total Sales', title='Total Sales by Subcategory')
fig.add_trace(fig_line.data[0])

st.plotly_chart(fig, use_container_width=True)
_,_, view3, dwn3, _ = st.columns([0.15, 0.20, 0.20, 0.20, 0.20])
with view3:
    expander = st.expander("Sales by Subcategory ")
    expander.write(subcategory_sales)
with dwn3:
    st.download_button("Get Data", data=subcategory_sales.to_csv().encode("utf-8"),
                       file_name="SubCategorySales.csv", mime="text/csv")


st.divider()
sales_data = filtered_data.groupby(['gender', 'masterCategory',  'usage', 'subCategory','baseColour']).size().reset_index(name='Total Sales')


fig = px.treemap(sales_data, path=['gender', 'masterCategory', 'baseColour', 'usage', 'subCategory'],
                 values='Total Sales', title='Hierarchical view of Sales Distribution by Gender, Category, Color, and Usage',
                 color='baseColour')

# Customize hover template and other visual aspects
fig.update_traces(marker=dict(line=dict(width=0.5, color='white')), selector=dict(type='treemap'),
                  hovertemplate='<b>%{label}</b><br>Total Sales: %{value}<extra></extra>')

fig.update_layout(coloraxis_showscale=False)

st.plotly_chart(fig, use_container_width=True)
_,_, view4, dwn4, _ = st.columns([0.15, 0.20, 0.20, 0.20, 0.20])
with view4:
    expander = st.expander("Sales by Each category ")
    expander.write(subcategory_sales)
with dwn4:
    st.download_button("Get Data", data=sales_data.to_csv().encode("utf-8"),
                       file_name="SalesData.csv", mime="text/csv")
col4, col5 = st.columns(2)

sales_per_year = df.groupby('year')['id'].count().reset_index()
with col4:

    sales_per_year.columns = ['year', 'Total Sales']  # Rename the columns for clarity

    # Creating the area chart
    area_chart_fig = px.area(sales_per_year, x='year', y='Total Sales', title='Total Sales per Year')
    st.plotly_chart(area_chart_fig, use_container_width=True)

sales_per_usage = df.groupby('usage')['id'].count().reset_index()
sales_per_usage.columns = ['Usage', 'Count']  # Renaming columns

# Sort the data in descending order
sales_per_usage = sales_per_usage.sort_values(by='Count', ascending=False)
with col5:
    funnel_fig = px.funnel(sales_per_usage, x='Count', y='Usage', title='Funnel Plot of Total Sales by Usage')
    st.plotly_chart(funnel_fig, use_container_width=True)
_, view5, dwn5, view6, dwn6 = st.columns([0.15, 0.20, 0.20, 0.20, 0.20])

with view5:
    expander = st.expander("Total Sales per Year")
    expander.write(sales_per_year)

with dwn5:
    st.download_button("Get Data", data=sales_per_year.to_csv().encode("utf-8"),
                       file_name="sales_per_year.csv", mime="text/csv")
# Creating the funnel plot
with view6:
    expander = st.expander("Total Sales by Usage")
    expander.write(sales_per_usage)

with dwn6:
    st.download_button("Get Data", data=sales_per_usage.to_csv().encode("utf-8"),
                       file_name="sales_per_usage.csv", mime="text/csv")
count_by_gender_color = df.groupby(['gender', 'baseColour']).size().reset_index(name='Count')
col6, col7 = st.columns(2)

# Creating stacked bar plot
with col6:
    stacked_bar_fig = px.bar(
        count_by_gender_color,
        x='gender',
        y='Count',
        color='baseColour',
        title='Product Count by Gender and Color',
        labels={'gender': 'Gender', 'Count': 'Product Count', 'baseColour': 'Product Color'},
        barmode='stack'  # Stacked bar chart
    )

    st.plotly_chart(stacked_bar_fig, use_container_width=True)
sunburst_data = df.groupby(['gender', 'masterCategory', 'subCategory', 'articleType']).size().reset_index(name='count')

with col7:
    # Create sunburst chart with added articleType
    fig = px.sunburst(sunburst_data, path=['gender', 'masterCategory', 'subCategory', 'articleType'], values='count',color='articleType')
    fig.update_layout(title='Sunburst Chart by Gender, Category, Subcategory, and Article Type')

    st.plotly_chart(fig, use_container_width=True)
    # Show the chart
_, view7, dwn7, view8, dwn8 = st.columns([0.15, 0.20, 0.20, 0.20, 0.20])

with view7:
    expander = st.expander("Product Count by Gender and Color")
    expander.write(count_by_gender_color)

with dwn7:
    st.download_button("Get Data", data=count_by_gender_color.to_csv().encode("utf-8"),
                       file_name="count_by_gender_color.csv", mime="text/csv")
# Creating the funnel plot
with view8:
    expander = st.expander("Sales Across Gender&Product Categories")
    expander.write(sunburst_data)

with dwn8:
    st.download_button("Get Data", data=sunburst_data.to_csv().encode("utf-8"),
                       file_name="sunburst_data.csv", mime="text/csv")
df_text = pd.read_csv("styles_text.csv")

df_text = df_text.dropna(subset=['productDisplayName'])
text = ' '.join(df_text['productDisplayName'].astype(str))


# Create a WordCloud object
wordcloud = WordCloud(width=800, height=300, max_words=200, background_color='black', colormap='Accent').generate(text)

# Display the word cloud using Matplotlib
plt.figure(figsize=(10, 2))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Word Cloud')
st.pyplot(plt)  # Display in Streamlit


