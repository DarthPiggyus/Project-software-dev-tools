import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# read in the csv file
car_ad_data = pd.read_csv(r'C:\Users\Darth Piggyus\Project-software-dev-tools\vehicles_us.csv')

# Define function to remove outliers based on z-score
def remove_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[(z_scores < threshold) & (z_scores > -threshold)]

# remove Null values from the data
car_ad_data['model_year'].fillna(car_ad_data['model_year'].mean(), inplace=True)
car_ad_data['cylinders'].fillna(car_ad_data['cylinders'].mean(), inplace=True)
car_ad_data['odometer'].fillna(car_ad_data['odometer'].mean(), inplace=True)
car_ad_data['paint_color'].fillna(car_ad_data['paint_color'].mode(), inplace=True)

# create a new column for the maker by taking the first word in the model column
car_ad_data['maker'] = car_ad_data['model'].apply(lambda x:x.split()[0])

# make the data easily viewable to the client
st.header('Data viewer')
show_manuf_1k_ads = st.checkbox('Include makers with less than 1000 ads')
if not show_manuf_1k_ads:
    car_ad_data = car_ad_data.groupby('maker').filter(lambda x: len(x) > 1000)

# show the breakdown of vehicle types by maker
st.dataframe(car_ad_data)
st.header('Vehicles by maker')
st.write(px.histogram(car_ad_data, 
                      x='maker', 
                      color='type'))

# creat a histogram comparing the vehicle condition to the year
#st.header('Histogram of `condition` vs `model_year`')
car_ad_data_years = remove_outliers(car_ad_data, 'model_year')
#st.write(px.histogram(car_ad_data_years, 
#                      nbins=20, 
#                      rwidth=0.8, 
#                      x='model_year', 
#                      color='condition'))

st.header('Histogram of `condition` vs `model_year`')
histogram_fig = px.histogram(car_ad_data_years, 
                              nbins=30,  
                              x='model_year', 
                              color='condition')

# Adjust the spacing between bins
histogram_fig.update_layout(bargap=0.1)  # Set the gap between bars to 0.1


st.write(histogram_fig)

st.header('Compare price distribution between makers')
# get the list of car makers
maker_list = sorted(car_ad_data['maker'].unique())
# get user inputs from a dropdown menu
maker_1 = st.selectbox('Select maker 1',
                              maker_list, index=maker_list.index('chevrolet')) # default pre-selected option)
# repeat for the second dropdown menu
maker_2 = st.selectbox('Select maker 2',
                              maker_list, index=maker_list.index('hyundai'))
# filter the dataframe 
mask_filter = (car_ad_data['maker'] == maker_1) | (car_ad_data['maker'] == maker_2)
df_filtered = car_ad_data[mask_filter]

# add a checkbox if a user wants to normalize the histogram
normalize = st.checkbox('Normalize histogram', value=True)
if normalize:
    histnorm = 'percent'
else:
    histnorm = None

# create a plotly histogram figure
st.write(px.histogram(df_filtered,
                      x='price',
                      nbins=30,
                      color='maker',
                      histnorm=histnorm,
                      barmode='overlay',
                      color_discrete_map={'chevrolet': 'red', 'hyundai': 'blue'}))

# Remove outliers from 'odometer' and 'price' columns
car_ad_data_clean = remove_outliers(car_ad_data, 'odometer')
car_ad_data_clean = remove_outliers(car_ad_data_clean, 'price')

# Default maker
default_maker = 'ford'

# Scatter plot using Plotly Express
st.header('Scatter Plot of Price vs Odometer')

# Dropdown for selecting maker
selected_maker = st.selectbox('Select Maker', 
                              options=maker_list, 
                              index=list(maker_list).index(default_maker))

# Filter data based on selected maker
filtered_data = car_ad_data_clean[car_ad_data_clean['maker'] == selected_maker]

# Scatter plot using Plotly Express
scatter_plot = px.scatter(filtered_data, 
                          x='odometer', 
                          y='price', 
                          title=f'Price vs Odometer for {selected_maker.capitalize()}')
st.plotly_chart(scatter_plot)

# Count frequency of each paint color
color_counts = car_ad_data['paint_color'].value_counts().to_dict()

# Create word cloud with each color displayed in its own color
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'white'  # We'll set all words to be displayed in white color

# Create word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black')

# Generate word cloud from frequencies
wordcloud.generate_from_frequencies(color_counts)

# Apply color function
wordcloud.recolor(color_func=color_func)

# Display word cloud
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud of Vehicle Colors')
st.pyplot(fig)

# create a box plot showing price distribution between fuel types and price
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=car_ad_data_clean, x='fuel', y='price', ax=ax)
ax.set_title('Comparison of Price by Fuel Type')
ax.set_xlabel('Fuel Type')
ax.set_ylabel('Price')
st.pyplot(fig)