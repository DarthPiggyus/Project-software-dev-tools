import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# read in the csv file
car_ad_data = pd.read_csv(r'vehicles_us.csv')

# Define function to remove outliers based on z-score
def remove_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[(z_scores < threshold) & (z_scores > -threshold)]

# remove Null values from the data
car_ad_data['model_year'].fillna(car_ad_data['model_year'].mean(), inplace=True)
car_ad_data['cylinders'].fillna(car_ad_data['cylinders'].mean(), inplace=True)
car_ad_data['odometer'].fillna(car_ad_data['odometer'].mean(), inplace=True)
car_ad_data['paint_color'].fillna(car_ad_data['paint_color'].mode().iloc[0], inplace=True)
car_ad_data['is_4wd'].fillna(0, inplace=True)

# create a new column for the maker by taking the first word in the model column
car_ad_data['maker'] = car_ad_data['model'].apply(lambda x:x.split()[0])

# add title and introduction
st.title('Vehicle Sales Analysis')
st.write("""
Welcome to the Vehicle Sales Analysis App! This dashboard provides insights into used vehicle sales data, including information about makers, models, prices, and more. Using this data you'll be able to compare different aspects of the used vehicle sales and make clear inferences from the plots provided.
""")

# make the data easily viewable to the client
st.header('Data viewer')
show_manuf_1k_ads = st.checkbox('Include makers with less than 1000 ads')
if not show_manuf_1k_ads:
    car_ad_data = car_ad_data.groupby('maker').filter(lambda x: len(x) > 1000)

# show the breakdown of vehicle types by maker
st.dataframe(car_ad_data)

st.header('Vehicles by maker')
fig4 = px.histogram(car_ad_data,  x='maker', color='type')

# Update axes labels if fig is not None
fig4.update_xaxes(title='Maker')
fig4.update_yaxes(title='Vehicles Sold')
st.plotly_chart(fig4)

st.write("""
This plot shows that Ford and Chevrolet far exceed other makers in the volume of used vehicles sold. The largest portion of these sales for the two companies are comprised of truck sales.
""")

# creat a histogram comparing the vehicle condition to the year
car_ad_data_years = remove_outliers(car_ad_data, 'model_year')

st.header('Vehicle Condition by Model Year')
histogram_fig = px.histogram(car_ad_data_years, 
                              nbins=30,  
                              x='model_year', 
                              color='condition')

# adjust the spacing between bins
histogram_fig.update_layout(bargap=0.1)  # Set the gap between bars to 0.1
histogram_fig.update_xaxes(title='Model Year')
histogram_fig.update_yaxes(title='Vehicles Sold')
st.plotly_chart(histogram_fig)

st.write("""
         In this plot you can see that the most cars sold were manufactured in 09-10 and are still in good to excellent condition. Vehilces made prior to 2000 don't seem to sell as often but are still in fairly good condition.
""")

# remove outliers from 'odometer' and 'price' columns
car_ad_data_clean = remove_outliers(car_ad_data, 'odometer')
car_ad_data_clean = remove_outliers(car_ad_data_clean, 'price')
car_ad_data_clean = remove_outliers(car_ad_data_clean, 'days_listed')

st.header('Compare price distribution between makers')
# get the list of car makers
maker_list = sorted(car_ad_data_clean['maker'].unique())
# get user inputs from a dropdown menu
maker_1 = st.selectbox('Select maker 1',
                              maker_list, index=maker_list.index('chevrolet')) # default pre-selected option)
# repeat for the second dropdown menu
maker_2 = st.selectbox('Select maker 2',
                              maker_list, index=maker_list.index('dodge'))
# filter the dataframe 
mask_filter = (car_ad_data_clean['maker'] == maker_1) | (car_ad_data_clean['maker'] == maker_2)
df_filtered = car_ad_data_clean[mask_filter]

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
                      color_discrete_map={maker_1: 'red', maker_2: 'blue'}))

st.write("""
Here you're able to compare 2 different manufactures and see the difference in the price ranges between them. Using the button to normalize the graph you convert the y-axis into a percentage of total vehicles sales instead of show the number of vehicles sold.
""")

# define the default maker
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

st.write("""
This plot compares a vehicles odometer to the price it sold for, based on it's maker. The plot shows a clear trend as the odometer increases the price of the vehicles go down.
""")

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
st.header('Word Cloud of Vehicle Colors')
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.write("""
Her you can see in the word cloud that white cars are the most common color of car resold.
""")

# create a box plot showing price distribution between fuel types and price
st.header('Comparison of Vheicle Price by Fuel Type')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=car_ad_data_clean, x='fuel', y='price', ax=ax)
ax.set_xlabel('Fuel Type')
ax.set_ylabel('Price in USD')
st.pyplot(fig)

st.write("""
This plot is showing that, on average, hybrid and electric vehicles sell at a much lower price than other fuel types do. While gas vehicles have a number of outliers on the upper end of the graph.
""")

# Convert 'is_4wd' column to boolean type
car_ad_data['is_4wd'] = car_ad_data['is_4wd'].astype(bool)

# Create a histogram comparing days listed on the market for 4WD and non-4WD vehicles
st.header('Histogram of Days Listed by 4WD')
color_map = {True: 'red', False: 'blue'}
fig2 = px.histogram(car_ad_data, x='days_listed', color='is_4wd', 
                   labels={'days_listed': 'Days Listed', 'is_4wd': '4WD'},
                   barmode='overlay', histnorm='percent', color_discrete_map=color_map)
fig2.update_traces(name=['4WD', 'Non-4WD'])
st.plotly_chart(fig2)

st.write("""
Here you can se there isn't really a large difference in the sales between 4wd and non-4wd vehicles.
""")