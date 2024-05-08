import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read in the csv file
car_ad_data = pd.read_csv(r'vehicles_us.csv')

# define function to remove outliers based on z-score
def remove_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[(z_scores < threshold) & (z_scores > -threshold)]

# create a new column for the maker by taking the first word in the model column
car_ad_data['maker'] = car_ad_data['model'].apply(lambda x:x.split()[0])

# remove Null values from the data
# Group by 'maker' and 'model' and calculate median model year
median_model_year = car_ad_data.groupby(['maker', 'model'])['model_year'].median()

# Fill missing values in 'model_year' column with median values based on groups
car_ad_data['model_year'] = car_ad_data.apply(
    lambda row: median_model_year.loc[(row['maker'], row['model'])] if pd.isnull(row['model_year']) else row['model_year'], axis=1)

# Group by 'model' and 'model_year' and calculate median number of cylinders
median_cylinders = car_ad_data.groupby(['model', 'model_year'])['cylinders'].median()

# Fill missing values in 'cylinders' column with median values based on groups
car_ad_data['cylinders'] = car_ad_data.apply(
    lambda row: median_cylinders.loc[(row['model'], row['model_year'])] if pd.isnull(row['cylinders']) else row['cylinders'], axis=1)

# Calculate the average of the 'cylinders' column
average_cylinders = car_ad_data['cylinders'].mean()

# Fill missing values in the 'cylinders' column with the average
car_ad_data['cylinders'] = car_ad_data['cylinders'].fillna(average_cylinders)

# Group by 'model' and 'model_year' and calculate median number of odometer
median_odometer = car_ad_data.groupby(['model', 'model_year'])['odometer'].median()

# Fill missing values in 'odometer' column with median values based on groups
car_ad_data['odometer'] = car_ad_data.apply(
    lambda row: median_odometer.loc[(row['model'], row['model_year'])] if pd.isnull(row['odometer']) else row['odometer'], axis=1)

# Calculate the average of the 'odometer' column
average_odometer = car_ad_data['odometer'].mean()

# Fill missing values in the 'odometer' column with the average
car_ad_data['odometer'] = car_ad_data['odometer'].fillna(average_odometer)

# Group by 'model' and 'model_year' and find the mode of 'paint_color'
mode_paint_color = car_ad_data.groupby(['model', 'model_year'])['paint_color'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Fill missing values in 'paint_color' column with mode values based on groups
car_ad_data['paint_color'] = car_ad_data.apply(
    lambda row: mode_paint_color.loc[(row['model'], row['model_year'])] if pd.isnull(row['paint_color']) else row['paint_color'], axis=1)

# Calculate the average of the 'paint_color' column
average_color = car_ad_data['paint_color'].mode().iloc[0]

# Fill missing values in the 'paint_color' column with the average
car_ad_data['paint_color'] = car_ad_data['paint_color'].fillna(average_color)

# Group by 'model' and 'model_year' and find the mode of 'is_4wd'
mode_is_4wd = car_ad_data.groupby(['model', 'model_year'])['is_4wd'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Fill missing values in 'is_4wd' column with mode values based on groups
car_ad_data['is_4wd'] = car_ad_data.apply(
    lambda row: mode_is_4wd.loc[(row['model'], row['model_year'])] if pd.isnull(row['is_4wd']) else row['is_4wd'], axis=1)

# Replace any remaining null values in 'is_4wd' column with 0
car_ad_data['is_4wd'] = car_ad_data['is_4wd'].fillna(0)

# convert columns to integer type
car_ad_data['model_year'] = car_ad_data['model_year'].astype(int)
car_ad_data['cylinders'] = car_ad_data['cylinders'].astype(int)
car_ad_data['odometer'] = car_ad_data['odometer'].astype(int)

# add title and introduction
st.title('Vehicle Sales Analysis')
st.write("""
Welcome to the Vehicle Sales Analysis App! This dashboard provides insights into used vehicle sales data, including information about makers, models, prices, and more. Using this data you'll be able to compare different aspects of the used vehicle sales and make clear inferences from the charts provided.
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

# update axes labels if fig is not None
fig4.update_xaxes(title='Maker')
fig4.update_yaxes(title='Vehicles Sold')
st.plotly_chart(fig4)

st.write("""
This plot shows that Ford and Chevrolet far exceed other makers in the volume of used vehicles sold. The largest portion of these sales for the two companies are comprised of truck sales.
""")

# create a histogram comparing the vehicle condition to the year
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

# remove outliers from 'odometer', 'price' and 'days_listed' columns
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
normalize = st.checkbox('Normalize histogram 1', value=True)
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

# dropdown for selecting maker
selected_maker = st.selectbox('Select Maker', 
                              options=maker_list, 
                              index=list(maker_list).index(default_maker))

# filter data based on selected maker
filtered_data = car_ad_data_clean[car_ad_data_clean['maker'] == selected_maker]

# scatter plot using Plotly Express
scatter_plot = px.scatter(filtered_data, 
                          x='odometer', 
                          y='price', 
                          title=f'Price vs Odometer for {selected_maker.capitalize()}')
st.plotly_chart(scatter_plot)

st.write("""
This plot compares a vehicles odometer to the price it sold for, based on it's maker. The plot shows a clear trend as the odometer increases the price of the vehicles go down.
""")

# count frequency of each paint color
color_counts = car_ad_data['paint_color'].value_counts().to_dict()

# create word cloud with each color displayed in its own color
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'white'  # We'll set all words to be displayed in white color

# create word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black')

# generate word cloud from frequencies
wordcloud.generate_from_frequencies(color_counts)

# apply color function
wordcloud.recolor(color_func=color_func)

# display word cloud
st.header('Word Cloud of Vehicle Colors')
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.write("""
Here you can see in the word cloud that white cars are the most common color of car resold.
""")

# create a box plot showing price distribution between fuel types and price
st.header('Comparison of Vheicle Price by Fuel Type')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=car_ad_data_clean, x='fuel', y='price', ax=ax)
ax.set_xlabel('Fuel Type')
ax.set_ylabel('Price in USD')
st.pyplot(fig)

st.write("""
This plot is showing that, on average, hybrid and electric vehicles sell at a much lower price than other fuel types do. While gas vehicles have a number of outliers on the upper end of the graph though their median price is on par with hybrids and electrics.
""")

# convert 'is_4wd' column to boolean type
car_ad_data['is_4wd'] = car_ad_data['is_4wd'].astype(bool)

# create a histogram comparing days listed on the market for 4WD and non-4WD vehicles
st.header('Histogram of Days Listed by 4WD')
color_map = {True: 'red', False: 'blue'}
fig2 = px.histogram(car_ad_data, x='days_listed', color='is_4wd', 
                   labels={'days_listed': 'Days Listed', 'is_4wd': 'Vehicle Type'},
                   barmode='overlay', histnorm='', color_discrete_map=color_map)
fig2.for_each_trace(lambda t: t.update(name='4WD' if t.name == 'True' else 'Non-4WD'))
fig2.update_layout(yaxis_title='Number of Vehicles')

# add a checkbox if a user wants to normalize the histogram
normalize = st.checkbox('Normalize histogram 2', value=True)
if normalize:
    histnorm = 'percent'
else:
    histnorm = None

# update histogram figure with the new histnorm value
fig2.update_traces(histnorm=histnorm)

st.plotly_chart(fig2)

st.write("""
Here you can see there isn't really a large difference in the time it takes to sell between 4wd and non-4wd vehicles, even though there are far more 4wd vehicles.
""")

st.header('Conclusion')
st.write("""
In conclusion we can see that Ford and Chevy are the most prolific makers in the used sales market. Most vehicles appear to be white or black trucks and SUVs in excellent condition. The higher the odometer on the vehicle, however, the less it will typically sell for despite the maker.
""")