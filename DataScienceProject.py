import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.backends.backend_pdf

# read in data
properties = pd.read_csv('assignment_data.csv')

# check basic properties
pd.options.display.float_format = '{:.8g}'.format
properties.head()
properties.info()
properties.describe()
properties.describe(include=['O'])

# check if any 'id' occur more than once
id_filter = properties['id'].value_counts() > 1
id_filter.any()

# check how many listing do not have living_area
zero_living_area = properties[properties['living_area'] == 0]
len(zero_living_area)

"""
4910 entries, 7 features
- 'id', no duplicates in id
- 'title', no Nans, 2404 unique titles, Apartment for Sale in Nagüeles most popular (276 counts)
- 'features', 941 Nans, 3372 unique features, Emissions rating  Awaiting details  Energy con... most popular (72 times)
- 'living_area', - 3 Nan, but many zeros (4015 rows)
- 'total_area', - 1 Nan, 1249 rows are zero
- 'plot_area', - 3 Nans, 3097 are zero
- 'price' - no Nans, 233 has zero price
"""

# PART I
props_one = properties.copy()


def classify_type(df_data, from_label, types):
    """
    Takes data frame column and look for a specific key word in it to determine the type of property
    :param df_data: data frame with properties' data
    :param from_label: name of the column to look for a key word in
    :param types: dictionary with key words and corresponding property types
    :return: data frame with new column 'type'
    """
    for x in types.keys():
        str_filter = df_data[from_label].str.contains(x, case=False)
        df_data.loc[str_filter, 'type'] = x
    df_data['type'] = df_data['type'].map(types)

types_dict = {'apartment':'apartments', 'house':'houses', 'penthouse':'apartments', 'duplex':'apartments',
              'villa':'houses', 'country estate':'houses', 'moradia':'houses', 'quinta':'houses', 'plot':'plots',
              'land':'plots'}

classify_type(props_one, 'title', types_dict)

# check how many listings do not have type
props_one['type'].isnull().sum()
nan_type_index = np.where(props_one['type'].isnull())[0]
props_one.iloc[nan_type_index]


def classify_location(df_data, from_label, locations):
    """
    Takes data frame column and look for a specific key word in it to determine the location of property
    :param df_data: data frame with properties' data
    :param from_label: name of the column to look for a key word in
    :param locations: dictionary with key words and corresponding property locations
    :return: data frame with new column 'location name'
    """
    for x in locations.keys():
        str_filter = df_data[from_label].str.contains(x, case=False)
        df_data.loc[str_filter, 'location name'] = x
    df_data['location name'] = df_data['location name'].map(locations)

locations_dict = {'Alenquer':'Alenquer', 'Quinta da Marinha':'Quinta da Marinha', 'Golden Mile':'Golden Mile',
                  'Nagüeles':'Nagüeles', 'Nagueles':'Nagüeles'}

classify_location(props_one, 'title', locations_dict)


def identify_features(df_data, from_label, features):
    """
    Takes data frame column and look for a specific key word in it to determine if the property has a specific feature
    :param df_data: data frame with properties' data
    :param from_label: name of the column to look for a key word in
    :param features: dictionary with key words and corresponding features
    :return: data frame with new columns according to dictionary values
    """
    for x in features.keys():
        str_filter = df_data[from_label].str.contains(x, case=False, na=False)
        df_data.loc[str_filter, features[x]] = 1
        df_data.loc[:,features[x]].fillna(0, inplace=True)

desired_features = {'pool':'pool', 'sea view':'sea view', 'garage':'garage', 'sea/lake view':'sea view'}
identify_features(props_one, 'title', desired_features)
identify_features(props_one, 'features', desired_features)

properties_part_one = props_one[['id', 'location name', 'type', 'title', 'features', 'pool', 'sea view', 'garage']]
properties_part_one.to_csv('python_part_one.csv')


# PART II
props_two = props_one.copy()
props_two.drop(['title', 'features', 'pool', 'sea view', 'garage'], axis=1, inplace=True)

# fill Nans from area columns with zero values
for i in ['living_area', 'total_area', 'plot_area']:
    props_two[i].fillna(0, inplace=True)

# drop Nans from type
props_two.drop(props_two.index[nan_type_index], inplace=True)


def determine_area(df_data):
    """
    Takes a listing and calculates the area of property choosing higher between living and total area
    for houses and apartments or plot area for plots
    :param df_data: data frame with properties' data
    :return: a data frame with a new column area
    """
    for row in df_data.itertuples():
        if df_data.loc[row.Index, 'type'] == 'plots':
            df_data.loc[row.Index, 'area'] = df_data.loc[row.Index, 'plot_area']
        else:
            if df_data.loc[row.Index, 'total_area'] >= df_data.loc[row.Index, 'living_area']:
                df_data.loc[row.Index, 'area'] = df_data.loc[row.Index, 'total_area']
            else:
                df_data.loc[row.Index, 'area'] = df_data.loc[row.Index, 'living_area']
    df_data.drop(['living_area', 'total_area', 'plot_area'], axis=1, inplace=True)

determine_area(props_two)


def calculate_price_per_meter(df_data):
    """
    Takes listing price and divides it by area. Deletes listings with zero price or zero area.
    :param df_data: data frame with properties' data
    :return: data frame with a new column 'price per sq m'
    """
    zero_price = df_data['price'] == 0
    df_data.drop(df_data.index[zero_price], inplace=True)
    zero_area = df_data['area'] == 0
    df_data.drop(df_data.index[zero_area], inplace=True)
    df_data['price per sq m'] = df_data['price'] / df_data['area']

calculate_price_per_meter(props_two)


def calculate_means(df_data):
    """
    Takes price per sq m for a group of specific type and location listings and calculates a mean
    :param df_data: data frame with properties' data
    :return: a dictionary with type-location tuples as keys and mean price values
    """
    means = {}
    types = ['apartments', 'houses', 'plots']
    locations = ['Alenquer', 'Quinta da Marinha', 'Golden Mile', 'Nagüeles']
    combinations = [(x, y) for x in types for y in locations]
    for i in combinations:
        type_filter = df_data['type'] == i[0]
        location_filter = df_data['location name'] == i[1]
        filter_comb = type_filter & location_filter
        mean = df_data.loc[filter_comb, 'price per sq m'].mean()
        means[i] = mean
    return means

mean_prices = calculate_means(props_two)


def discriminate_valuation(df_data, means):
    """
    Compares every listing price per sq m with a mean for a given type and location
    :param df_data: data frame with properties' data
    :param means: dictionary with type-location tuples as keys and mean price values
    :return: a data frame with new columns 'over-valued', 'under-valued', 'normal'
    """
    for row in df_data.itertuples():
        mean = means[(df_data.loc[row.Index, 'type'], df_data.loc[row.Index, 'location name'])]
        price = df_data.loc[row.Index, 'price per sq m']
        if price > 1.1 * mean:
            df_data.loc[row.Index, 'over-valued'] = 1
        elif price < 0.9 * mean:
            df_data.loc[row.Index, 'under-valued'] = 1
        else:
            df_data.loc[row.Index, 'normal'] = 1
    for i in ['over-valued', 'under-valued', 'normal']:
        df_data.loc[:, i].fillna(0, inplace=True)

discriminate_valuation(props_two, mean_prices)

properties_part_two = props_two[['id', 'location name', 'type', 'area', 'price', 'over-valued', 'under-valued', 'normal']]
properties_part_two.to_csv('python_part_two.csv')


def plot_scatter_valuation(df_data, type_location, mean):
    """
    Creates scatter plot with listing area and price per sq m showing under and over valued properties
    and the mean price per category
    :param df_data: data frame with properties' data
    :param type_location: a tuple with listing type and location
    :param mean: mean price per sq m for given category (type and location)
    :return: scatterplot
    """
    type_filter = df_data['type'] == type_location[0]
    location_filter = df_data['location name'] == type_location[1]
    under_filter = df_data['under-valued'] == 1
    over_filter = df_data['over-valued'] == 1
    under = type_filter & location_filter & under_filter
    over = type_filter & location_filter & over_filter
    area = df_data.loc[type_filter & location_filter, 'area']
    plt.plot(df_data.loc[under, 'area'], df_data.loc[under, 'price per sq m'], "o",label='under-valued',color='purple')
    plt.plot(df_data.loc[over, 'area'], df_data.loc[over, 'price per sq m'], "o", label='over-valued', color='green')
    plt.plot([0, max(area)], [mean, mean], label='mean price per sq m', color='red', linestyle='--')
    plt.title(str(type_location[0]) + " in " + str(type_location[1]))
    plt.xlabel('property area')
    plt.ylabel('price per sq m')
    plt.legend()

pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

for i in mean_prices.keys():
    if np.isnan(mean_prices[i]):
        continue
    plt.figure()
    plot_scatter_valuation(props_two, i, mean_prices[i])
    pdf.savefig()
    plt.close()
pdf.close()


props_out = props_two.copy()

"""delete outliers from:
- apartments in Alenquer where area > 4000 and price > 1600
- apartments in Golden Mile where area > 5000 
- apartments in Nagüeles where area > 500
- houses in Alenquer where area > 250000 and price > 5000
- houses in Quinta da Marinha where area > 12000 
- houses in Golden Mile where area > 1000 and price > 15000
- houses in Nagüeles where area > 3000 and price > 10000
- plots in Alenquer where price > 1500
- plots in Nagüeles where area > 8000 
"""

type_filter = props_out['type'] == 'apartments'
location_filter = props_out['location name'] == 'Alenquer'
area_filter = props_out['area'] > 4000
price_filter = props_out['price per sq m'] > 1600
filter_combined = type_filter & location_filter & area_filter
filter_combined = type_filter & location_filter & price_filter
props_out[filter_combined]

# id to be removed:
# 1743039, 167659, 822905, 894884, 3367445, 434996, 732880, 334942, 431629, 2391890, 130728, 327708, 1606969, 768631, 768637
# 2049815, 1563979, 1563980, 1563985, 1739715, 3232786, 3464573, 2923032,  2930310, 1629401, 2710916, 3100953

out_id_list = [1743039, 167659, 822905, 894884, 3367445, 434996, 732880, 334942, 431629, 2391890, 130728, 327708,
               1606969, 768631, 768637, 2049815, 1563979, 1563980, 1563985, 1739715, 3232786, 3464573, 2923032,
               2930310, 1629401, 2710916, 3100953]

for i in out_id_list:
    props_out.drop(props_out[props_out.id == i].index, inplace=True)


pdf = matplotlib.backends.backend_pdf.PdfPages("output_outliers.pdf")

for i in mean_prices.keys():
    if np.isnan(mean_prices[i]):
        continue
    plt.figure()
    plot_scatter_valuation(props_out, i, mean_prices[i])
    pdf.savefig()
    plt.close()
pdf.close()





















