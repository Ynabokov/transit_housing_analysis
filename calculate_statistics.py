import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
from haversine import haversine_vector, Unit
from tqdm import tqdm

# make sure we have Python 3.5+
assert sys.version_info >= (3, 5)

import data_collection as dc
from get_stations import get_stations

FILE_DIR = 'data/stat'
BST_FILE_PATH = FILE_DIR + '/before_stations.json'
BWST_FILE_PATH = FILE_DIR + '/before_wout_stations.json'
AST_FILE_PATH = FILE_DIR + '/after_stations.json'
AWST_FILE_PATH = FILE_DIR + '/after_wout_stations.json'

def create_data_dir():
	os.makedirs(FILE_DIR)

def drop_unused_cols(df):
  return df[['Lng', 'Lat', 'tradeTime',	'totalPrice']]

def divide_by_time(df):
  before_stations = df[df['tradeTime'] <= '2012-12-30']
  after_stations = df[df['tradeTime'] > '2012-12-30']
  return (before_stations, after_stations)

def calc_closest_station(house, stations):
  distances = haversine_vector( (house['Lat'],house['Lng']), stations, Unit.METERS, comb=True)
  # According to https://blog.mapbox.com/fast-geodesic-approximations-with-cheap-ruler-106f229ad016, 
  # the error rate for haversine at Beijing latitude is about 0.05%, so use slightly larger max distance value here
  if distances.min() <= 2100:
    return distances.argmin()
  return np.nan

def get_closest_station(houses, list_stations):
  cl_houses = houses.copy()
  tqdm.pandas()
  cl_houses['station'] = cl_houses.progress_apply(calc_closest_station, stations=list_stations, axis=1)
  houses_stations = cl_houses[cl_houses['station'].notna()]
  houses_wout_stations = cl_houses[cl_houses['station'].isna()]
  return (houses_stations, houses_wout_stations)

def prepare_dfs():
	stat_housing = drop_unused_cols(dc.housing_Beijing_with_time)
	before, after = divide_by_time(stat_housing)
	stations = get_stations()
	list_stations = list(zip(stations['Lat'],stations['Lng']))
	before_stations, before_wout_stations = get_closest_station(before, list_stations)
	after_stations, after_wout_stations = get_closest_station(after, list_stations)

	before_wout_stations = before_wout_stations[before_wout_stations['tradeTime']>='2011-05']
	before_stations = before_stations[before_stations['tradeTime']>='2011-05']

	# Save data for future use
	if not os.path.exists(FILE_DIR):
		create_data_dir()

	before_stations.to_json(BST_FILE_PATH)
	before_wout_stations.to_json(BWST_FILE_PATH)
	after_stations.to_json(AST_FILE_PATH)
	after_wout_stations.to_json(AWST_FILE_PATH)
	return (before_stations, before_wout_stations, after_stations, after_wout_stations)

def prepare_data():
	if os.path.exists(FILE_DIR) and \
	   len ([file for file in os.listdir(FILE_DIR) if file.endswith('.json')]) == 4:
		print('Using cached DataFrames from ' + FILE_DIR)
		before_stations = pd.read_json(BST_FILE_PATH, convert_dates=['tradeTime'])
		before_wout_stations = pd.read_json(BWST_FILE_PATH, convert_dates=['tradeTime'])
		after_stations = pd.read_json(AST_FILE_PATH, convert_dates=['tradeTime'])
		after_wout_stations = pd.read_json(AWST_FILE_PATH, convert_dates=['tradeTime'])
		return (before_stations, before_wout_stations, after_stations, after_wout_stations)
	return prepare_dfs()

def draw_boxplots(before_stations, before_wout_stations, after_stations, after_wout_stations):
	bs_pl = before_stations.assign(Location='Before/Near Staions')
	bws_pl =  before_wout_stations.assign(Location='Before/Not Near Staions')
	as_pl = after_stations.assign(Location='After/Near Staions')
	aws_pl = after_wout_stations.assign(Location='After/Not Near Staions')

	cdf = pd.concat([bs_pl, as_pl, bws_pl, aws_pl])    

	f, ax = plt.subplots(figsize=(7, 6))
	ax.set_xscale("log")
	sns.boxplot(x="totalPrice", y = "Location", data=cdf)  
	plt.savefig('data/box.png')

def print_stat(before_stations, before_wout_stations, after_stations, after_wout_stations):
	print(before_stations['totalPrice'].count())
	print(before_wout_stations['totalPrice'].count())
	print(after_stations['totalPrice'].count())
	print(after_wout_stations['totalPrice'].count())
	print("--------------==========-----------------")
	print(before_stations['totalPrice'].mean())
	print(before_wout_stations['totalPrice'].mean())
	print(after_stations['totalPrice'].mean())
	print(after_wout_stations['totalPrice'].mean())
	print("--------------==========-----------------")
	print(before_stations['totalPrice'].median())
	print(before_wout_stations['totalPrice'].median())
	print(after_stations['totalPrice'].median())
	print(after_wout_stations['totalPrice'].median())
	# print mannwhitneyu
	print(stats.mannwhitneyu(before_stations['totalPrice'], after_stations['totalPrice']).pvalue)
	print(stats.mannwhitneyu(after_wout_stations['totalPrice'], after_stations['totalPrice']).pvalue)
	print(stats.mannwhitneyu(before_stations['totalPrice'], before_wout_stations['totalPrice']).pvalue)

def plot_difference(before_stations, before_wout_stations, after_stations, after_wout_stations):
	sns.set_theme(style="whitegrid")
	plt.figure(figsize = (15, 6))
	# build plot 1
	plt.subplot(1, 2, 1)
	dii_ba = before_stations.groupby(['tradeTime']).aggregate('median')['totalPrice'] - before_wout_stations.groupby(['tradeTime']).aggregate('median')['totalPrice'] 
	plt.plot(dii_ba[dii_ba>0], 'g.')
	plt.plot(dii_ba[dii_ba<0], 'r.')
	plt.plot(dii_ba[dii_ba == 0], 'y.')
	plt.xlabel("Time", fontsize = 12)
	plt.xticks(rotation = 25)
	plt.title("Difference between daily price median of houses \n  close to stations/far from stations before opening date", fontsize = 12)
	plt.ylabel("Difference", fontsize = 12)
	plt.subplot(1, 2, 2)
	dii_a = after_stations.groupby(['tradeTime']).aggregate('median')['totalPrice'] - after_wout_stations.groupby(['tradeTime']).aggregate('median')['totalPrice'] 
	plt.plot(dii_a[dii_a>0], 'g.')
	plt.plot(dii_a[dii_a<0], 'r.')
	plt.plot(dii_a[dii_a == 0], 'y.')
	plt.xlabel("Time", fontsize = 12)
	plt.xticks(rotation = 25)
	plt.title("Difference between daily price median of houses \n   close to stations/far from stations after opening date", fontsize = 12)
	plt.ylabel("Difference", fontsize = 12)
	plt.show()
	plt.savefig('data/diff.png')

def plot_log_distr(before_stations, before_wout_stations, after_stations, after_wout_stations):
	
	plt.figure(figsize = (18, 7))
	sns.set_theme(style="whitegrid")
	plt.subplot(2, 2, 1)
	fig = plt.plot(before_stations['tradeTime'], before_stations['totalPrice'], 'c.', alpha=0.5)
	plt.yscale('log')
	plt.title("Total prices of each house near stations \n sold before opening")

	plt.subplot(2, 2, 2)
	plt.plot(before_wout_stations['tradeTime'], before_wout_stations['totalPrice'], 'c.', alpha=0.5)
	plt.yscale('log')
	plt.title("Total prices of each house not near stations \n sold before opening")

	plt.subplot(2, 2, 3)
	plt.plot(after_stations['tradeTime'], after_stations['totalPrice'], 'c.', alpha=0.5)
	plt.yscale('log')
	plt.title("\n sold after opening")

	plt.subplot(2, 2, 4)
	plt.plot(after_wout_stations['tradeTime'], after_wout_stations['totalPrice'], 'c.', alpha=0.5)
	plt.yscale('log')
	plt.title("sold after opening")

	plt.show()
	plt.savefig('data/distr.png')

def main():
	before_stations, before_wout_stations, after_stations, after_wout_stations = prepare_data()
	draw_boxplots(before_stations, before_wout_stations, after_stations, after_wout_stations)
	print_stat(before_stations, before_wout_stations, after_stations, after_wout_stations)
	plot_difference(before_stations, before_wout_stations, after_stations, after_wout_stations)
	plot_log_distr(before_stations, before_wout_stations, after_stations, after_wout_stations)

if __name__== "__main__":
	main()