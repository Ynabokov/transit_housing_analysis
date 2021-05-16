import os
import pandas as pd
import sys
import json
from qwikidata.sparql import return_sparql_query_results

# make sure we have Python 3.5+
assert sys.version_info >= (3, 5)

FILE_DIR = 'data'
DUMP_FILE_PATH = FILE_DIR + '/stations_raw.json'
CLEAR_FILE_PATH = FILE_DIR + '/stations.json'

def create_data_dir():
	os.makedirs(FILE_DIR)

def save_results(results):
	if not os.path.exists(FILE_DIR):
		create_data_dir()
	with open(DUMP_FILE_PATH, 'w') as filename:
		json.dump(results, filename, indent = 1)

# Chaoyangmen (朝阳门), Chegongzhuang (车公庄), Ping'anli (平安里) stations 
# are intentionally skipped because all three of them
# are interchange stations with lines 2 and 4 that were opened before line 6
def upload_data_from_wikidata():
	# Query created and formatted in https://query.wikidata.org/ 
	sparql_query = """
	SELECT ?station (SAMPLE(?station_en) AS ?EnglishName) (SAMPLE(?station_zh) AS ?RealName)
	 (SAMPLE(?loc) AS ?Location) (SAMPLE(?city_en) AS ?District) (SAMPLE(?op_date) AS ?OpeningDate)
	WHERE {
	  ?station wdt:P81 wd:Q6553138.
	  OPTIONAL { ?station wdt:P131 ?city. }
	  OPTIONAL { ?station wdt:P625 ?loc. }
	  ?station wdt:P1619 ?op_date.
	  OPTIONAL {
	    ?station rdfs:label ?station_en.
	    FILTER((LANG(?station_en)) = "en")
	  }
	  OPTIONAL {
	    ?station rdfs:label ?station_zh.
	    FILTER((LANG(?station_zh)) = "zh")
	  }
	  OPTIONAL {
	    ?city rdfs:label ?city_en.
	    FILTER((LANG(?city_en)) = "en")
	  }
	  FILTER((YEAR(?op_date)) = 2012 )
	}
	GROUP BY ?station
	"""
	results = return_sparql_query_results(sparql_query)
	save_results(results)
	return results


def transform_data(raw_stations):
	# Based on Robert Smith's answer @ 
	# https://stackoverflow.com/questions/31551412/how-to-select-dataframe-columns-based-on-partial-matching
	filter_cols = raw_stations.columns.to_series().str.contains('value')
	raw_stations = raw_stations[raw_stations.columns[filter_cols]]
	stations = raw_stations.drop(columns=['station.value']) # Drop a Wikidata ID, as there is no application to it
	stations.columns = ['English Name', 'Name', 'Location', 'District', 'Opening Date']
	stations[['Lng','Lat']] = stations['Location'].str.extract(pat = r'Point\((\d+.\d+) (\d+.\d+)\)',expand=True).astype(float)
	return stations

def get_stations():
	if os.path.exists(CLEAR_FILE_PATH):
		print('Using cached DataFrame from ' + CLEAR_FILE_PATH)
		return pd.read_json(CLEAR_FILE_PATH)

	if os.path.exists(DUMP_FILE_PATH):
		print('Using Wikidata json dump from ' + DUMP_FILE_PATH)
		with open(DUMP_FILE_PATH, 'r') as filename:
			data = json.load(filename)
	else:
		data = upload_data_from_wikidata()

	raw_stations = pd.json_normalize(data['results']['bindings'])
	stations = transform_data(raw_stations)

	# Save data for future use
	stations.to_json(CLEAR_FILE_PATH)
	return stations

def main():
	print(get_stations())


if __name__== "__main__":
	main()