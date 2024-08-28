import pandas as pd

ceps = pd.read_excel('Dados/dados_DBSCAN.xlsx' , dtype=str)
ceps['Latitude'] = ceps['Latitude'].astype(float)
ceps['Longitude'] = ceps['Longitude'].astype(float)

ceps_med = ceps[ceps['dbscan_outlier'] == '0']

ceps_med = ceps_med.groupby(['postcode', 'postcodeRaz']).agg({
    'Latitude': 'median',
    'Longitude': 'median',
}).reset_index()

ceps_med.to_excel('Dados/mediana_cep.xlsx', index=False)
