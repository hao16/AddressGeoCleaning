import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import geopandas as gpd


# Carregar o dataframe 'cep'
# Supondo que o dataframe 'cep' já foi carregado contendo as colunas Latitude, Longitude, Zipcode, ZipcodeRaz

def process_hdbscan(df, min_cluster_size=5, min_samples=1):
    if len(df) < min_cluster_size:
        # Se o grupo tiver menos pontos do que o tamanho mínimo do cluster, não processa
        return df.assign(cluster=-1, is_outlier=True)

    # Normalizar os dados de Latitude e Longitude
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Latitude', 'Longitude']])

    # Ajustar o min_cluster_size para o tamanho do grupo
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(len(df)/2), min_samples=min_samples, gen_min_span_tree=True)
    df['cluster'] = clusterer.fit_predict(X)

    # Identificar outliers (pontos rotulados com -1 pelo HDBSCAN)
    df['is_outlier'] = df['cluster'] == -1

    # Filtrar os CEPs que não são outliers
    # df_filtered = df[~df['is_outlier']]

    return df


# Primeiro passo: Cluster por raiz do CEP (ZipcodeRaz)

gdf = gpd.read_file('Dados/source.geojson')
gdf = gdf.replace('', None, regex=False)
gdf['postcode'] = gdf['postcode'].str.replace('-', '')
gdf = gdf.dropna(axis=1, how='all')
gdf = gdf.dropna(subset=['postcode'])
gdf = gdf[gdf['postcode'].str.isnumeric()]

gdf['postcodeRaz'] = gdf['postcode'].str[:5]

gdf['Latitude'] = gdf.geometry.y
gdf['Longitude'] = gdf.geometry.x
gdf['LatitudeBR'] = gdf['Latitude'].astype(str).str.replace('.', ',')
gdf['LongitudeBR'] = gdf['Longitude'].astype(str).str.replace('.', ',')

cep_clusters_by_root = gdf.groupby('postcodeRaz', group_keys=False).apply(lambda x: process_hdbscan(x, min_samples=1))

cep_clusters_by_root[['postcodeRaz', 'postcode', 'LatitudeBR', 'LongitudeBR', 'cluster', 'is_outlier']].to_csv('Dados/outliersRaiz.csv', sep='$')
# Segundo passo: Cluster por CEP completo
cep_clusters_final = cep_clusters_by_root.groupby('postcode', group_keys=False).apply(lambda x: process_hdbscan(x))
