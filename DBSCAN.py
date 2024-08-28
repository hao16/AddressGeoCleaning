from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import geopandas as gpd
from tqdm import tqdm


class TratamentoGeoOutliers:
    def __init__(self, cep_raz_column, cep_column, lat_column, long_column):
        self.dinamic = True
        self.precision = True
        self.cep_raz_column = cep_raz_column
        self.cep_column = cep_column
        self.lat_column = lat_column
        self.long_column = long_column

    @staticmethod
    def lat_lon_to_cartesian(lat, lon):
        R = 6371000  # Raio da Terra em metros
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)
        return x, y, z

    @staticmethod
    def meters_to_degrees(meters, at_latitude=0):
        # 1 grau de latitude é aproximadamente 111,320 metros
        latitude_degrees = meters / 111320

        # 1 grau de longitude em metros depende da latitude
        longitude_degrees = meters / (111320 * np.cos(np.radians(at_latitude)))

        return latitude_degrees, longitude_degrees

    def calcular_eps(self, df_orig):
        """
        Calcula o valor de eps dinamicamente com base na maior sequência de faixas que possua mais do que 10% dos dados
        ou o maior isolado se ele tiver mais do que 30%.
        """
        df = df_orig.copy().reset_index(drop=True)

        if self.dinamic:

            df['x'], df['y'], df['z'] = self.lat_lon_to_cartesian(df[self.lat_column], df[self.long_column])

            # Criar uma matriz de coordenadas
            coordinates = df[['x', 'y', 'z']].values

            # Calcular a matriz de distâncias usando pdist
            distances = pdist(coordinates)

            # Contar a quantidade de distâncias em faixas de 300 metros
            bins = np.arange(0, 10000 + 300, 300)
            hist, bin_edges = np.histogram(distances, bins=bins)

            # Calcular a porcentagem de distâncias em cada faixa
            total_distances = distances.size
            hist_percent = (hist / total_distances) * 100

            # Encontrar a maior sequência de faixas com mais de 10% dos dados
            max_sequence_sum = 0
            max_sequence_center = 0
            current_sequence_sum = 0
            current_sequence_start = 0

            if self.precision:
                mean_percent = np.mean(hist_percent)
                std_percent = np.std(hist_percent)
                threshold = mean_percent + std_percent
            else:
                mean_percent = np.mean(hist_percent)
                std_percent = np.std(hist_percent)
                threshold = mean_percent + std_percent

            for i in range(len(hist_percent)):
                if hist_percent[i] > threshold:
                    if current_sequence_sum == 0:
                        current_sequence_start = i
                    current_sequence_sum += hist_percent[i]
                    if current_sequence_sum > max_sequence_sum:
                        max_sequence_sum = current_sequence_sum
                        max_sequence_center = (bin_edges[current_sequence_start] + bin_edges[i + 1]) / 2
                else:
                    current_sequence_sum = 0

            # Verificar se existe uma faixa isolada com mais de 30% dos dados
            max_single_bin_center = 0
            max_single_bin_percent = 0
            for i in range(len(hist_percent)):
                if hist_percent[i] > 20:
                    max_single_bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                    max_single_bin_percent = hist_percent[i]

            if max_single_bin_percent > max_sequence_sum:
                eps_meters = max_single_bin_center
            else:
                eps_meters = max_sequence_center

            if eps_meters < 300:
                eps_meters = 300

        else:
            eps_meters = 10000

        lat_degrees, long_degrees = self.meters_to_degrees(eps_meters, df[self.lat_column].mean())
        return lat_degrees, long_degrees

    def dbscan_outliers(self, df):
        """
        Aplica o DBSCAN e classifica os pontos como outliers ou não.
        """
        cep_raizes = df[self.cep_raz_column].unique()

        if self.precision:
            df['dbscan_outlier'] = 1
        else:
            df['dbscan_outlier'] = 2
        if not self.dinamic:
            df['dbscan_outlier'] = 3

        df['Cluster'] = -1  # Inicializar a coluna de cluster

        for raiz in tqdm(cep_raizes):
            sub_df_raiz = df[df[self.cep_raz_column] == raiz]

            if sub_df_raiz[self.cep_column].nunique() > 1:
                for cep in sub_df_raiz[self.cep_column].unique():
                    sub_df_cep = sub_df_raiz[sub_df_raiz[self.cep_column] == cep]

                    if len(sub_df_cep) > 5:
                        latitude_degrees, longitude_degrees = self.calcular_eps(sub_df_cep)
                        df.loc[df[self.cep_column] == cep, 'Latitude_norm'] = df.loc[df[self.cep_column] == cep, self.lat_column] / latitude_degrees
                        df.loc[df[self.cep_column] == cep, 'Longitude_norm'] = df.loc[df[self.cep_column] == cep, self.long_column] / longitude_degrees

                        db = DBSCAN(eps=1, min_samples=1).fit(df.loc[df[self.cep_column] == cep, ['Latitude_norm', 'Longitude_norm']])
                        df.loc[df[self.cep_column] == cep, 'Cluster'] = db.labels_

                        # Identificar os maiores clusters
                        cluster_sizes = df[df[self.cep_column] == cep]['Cluster'].value_counts()

                        largest_cluster_label = cluster_sizes.idxmax()
                        largest_cluster = df[(df[self.cep_column] == cep) & (df['Cluster'] == largest_cluster_label)]

                        df.loc[largest_cluster.index, 'dbscan_outlier'] = 0

                    else:
                        # Caso o CEP não tenha mais de 5 pontos, aplica o DBSCAN na raiz do CEP
                        latitude_degrees, longitude_degrees = self.calcular_eps(sub_df_raiz)
                        df.loc[df[self.cep_raz_column] == raiz, 'Latitude_norm'] = df.loc[df[self.cep_raz_column] == raiz, self.lat_column] / latitude_degrees
                        df.loc[df[self.cep_raz_column] == raiz, 'Longitude_norm'] = df.loc[df[self.cep_raz_column] == raiz, self.long_column] / longitude_degrees

                        db = DBSCAN(eps=1, min_samples=1).fit(df.loc[df[self.cep_raz_column] == raiz, ['Latitude_norm', 'Longitude_norm']])
                        df.loc[df[self.cep_raz_column] == raiz, 'Cluster'] = db.labels_

                        # Identificar os maiores clusters
                        cluster_sizes = df[df[self.cep_raz_column] == raiz]['Cluster'].value_counts()

                        largest_cluster_label = cluster_sizes.idxmax()
                        largest_cluster = df[(df[self.cep_raz_column] == raiz) & (df['Cluster'] == largest_cluster_label)]

                        df.loc[largest_cluster.index, 'dbscan_outlier'] = 0

            else:
                # Caso a raiz tenha menos de 5 pontos, aplica o DBSCAN na raiz do CEP
                if len(sub_df_raiz) > 5:
                    latitude_degrees, longitude_degrees = self.calcular_eps(sub_df_raiz)
                    df.loc[df[self.cep_raz_column] == raiz, 'Latitude_norm'] = df.loc[df[self.cep_raz_column] == raiz, self.lat_column] / latitude_degrees
                    df.loc[df[self.cep_raz_column] == raiz, 'Longitude_norm'] = df.loc[df[self.cep_raz_column] == raiz, self.long_column] / longitude_degrees

                    db = DBSCAN(eps=1, min_samples=1).fit(df.loc[df[self.cep_raz_column] == raiz, ['Latitude_norm', 'Longitude_norm']])
                    df.loc[df[self.cep_raz_column] == raiz, 'Cluster'] = db.labels_

                    # Identificar os maiores clusters
                    cluster_sizes = df[df[self.cep_raz_column] == raiz]['Cluster'].value_counts()

                    largest_cluster_label = cluster_sizes.idxmax()
                    largest_cluster = df[(df[self.cep_raz_column] == raiz) & (df['Cluster'] == largest_cluster_label)]

                    df.loc[largest_cluster.index, 'dbscan_outlier'] = 0

        return df

    def processar_outliers(self, df, alta_precisao=True, dinamic=True):
        """
        Processa o dataframe utilizando DBSCAN para detecção de outliers.
        """
        self.precision = alta_precisao
        self.dinamic = dinamic

        df[self.lat_column] = df[self.lat_column].astype('str').str.replace('[^0-9.-]', '', regex=True)
        df[self.long_column] = df[self.long_column].astype('str').str.replace('[^0-9.-]', '', regex=True)

        df['lat_long'] = df[self.lat_column] + ' ' + df[self.long_column]

        df[self.lat_column] = df[self.lat_column].astype('float64')
        df[self.long_column] = df[self.long_column].astype('float64')

        df = pd.merge(
            df,
            df.groupby('lat_long')[self.cep_column].nunique().reset_index().rename(columns={self.cep_column: 'duplicated'}),
            how='inner',
            on='lat_long'
        )

        df['ponto_unico'] = df['duplicated'] == 1

        df = df[df['ponto_unico']].copy()

        df = pd.merge(
            df,
            df.groupby(self.cep_raz_column).agg(repetitions=(self.cep_raz_column, 'count')).reset_index(),
            how='inner',
            on=self.cep_raz_column
        ).rename(columns={'repetitions': 'QtdPorCepRaz'})

        df = df[df['QtdPorCepRaz'] >= 4]

        timers = [datetime.now()]

        df = self.dbscan_outliers(df)

        timers.append(datetime.now())
        print(f'DBSCAN: {timers[-1] - timers[-2]}')

        return df


if __name__ == '__main__':
    # Carregando o arquivo GeoJSON
    gdf = gpd.read_file('Dados/source.geojson')
    gdf = gdf.replace('', None, regex=False)
    gdf['postcode'] = gdf['postcode'].str.replace('-', '')
    gdf = gdf.dropna(axis=1, how='all')
    gdf = gdf.dropna(subset=['postcode'])
    gdf = gdf[gdf['postcode'].str.isnumeric()]
    gdf['Latitude'] = gdf.geometry.y
    gdf['Longitude'] = gdf.geometry.x
    gdf['postcodeRaz'] = gdf['postcode'].str[:5]

    # Aplicando o processamento
    tratamento = TratamentoGeoOutliers('postcodeRaz', 'postcode', 'Latitude', 'Longitude')
    cleaned_gdf = tratamento.processar_outliers(gdf)

    cleaned_gdf.to_excel('Dados/dados_DBSCAN.xlsx', index=False)
