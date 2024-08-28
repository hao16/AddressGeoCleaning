import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from skopt import BayesSearchCV
from skopt.space import Real, Integer


# Função para ajustar o GMM e encontrar os melhores parâmetros
def fit_gmm_with_bayes_optimization(data):
    # Espaços de busca para os parâmetros
    param_space = {
        'n_components': Integer(1, 5),
        'covariance_type': ['full', 'tied', 'diag', 'spherical']
    }

    # GMM com Bayes Search CV
    gmm = GaussianMixture()
    bayes_search = BayesSearchCV(
        estimator=gmm,
        search_spaces=param_space,
        n_iter=32,
        cv=3,
        n_jobs=-1
    )

    bayes_search.fit(data)
    best_gmm = bayes_search.best_estimator_

    return best_gmm


# Função para aplicar DBSCAN e GMM dinamicamente
def process_cep(gdf, cep_col='postcode', eps_range=(0.001, 0.01), min_samples=5):
    unique_ceps = gdf[cep_col].unique()
    cleaned_data = []

    for cep in unique_ceps:
        cep_data = gdf[gdf[cep_col] == cep].geometry.apply(lambda geom: [geom.x, geom.y]).tolist()

        # DBSCAN para detectar clusters iniciais
        best_eps = None
        best_labels = None
        best_n_clusters = 0

        for eps in np.linspace(eps_range[0], eps_range[1], 10):
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(cep_data)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > best_n_clusters:
                best_eps = eps
                best_labels = labels
                best_n_clusters = n_clusters

        # Refine os clusters usando GMM
        for label in set(best_labels):
            if label == -1:
                continue  # Ignorar ruído identificado pelo DBSCAN

            cluster_data = [coord for coord, l in zip(cep_data, best_labels) if l == label]

            if len(cluster_data) > 1:  # Evitar erro no GMM com um único ponto
                gmm = fit_gmm_with_bayes_optimization(cluster_data)
                gmm_labels = gmm.predict(cluster_data)

                # Cálculo da mediana para cada sub-cluster
                for gmm_label in np.unique(gmm_labels):
                    gmm_cluster_data = [coord for coord, gl in zip(cluster_data, gmm_labels) if gl == gmm_label]
                    median_lat = np.median([coord[1] for coord in gmm_cluster_data])
                    median_lon = np.median([coord[0] for coord in gmm_cluster_data])
                    cleaned_data.append({
                        'postcode': cep,
                        'geometry': f'POINT ({median_lon} {median_lat})',
                        'cluster': f'{label}-{gmm_label}'
                    })

    return gpd.GeoDataFrame(cleaned_data, geometry=gpd.points_from_xy(
        [data['geometry'].x for data in cleaned_data],
        [data['geometry'].y for data in cleaned_data]
    ))


# Carregando o arquivo GeoJSON
gdf = gpd.read_file('Dados/source.geojson')
gdf = gdf.replace('', None, regex=False)
gdf['postcode'] = gdf['postcode'].str.replace('-', '')
gdf = gdf.dropna(axis=1, how='all')
gdf = gdf.dropna(subset=['postcode'])

# Aplicando o processamento
cleaned_gdf = process_cep(gdf)

# Salvando os dados limpos de volta para GeoJSON
cleaned_gdf.to_file('Dados/dados_ceps_limpos.geojson', driver='GeoJSON')

# Exibindo os dados processados
print(cleaned_gdf)
