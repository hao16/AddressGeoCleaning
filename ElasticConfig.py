import pandas as pd
import re
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError

def filter_and_clean_single_digit_sequence(df, column_name):
    # Define uma função para verificar e extrair exatamente uma sequência de dígitos
    def extract_single_digit_sequence(value):
        # Use regex para encontrar todas as sequências de dígitos
        digit_sequences = re.findall(r'\d+', str(value))
        # Verifica se há exatamente uma sequência de dígitos
        if len(digit_sequences) == 1:
            return digit_sequences[0]  # Retorna a sequência de dígitos encontrada
        else:
            return None  # Se não houver exatamente uma sequência de dígitos, retorna None

    # Aplica a função de extração ao DataFrame e remove as linhas onde o resultado é None
    df[column_name] = df[column_name].apply(extract_single_digit_sequence)
    filtered_df = df.dropna(subset=[column_name])

    return filtered_df


mediana = pd.read_excel('Dados/mediana_cep.xlsx')
porta = pd.read_excel('Dados/dados_DBSCAN.xlsx')


mediana['TipoPonto'] = 'Mediana'
porta['TipoPonto'] = 'Porta'

porta = filter_and_clean_single_digit_sequence(porta, 'number')
# porta = porta[porta['dbscan_outlier'] == 0]

porta = porta[['number', 'postcode', 'postcodeRaz', 'Latitude', 'Longitude', 'TipoPonto', 'dbscan_outlier']].drop_duplicates()


dado_final = pd.concat([mediana, porta], ignore_index=True)

dado_final['coordenada'] = dado_final.apply(lambda row: {'lat': row['Latitude'], 'lon': row['Longitude']}, axis=1)

json_data = dado_final[['number', 'postcode', 'postcodeRaz', 'dbscan_outlier', 'coordenada', 'TipoPonto']].rename(columns={
    'number': 'numero',
    'postcode': 'cep',
    'postcodeRaz': 'raiz_cep',
    'TipoPonto': 'tipo_ponto'
}).fillna("").to_dict(orient='records')

def bulk_import_to_elasticsearch(es_client, index_name, json_data):
    actions = [
        {
            "_index": index_name,
            "_source": data
        }
        for data in json_data
    ]
    helpers.bulk(es_client, actions)

es_client = Elasticsearch("http://localhost:9200")

try:
    bulk_import_to_elasticsearch(es_client, "ceps", json_data)
except BulkIndexError as e:
    print(f"Erro ao indexar: {e.errors}")
