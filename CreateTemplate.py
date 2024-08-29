from elasticsearch import Elasticsearch


def create_elasticsearch_index(es_client, index_name):
    mapping = {
        "mappings": {
            "properties": {
                "cep": {
                    "type": "keyword"
                },
                "raiz_cep": {
                    "type": "keyword"
                },
                "numero": {
                    "type": "keyword"
                },
                "coordenada": {
                    "type": "geo_point"
                },
                "tipo_ponto": {
                    "type": "keyword"
                },
                "dbscan_outlier": {
                    "type": "keyword"
                }
            }
        }
    }

    # Criar o índice no Elasticsearch com o mapping definido
    es_client.indices.create(index=index_name, body=mapping, ignore=400)

# Exemplo de uso:
es_client = Elasticsearch("http://localhost:9200")
create_elasticsearch_index(es_client, "ceps")
