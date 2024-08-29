import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from elasticsearch import Elasticsearch

# Conectando ao Elasticsearch local
es_client = Elasticsearch("http://localhost:9200")


# Função para consultar o Elasticsearch
def query_elasticsearch(cep=None, numero=None):
    if cep:
        if numero == "-1":
            # Buscar todos os pontos com tipo Porta para o CEP ou raiz de CEP fornecido
            if len(cep) == 8:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"cep": cep}},
                                {"match": {"tipo_ponto": "Porta"}}
                            ]
                        }
                    }
                }
            elif len(cep) == 5:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"raiz_cep": cep}},
                                {"match": {"tipo_ponto": "Porta"}}
                            ]
                        }
                    }
                }
        elif numero:
            # Buscar pelo CEP/raiz e número específico com tipo Porta
            if len(cep) == 8:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"cep": cep}},
                                {"match": {"numero": numero}},
                                {"match": {"tipo_ponto": "Porta"}}
                            ]
                        }
                    }
                }
            elif len(cep) == 5:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"raiz_cep": cep}},
                                {"match": {"numero": numero}},
                                {"match": {"tipo_ponto": "Porta"}}
                            ]
                        }
                    }
                }
        else:
            # Buscar apenas pelo CEP/raiz com tipo Mediana
            if len(cep) == 8:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"cep": cep}},
                                {"match": {"tipo_ponto": "Mediana"}}
                            ]
                        }
                    }
                }
            elif len(cep) == 5:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"raiz_cep": cep}},
                                {"match": {"tipo_ponto": "Mediana"}}
                            ]
                        }
                    }
                }
    else:
        query = {"query": {"match_all": {}}}

    res = es_client.search(index="ceps", body=query, size=10000)
    return res['hits']['hits']


# Função para criar o mapa
def create_map(data):
    # Extrair coordenadas dos pontos
    coords = [(item['_source']['coordenada']['lat'], item['_source']['coordenada']['lon']) for item in data]

    # Calcular o centro do mapa
    avg_lat = sum([coord[0] for coord in coords]) / len(coords)
    avg_lon = sum([coord[1] for coord in coords]) / len(coords)

    # Criar o mapa centralizado nos pontos
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles='OpenStreetMap')

    for item in data:
        source = item['_source']
        legenda = f"CEP: {source.get('cep', '')}, Tipo: {source.get('tipo_ponto', '')}"
        if 'numero' in source:
            legenda += f", Número: {source.get('numero', '')}"

        # Definindo a cor do marcador com base em dbscan_outlier
        color = 'red' if source.get('dbscan_outlier') == 1 else 'green'

        folium.Marker(
            location=[source['coordenada']['lat'], source['coordenada']['lon']],
            popup=legenda,
            icon=folium.Icon(color=color)
        ).add_to(m)

    return m


def main():
    st.title("Consulta de Pontos por CEP/Raiz ou Número")

    cep = st.text_input("Digite o CEP ou Raiz do CEP:")
    numero = st.text_input("Digite o Número (use -1 para todos os pontos):")

    if st.button("Pesquisar"):
        if numero and not cep:
            st.warning("Por favor, preencha o CEP/Raiz junto com o Número.")
        else:
            data = query_elasticsearch(cep=cep, numero=numero)
            if data:
                st.subheader("Mapa de Pontos")
                map = create_map(data)
                folium_static(map)
            else:
                st.warning("Nenhum resultado encontrado para os critérios de pesquisa.")


if __name__ == "__main__":
    main()
