import pandas as pd
import geopandas as gpd
import folium
from folium import *
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import folium_static

# Função para carregar o Excel e processar os dados
def load_data(file):
    df = pd.read_excel(file)
    df['postcode'] = df['postcodeRaz'].astype(str).str[:5]  # Considerando que as raízes têm 5 dígitos
    return df


# Função para filtrar o DataFrame com base nas raízes de CEP selecionadas
def filter_data(df, selected_roots):
    if selected_roots:
        df = df[df['postcode'].isin(selected_roots)]
    return df


# Função para criar o mapa
def create_map(df):
    # Centro do mapa: Brasília
    m = folium.Map(location=[-15.7941, -47.8825], zoom_start=12, tiles='OpenStreetMap')

    # Cluster para os pontos
    marker_cluster = MarkerCluster().add_to(m)

    # Adicionar os pontos ao mapa
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"CEP: {row['postcodeRaz']}",
        ).add_to(marker_cluster)

    return m


# Função principal para o Streamlit
def main():
    st.title("Mapa de Pontos por CEP")

    # Upload do arquivo Excel
    uploaded_file = st.file_uploader("Dados/mediana_cep.xlsx", type="xlsx")

    if uploaded_file:
        df = load_data(uploaded_file)
        postcodes = df['postcode'].unique().tolist()

        # Sidebar com os filtros de raízes de CEP
        selected_roots = st.sidebar.multiselect("Selecione as raízes de CEP para filtrar", postcodes)

        # Filtrando os dados com base nas raízes de CEP selecionadas
        filtered_df = filter_data(df, selected_roots)

        # Gerando o mapa
        st.subheader("Mapa de Pontos")
        map = create_map(filtered_df)

        # Renderizando o mapa no Streamlit
        folium_static(map)


if __name__ == "__main__":
    main()
