import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv # pip install python-dotenv
import os
import psycopg2
import seaborn as sns
import io
import base64
import tensorflow as tf
from keras.models import load_model
import joblib


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Datos
df = pd.read_csv("C:\\Users\\LORE\\Downloads\\Soporte_4\\datos_final.csv", sep=';')

# cargar archivo de disco
model = tf.keras.models.load_model("C:\\Users\\LORE\\Downloads\\Soporte_4\\Modelito_Triunfando.h5")

#Promedio puntaje por clasificación
df['clasificacion'] = df['punt_global'].apply(lambda x: 1 if x > 320 else 0)
materias = ['punt_ingles', 'punt_matematicas', 'punt_c_naturales', 'punt_sociales_ciudadanas', 'punt_lectura_critica']
promedios_clasificacion = df.groupby('clasificacion')[materias].mean().transpose()

def promedio_punt_clasificacion():
    fig = go.Figure()
    for materias in promedios_clasificacion.columns:
        fig.add_trace(go.Scatter(x=promedios_clasificacion.index, y=promedios_clasificacion[materias], mode ='lines+markers', name=materias))
    fig.update_layout(title='Promedio de Puntajes por Clasificación',
                      xaxis_title='Materia',
                      yaxis_title='Promedio de Puntaje',
                      xaxis_tickangle=-45)
    return fig    
    
#ESTRATO - Puntaje
def distribucion_estrato():
    fig = px.box(df, x='fami_estratovivienda', y='punt_global', color='clasificacion',
                 title='Distribución de puntajes por estrato',
                 labels={'fami_estratovivienda': 'Estrato', 'punt_global': 'Puntaje Global'},
                 category_orders={'fami_estratovivienda': ['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6']})
    fig.update_layout(yaxis_title='Puntaje Global')
    return fig

#COMPUTADOR - Puntaje
def distribucion_computador():
    fig = px.box(df, x='fami_tienecomputador', y='punt_global', color='clasificacion',
                 title='Distribución de puntajes por tenencia de computador',
                 labels={'fami_tienecomputador': 'Computador en casa', 'punt_global': 'Puntaje Global'})
    fig.update_layout(yaxis_title='Puntaje Global')
    return fig

#UBICACIÓN
def distribucion_ubicacion():
    fig = px.histogram(df, x='cole_area_ubicacion', color='clasificacion', barmode='group',
                       title='Distribución de clasificación por ubicacion',
                       labels={'cole_area_ubicacion': 'Ubicación', 'count': 'Frecuencia'})
    fig.update_layout(yaxis_title='Frecuencia')
    return fig 

# Layout de la aplicación Dash
app.layout = html.Div([
    html.Div([        
        html.H1("Predicción y Análisis de Puntaje del Icfes en el Departamento del Valle del Cauca", style={'font-family': 'Calibri Light'}),       
    ], style={'posotion':'relative'}),
    
        html.Div([
        
        # Familia tiene computador
        html.Div([
            html.Label('Tiene computador'),
            dcc.Dropdown(
                id='fami-tienecomputador-dropdown',
                options=[
                    {'label': 'No', 'value': 'No'},
                    {'label': 'Si', 'value': 'Si'}
                ],
                value=1
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),
        # BILINGÜE 
        html.Div([
            html.Label('Colegio bilingüe'),
            dcc.Dropdown(
                id='cole-bilingue-dropdown',
                options=[
                    {'label': 'No', 'value': 'N'},
                    {'label': 'Si', 'value': 'S'}
                ],
                value=1
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        # UBICACIÓN
        html.Div([    
            html.Label('Área de ubicación'),
            dcc.Dropdown(
                id='cole-area-ubicacion-dropdown',
                options=[
                    {'label': 'Urbano', 'value': 'URBANO'},
                    {'label': 'Rural', 'value': 'RURAL'}
                ],
                value=1
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        #GENERO DEL ESTUDIANTE
        html.Div([    
            html.Label('Género del estudiante'),
            dcc.Dropdown(
                id='estu-genero-dropdown',
                options=[
                    {'label': 'Femenino', 'value': 'F'},
                    {'label': 'Masculino', 'value': 'M'}
                ],
                value=1
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        #ESTRATO
        html.Label('Estrato'),
        dcc.Dropdown(
            id='fami-estratovivienda-dropdown',
            options=[
                {'label': '1', 'value': 'Estrato 1'},
                {'label': '2', 'value': 'Estrato 2'},
                {'label': '3', 'value': 'Estrato 3'},
                {'label': '4', 'value': 'Estrato 4'},
                {'label': '5', 'value': 'Estrato 5'},
                {'label': '6', 'value': 'Estrato 6'}
            ],
            value=1
        ),
        #NATURALEZA DEL COLEGIO)
        html.Label('Naturaleza del colegio'),
        dcc.Dropdown(
            id='cole-naturaleza-dropdown',
            options=[
                {'label': 'Oficial', 'value': 'OFICIAL'},
                {'label': 'No Oficial', 'value': 'NO OFICIAL'}
            ],
            value=1
        ),
        #INTERNET
        html.Label('Tiene internet'),
        dcc.Dropdown(
            id='fami-tieneinternet-dropdown',
            options=[
                {'label': 'No', 'value': 'No'},
                {'label': 'Si', 'value': 'Si'}
            ],
            value=1
        ),
        #EDUCACION DEL PADRE
        html.Label('Educación del padre'),
        dcc.Dropdown(
            id='fami-educacionpadre-dropdown',
            options=[
                {'label': 'Educación profesional completa', 'value': 'Educación profesional completa'},
                {'label': 'Educación profesional incompleta', 'value': 'Educación profesional incompleta'},
                {'label': 'Ninguno', 'value': 'Ninguno'},
                {'label': 'No sabe', 'value': 'No sabe'},
                {'label': 'No Aplica', 'value': 'No Aplica'},
                {'label': 'Postgrado', 'value': 'Postgrado'},
                {'label': 'Primaria completa', 'value': 'Primaria completa'},
                {'label': 'Primaria incompleta', 'value': 'Primaria incompleta'},
                {'label': 'Secuendaria completa', 'value': 'Secundaria (Bachillerato) completa'},
                {'label': 'Secundaria incompleta', 'value': 'Secundaria (Bachillerato) incompleta'},
                {'label': 'Técnica o tecnológica completa', 'value': 'Técnica o tecnológica completa'},
                {'label': 'Técnica o tecnológica incompleta', 'value': 'Técnica o tecnológica incompleta'},
                
            ],
            value=1
        ),
        #EDUCACION MADRE
        html.Label('Educación de la madre'),
        dcc.Dropdown(
            id='fami-educacionmadre-dropdown',
            options=[
                {'label': 'Educación profesional completa', 'value': 'Educación profesional completa'},
                {'label': 'Educación profesional incompleta', 'value': 'Educación profesional incompleta'},
                {'label': 'Ninguno', 'value': 'Ninguno'},
                {'label': 'No sabe', 'value': 'No sabe'},
                {'label': 'No Aplica', 'value': 'No Aplica'},
                {'label': 'Postgrado', 'value': 'Postgrado'},
                {'label': 'Primaria completa', 'value': 'Primaria completa'},
                {'label': 'Primaria incompleta', 'value': 'Primaria incompleta'},
                {'label': 'Secuendaria completa', 'value': 'Secundaria (Bachillerato) completa'},
                {'label': 'Secundaria incompleta', 'value': 'Secundaria (Bachillerato) incompleta'},
                {'label': 'Técnica o tecnológica completa', 'value': 'Técnica o tecnológica completa'},
                {'label': 'Técnica o tecnológica incompleta', 'value': 'Técnica o tecnológica incompleta'},
                
            ],
            value=1
        ),       
        
        html.Button('Predecir', id='button', style={'display': 'block', 'margin': 'auto'}),
        html.Div(id='output-prediction', style = {'fontsize':'24px','textAlign': 'center', 'font-weight': 'bold'})
    ]),

    #Graficos
    
    html.H2("Gráficos de interés", style={'font-family': 'Calibri Light'}),
    html.Div([
        html.Div(dcc.Graph(figure=promedio_punt_clasificacion()), style={'display': 'inline-block', 'width': '50%'}),
        html.Div(dcc.Graph(figure=distribucion_estrato()), style={'display': 'inline-block', 'width': '50%'})
    ]),
    html.Div([
        html.Div(dcc.Graph(figure=distribucion_computador()), style={'display': 'inline-block', 'width': '50%'}),
        html.Div(dcc.Graph(figure=distribucion_ubicacion()), style={'display': 'inline-block', 'width': '50%'})
    ])

], style={'font-family': 'Calibri Light'})


# Cargar el pipeline desde el archivo 'pipeline.pkl'
pipeline = joblib.load("C:\\Users\\LORE\\Downloads\\Soporte_4\\pipeline.pkl")

# Callback para realizar la predicción
@app.callback(
    Output('output-prediction', 'children'),
    [Input('button', 'n_clicks')],
    [dash.dependencies.State('fami-tienecomputador-dropdown', 'value'),
     dash.dependencies.State('cole-bilingue-dropdown', 'value'),
     dash.dependencies.State('cole-area-ubicacion-dropdown', 'value'),
     dash.dependencies.State('estu-genero-dropdown', 'value'),
     dash.dependencies.State('fami-estratovivienda-dropdown', 'value'),
     dash.dependencies.State('cole-naturaleza-dropdown', 'value'),
     dash.dependencies.State('fami-tieneinternet-dropdown', 'value'),
     dash.dependencies.State('fami-educacionpadre-dropdown', 'value'),
     dash.dependencies.State('fami-educacionmadre-dropdown', 'value')]
)

def update_prediction(n_clicks, fami_tienecomputador,cole_bilingue, cole_area_ubicacion, estu_genero, fami_estratovivienda, cole_naturaleza, fami_tieneinternet, 
            fami_educacionpadre, fami_educacionmadre):
    # Preprocess user input data
    if n_clicks is not None:
        user_input = {
            'fami_tienecomputador': [fami_tienecomputador],
            'cole_bilingue': [cole_bilingue],
            'cole_area_ubicacion' : [cole_area_ubicacion],
            'estu_genero': [estu_genero],
            'fami_estratovivienda' : [fami_estratovivienda],
            'cole_naturaleza' : [cole_naturaleza],
            'fami_tieneinternet' : [fami_tieneinternet],
            'fami_educacionpadre': [fami_educacionpadre],
            'fami_educacionmadre': [fami_educacionmadre],
        }
            
        print(user_input)

        pipeline = joblib.load("C:\\Users\\LORE\\Downloads\\Soporte_4\\pipeline.pkl")

        user_input = pd.DataFrame(user_input)

        model_input = pipeline.transform(user_input)
   
        # Realizar la predicción
        prediction = model.predict(model_input)

        
        # Umbral para la predicción
        umbral = 0.7
        
        # Obtener la probabilidad
        
        predicted_probability = prediction[0, 0]

        binary_prediction = 1 if predicted_probability > umbral else 0

        # Procesar la salida de la predicción según sea necesario
        if binary_prediction == 1:
            result = "Puntaje mayor a 320"
        else:
            result = "Puntaje menor a 320"

        return f'Predicción: {result} con una probabilidad de {predicted_probability:.2f}'



if __name__ == '__main__':
    app.run_server(debug=True)