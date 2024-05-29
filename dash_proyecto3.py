import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv # pip install python-dotenv
import os
import psycopg2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
env_path="C://Users//angie//OneDrive//Desktop//Octavo semestre//Analitica computacional//Proyecto 2 personal//env//app.env"
#load env 
load_dotenv(dotenv_path=env_path)
#extract env variables
USER=os.getenv('USER')
PASSWORD=os.getenv('PASSWORD')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')
engine = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)


# cargar archivo de disco
model = tf.keras.models.load_model("models/modelo.h5")

#Hasta aqui todo esta bien
#GRAFICOS
import pandas.io.sql as sqlio
cursor = engine.cursor()
query = """
SELECT * 
FROM proy1;"""
# Calcular el promedio de puntajes para cada materia y clasificación

# Configurar la visualización
plt.figure(figsize=(12, 6))

# Graficar las líneas para cada materia
def promedio_punt_clasificació(df):
    promedios_clasificacion = df.groupby('clasificacion')[materias].mean().transpose()
    plt.figure(figsize=(12,6))
    for materia in promedios_clasificacion.columns:
        plt.plot(promedios_clasificacion.index, promedios_clasificacion[materia], marker='o', label=materia)
  
    # Configurar título y etiquetas de los ejes
    plt.title('Promedio de Puntajes por Clasificación')
    plt.xlabel('Materia')
    plt.ylabel('Promedio de Puntaje')
    plt.xticks(rotation=45)  
    plt.legend()
    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

#ESTRATO - Puntaje
plt.figure(figsize=(10, 6))
# Gráfico de barras para la variable categórica 'estu_genero' con valores encima de las barras
ax = sns.countplot(x='fami_estratovivienda', hue='clasificacion', data=df, palette='Set2')
# Configurar título y etiquetas de los ejes
plt.title('Distribución de clasificación por estrato',fontsize=16, fontweight='bold')
plt.xlabel('Estrato',fontsize=14)
plt.ylabel('Frecuencia',fontsize=14)
# Lista para almacenar los valores únicos de la clasificación
valores_clasificacion = []
# Añadir etiquetas con las frecuencias encima de cada barra
for p in ax.patches:
      if p.get_height() != 0: 
        valores_clasificacion.append(p.get_height())
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2.,
                                          p.get_height()), ha='center', va='bottom', xytext=(0, 5), 
                            textcoords='offset points',
                             color='black', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
# Mostrar el gráfico
plt.show()

#COMPUTADOR - Puntaje
plt.figure(figsize=(10, 6))
# Gráfico de barras para la variable categórica 'estu_genero' con valores encima de las barras
ax = sns.countplot(x='fami_tienecomputador', hue='clasificacion', data=df, palette='Set2')
# Configurar título y etiquetas de los ejes
plt.title('Distribución de clasificación por tenencia de computador',fontsize=16, fontweight='bold')
plt.xlabel('Computador en casa',fontsize=14)
plt.ylabel('Frecuencia',fontsize=14)
# Lista para almacenar los valores únicos de la clasificación
valores_clasificacion = []
# Añadir etiquetas con las frecuencias encima de cada barra
for p in ax.patches:
      if p.get_height() != 0: 
        valores_clasificacion.append(p.get_height())
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2.,
                                          p.get_height()), ha='center', va='bottom', xytext=(0, 5), 
                            textcoords='offset points',
                             color='black', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
# Mostrar el gráfico
plt.show()
#Dash pasado
df = sqlio.read_sql_query(query, engine)
fig_edades = px.histogram(df, x="x5", title="Distribución de Edades")
fig_edades.update_xaxes(title_text="Edades")

gender_labels = {1: 'Masculino', 2: 'Femenino'}
education_labels = {1: 'Posgrado', 2: 'Universidad', 3: 'Bachillerato', 4: 'Otro'}
marital_status_labels = {1: 'Casado', 2: 'Soltero', 3: 'Otro'}
categorical_columns = ['x2', 'x3', 'x4']

# Obtener las frecuencias de las variables categóricas
gender_freq = df['x2'].map(gender_labels).value_counts()
education_freq = df['x3'].map(education_labels).value_counts()
marital_status_freq = df['x4'].map(marital_status_labels).value_counts()

# Crear gráficos de barras
gender_fig = go.Figure(data=[go.Bar(x=gender_freq.index, y=gender_freq.values, marker=dict(color='lightgreen'))])
gender_fig.update_layout(title=' Género')

education_fig = go.Figure(data=[go.Bar(x=education_freq.index, y=education_freq.values, marker=dict(color='salmon'))])
education_fig.update_layout(title='Nivel de Educación')

marital_status_fig = go.Figure(data=[go.Bar(x=marital_status_freq.index, y=marital_status_freq.values, marker=dict(color='skyblue'))])
marital_status_fig.update_layout(title='Estado Civil')

# Layout de la aplicación Dash
app.layout = html.Div([
    html.H1("Predicción de modelo", style={'font-family': 'Calibri Light'}),
    html.Div([
        
        # Familia tiene computador
        html.Div([
            html.Label('Tiene computador'),
            dcc.Dropdown(
                id='fami_tienecomputador-dropdown',
                options=[
                    {'label': 'No', 'value': 'No'},
                    {'label': 'Si', 'value': 'Si'}
                ],
                value='No'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),
        # BILINGÜE 
        html.Div([
            html.Label('Colegio bilingüe'),
            dcc.Dropdown(
                id='cole_bilingue-dropdown',
                options=[
                    {'label': 'No', 'value': 'N'},
                    {'label': 'Si', 'value': 'S'}
                ],
                value='N'
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        # UBICACIÓN
        html.Div([    
            html.Label('Área de ubicación'),
            dcc.Dropdown(
                id='cole_area_ubicacion-dropdown',
                options=[
                    {'label': 'Urbano', 'value': 'URBANO'},
                    {'label': 'Rural', 'value': 'RURAL'}
                ],
                value='URBANO'
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        #GENERO DEL ESTUDIANTE
        html.Div([    
            html.Label('Género del estudiante'),
            dcc.Dropdown(
                id='estu_genero-dropdown',
                options=[
                    {'label': 'Femenino', 'value': 'F'},
                    {'label': 'Masculino', 'value': 'M'}
                ],
                value='F'
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        #ESTRATO
        html.Label('Estrato'),
        dcc.Dropdown(
            id='fami_estratovivienda-dropdown',
            options=[
                {'label': 'Estrato 1', 'value':'Estrato 1' },
                {'label': 'Estrato 2', 'value': 'Estrato 2'},
                {'label': 'Estrato 3', 'value': 'Estrato 3'},
                {'label': 'Estrato 4', 'value': 'Estrato 4'},
                {'label': 'Estrato 5', 'value': 'Estrato 5'},
                {'label': 'Estrato 6', 'value': 'Estrato 6'}
            ],
            value='Estrato 1'
        ),
        #AGOSTO (X7)
        html.Label('Naturaleza del colegio'),
        dcc.Dropdown(
            id='cole_naturaleza-dropdown',
            options=[
                {'label': 'Oficial', 'value': 'OFICIAL'},
                {'label': 'No Oficial', 'value': 'NO OFICIAL'}
            ],
            value='OFICIAL'
        ),
        #INTERNET
        html.Label('Tiene internet'),
        dcc.Dropdown(
            id='fami_tieneinternet-dropdown',
            options=[
                {'label': 'No', 'value': 'No'},
                {'label': 'Si', 'value': 'Si'}
            ],
            value='No'
        ),
        #EDUCACION DEL PADRE
        html.Label('Educación del padre'),
        dcc.Dropdown(
            id='fami_educacionpadre-dropdown',
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
            value='Educación profesional completa'
        ),
        #EDUCACION MADRE
        html.Label('Educación de la madre'),
        dcc.Dropdown(
            id='fami_educacionmadre-dropdown',
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
            value='Educación profesional completa'
        ),

        # VARIABLES CONTINUAS PUNTOS
        
        #INGLES
        html.Div([
            html.Label('Puntos de inglés'),
            dcc.Input(id='punt_ingles-input', type='number', min=0, max=100, step=1, value=0),
        ], style = {'width': '20%', 'display': 'inline-block', 'margin': 'auto'}),
        
        #MATEMATICAS
        html.Div([
            html.Label('Puntos de matemáticas'),
            dcc.Input(id='punt_matematicas-input', type='number', min=0, max=100, step=1, value=0),
        ], style = {'width': '20%', 'display': 'inline-block', 'margin': 'auto'}),
        
        #SOCIALES Y CIUDADANAS 
        html.Div([
            html.Label('Puntos de sociales y ciudadanas'),
            dcc.Input(id='punt_sociales_ciudadanas-input', type='number', min=0, max=100, step=1, value=0),
        ], style = {'width': '20%', 'display': 'inline-block', 'margin': 'auto'}),
        
        #CIENCIAS NATURALES
        html.Div([
            html.Label('Puntos de ciencias naturales'),
            dcc.Input(id='punt_c_naturales-input', type='number', min=0, max=100, step=1, value=0),
        ], style = {'width': '20%', 'display': 'inline-block', 'margin': 'auto'}),
        
        #LECTURA CRITICA
        html.Div([
            html.Label('Puntos de lectura crítica'),
            dcc.Input(id='punt_lectura_critica-input', type='number', min=0, max=100, step=1, value=0),
        ], style = {'width': '20%', 'display': 'inline-block', 'margin': 'auto'}),
        
        
        html.Button('Predecir', id='button', style={'display': 'block', 'margin': 'auto'}),
        html.Div(id='output-prediction', style = {'fontsize':'24px','textAlign': 'center', 'font-weight': 'bold'})
    ]),

    #Graficos
    
    html.H2("Gráficos de interés", style={'font-family': 'Calibri Light'}),
    html.Div([
        html.Div(dcc.Graph(figure=fig_edades), style={'display': 'inline-block', 'width': '25%'}),
        html.Div(dcc.Graph(figure=gender_fig), style={'display': 'inline-block', 'width': '25%'}),
        html.Div(dcc.Graph(figure=education_fig), style={'display': 'inline-block', 'width': '25%'}),
        html.Div(dcc.Graph(figure=marital_status_fig), style={'display': 'inline-block', 'width': '25%'})
    ])

], style={'font-family': 'Calibri Light'})


# Callback para realizar la predicción
@app.callback(
    Output('output-prediction', 'children'),
    [Input('button', 'n_clicks')],
    [dash.dependencies.State('fami_tienecomputador-dropdown', 'value'),
     dash.dependencies.State('cole_bilingue-dropdown', 'value'),
     dash.dependencies.State('cole_area_ubicacion-dropdown', 'value'),
     dash.dependencies.State('estu_genero-dropdown', 'value'),
     dash.dependencies.State('fami_estratovivienda-dropdown', 'value'),
     dash.dependencies.State('cole_naturaleza-dropdown', 'value'),
     dash.dependencies.State('fami_tieneinternet-dropdown', 'value'),
     dash.dependencies.State('fami_educacionpadre-dropdown', 'value'),
     dash.dependencies.State('fami_educacionmadre-dropdown', 'value'),
     dash.dependencies.State('punt_ingles-input', 'value'),
     dash.dependencies.State('punt_matematicas-input', 'value'),
     dash.dependencies.State('punt_sociales_ciudadanas-input', 'value'),
     dash.dependencies.State('punt_c_naturales-input', 'value'),
     dash.dependencies.State('punt_lectura_critica-input', 'value')]
    
)

def predict(n_clicks, fami_tienecomputador, cole_bilingue, cole_area_ubicacion, estu_genero, fami_estratovivienda, cole_naturaleza, fami_tieneinternet, 
            fami_educacionpadre, fami_educacionmadre, punt_ingles, punt_matematicas, punt_sociales_ciudadanas, punt_c_naturales, punt_lectura_critica):
    if n_clicks is None:
        return ''
    
    # Combinar las variables categóricas y continuas en un solo vector de entrada
    input_values = [fami_tienecomputador, cole_bilingue, cole_area_ubicacion, estu_genero, fami_estratovivienda, cole_naturaleza, fami_tieneinternet, 
            fami_educacionpadre, fami_educacionmadre, punt_ingles, punt_matematicas, punt_sociales_ciudadanas, punt_c_naturales, punt_lectura_critica]
    
    # Preprocesar los valores ingresados (por ejemplo, escalarlos)
    input_values = np.array([input_values])  # Formato de entrada esperado por el modelo
    
    
    print("Datos de entrada:", input_values)
    print("Forma de los datos de entrada:", input_values.shape)

    # Realizar la predicción
    prediction = model.predict(input_values)
    predicted_probability = prediction[0, 1]  
    predicted_probability2 = prediction[0, 0]  
    print(predicted_probability)
    print(predicted_probability2)
    
    # Aquí puedes procesar la salida de la predicción según sea necesario
    # Por ejemplo, mostrar el resultado
    if prediction[0][0] < prediction[0][1]:
        result = "Riesgo de incumplimiento de pago"
    else:
        result = "Sin riesgo de incumplimiento de pago"
    
    return f'Predicción: {result}'

if __name__ == '__main__':
    app.run_server(debug=True)
