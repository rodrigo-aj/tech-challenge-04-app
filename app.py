import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR                        = Path(__file__).resolve().parent
DATA_DIR                        = BASE_DIR / 'data'
MODELS_DIR                      = BASE_DIR / 'models'
ENCODER_CLASSES_PATH            = MODELS_DIR / 'encoder_classes.pkl'
MODELO_OBESIDADE_PATH           = MODELS_DIR / 'modelo_obesidade.pkl'
CSV_FILE                        = DATA_DIR / 'Obesity.csv'

# Configuração da Página
st.set_page_config(
    page_title="Obesity Predictor | Tech Challenge",
    layout="wide"
)

# --- FUNÇÕES DE CARGA (CACHED) ---
@st.cache_resource
def load_model():
    # Carrega o pipeline completo e o encoder das classes
    pipeline = joblib.load(MODELO_OBESIDADE_PATH)
    target_encoder = joblib.load(ENCODER_CLASSES_PATH)
    return pipeline, target_encoder

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    return df

# Carrega os artefatos
try:
    pipeline, target_encoder = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("Erro: Arquivos de modelo (.pkl) ou dados (.csv) não encontrados. Execute o script de treino primeiro.")
    st.stop()

# --- SIDEBAR DE NAVEGAÇÃO ---
st.sidebar.image("https://img.icons8.com/clouds/200/health-check.png", width=150)
st.sidebar.title("Navegação")
# page = st.sidebar.radio("Ir para:", ["Simulador de Diagnóstico", "Dashboard Analítico"])
page = st.sidebar.radio("Ir para:", ["Simulador de Diagnóstico"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tech Challenge - Fase 4**\n\n"
    "Sistema de apoio à decisão médica para diagnóstico de obesidade.\n\n"
)

# --- PÁGINA 1: SIMULADOR (PREDIÇÃO) ---
if page == "Simulador de Diagnóstico":
    st.title("🩺 Sistema de Predição de Obesidade")
    st.markdown("Preencha os dados do paciente abaixo para obter o diagnóstico sugerido pelo modelo.")

    with st.form("form_paciente"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Dados Biométricos")
            gender = st.selectbox("Gênero", ["Male", "Female"])
            age = st.number_input("Idade", min_value=10, max_value=100, value=25)
            height = st.number_input("Altura (m)", min_value=1.00, max_value=2.50, value=1.70, step=0.01)
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            family_history = st.radio("Histórico Familiar de Obesidade?", ["yes", "no"])

        with col2:
            st.subheader("Hábitos Alimentares")
            favc = st.radio("Consome alimentos calóricos frequentemente?", ["yes", "no"])
            fcvc = st.slider("Consumo de Vegetais (1=Nunca, 3=Sempre)", 1.0, 3.0, 2.0)
            ncp = st.slider("Refeições por dia", 1.0, 4.0, 3.0)
            caec = st.selectbox("Comer entre refeições", ["no", "Sometimes", "Frequently", "Always"])
            ch2o = st.slider("Consumo de Água (L/dia aprox)", 1.0, 3.0, 2.0)
            calc = st.selectbox("Consumo de Álcool", ["no", "Sometimes", "Frequently", "Always"])

        with col3:
            st.subheader("Estilo de Vida")
            smoke = st.radio("Fumante?", ["yes", "no"])
            scc = st.radio("Monitora calorias?", ["yes", "no"])
            faf = st.slider("Freq. Atividade Física (0=Sedentário, 3=Alto)", 0.0, 3.0, 1.0)
            tue = st.slider("Tempo em Dispositivos (Tecnologia)", 0.0, 2.0, 1.0)
            mtrans = st.selectbox("Meio de Transporte", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

        submit_btn = st.form_submit_button("Gerar Diagnóstico")

if submit_btn:
        # 1. Monta o DataFrame com os inputs
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history': [family_history],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [caec],
            'SMOKE': [smoke],
            'CH2O': [ch2o],
            'SCC': [scc],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc],
            'MTRANS': [mtrans]
        })

        # 2. Engenharia de Features (CRÍTICO: Deve ser idêntico ao treino)
        # ---------------------------------------------------------
        
        # Feature 1: BMI
        input_data['BMI'] = input_data['Weight'] / (input_data['Height'] ** 2)
        
        # Feature 2: Sedentary_Ratio
        input_data['Sedentary_Ratio'] = input_data['TUE'] - input_data['FAF']
        
        # Feature 3: Hydration_Efficiency
        input_data['Hydration_Efficiency'] = input_data['CH2O'] / input_data['Weight']
        
        # Feature 4: Unhealthy_Score
        # Mapeamento manual igual ao realizado no treino
        flag_smoke = (input_data['SMOKE'] == 'yes').astype(int)
        flag_favc = (input_data['FAVC'] == 'yes').astype(int)
        flag_calc = (input_data['CALC'] != 'no').astype(int)
        
        input_data['Unhealthy_Score'] = flag_smoke + flag_favc + flag_calc
        
        # ---------------------------------------------------------

        # 3. Predição
        try:
            # O pipeline agora receberá todas as colunas que espera
            prediction_encoded = pipeline.predict(input_data)
            prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]
            
            # 4. Exibição do Resultado
            st.markdown("---")
            st.subheader("Resultado da Análise:")
            
            # Lógica de cores para o resultado
            color = "green"
            if "Overweight" in prediction_label: color = "orange"
            if "Obesity" in prediction_label: color = "red"
            
            st.markdown(f"### O modelo identificou: :{color}[{prediction_label}]")
            
            # Exibe métricas calculadas para feedback visual
            c1, c2, c3 = st.columns(3)
            c1.metric("IMC Calculado", f"{input_data['BMI'][0]:.2f} kg/m²")
            c2.metric("Score de Hábitos Nocivos", f"{input_data['Unhealthy_Score'][0]}")
            c3.metric("Hidratação Relativa", f"{input_data['Hydration_Efficiency'][0]:.3f} L/kg")
            
            if color == "red":
                st.warning("⚠️ Atenção: Este resultado indica um grau de obesidade. Recomenda-se acompanhamento médico especializado.")
                
        except Exception as e:
            st.error(f"Erro na predição: {e}")

# --- PÁGINA 2: DASHBOARD (VISÃO DE NEGÓCIO) ---
elif page == "Dashboard Analítico":
    st.title("📊 Dashboard Analítico de Obesidade")
    st.markdown("Insights sobre os fatores de risco baseados na base histórica de pacientes.")

    # Métricas Gerais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Pacientes", len(df))
    col2.metric("Média de Idade", f"{df['Age'].mean():.1f} anos")
    col3.metric("Peso Médio", f"{df['Weight'].mean():.1f} kg")
    col4.metric("% Com Histórico Familiar", f"{(df['family_history'] == 'yes').mean():.1%}")

    st.markdown("---")

    # Linha 1 de Gráficos
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribuição dos Níveis de Obesidade")
        fig, ax = plt.subplots()
        sns.countplot(y='Obesity', data=df, order=df['Obesity'].value_counts().index, palette='viridis', ax=ax)
        ax.set_title("Quantidade de Pacientes por Categoria")
        st.pyplot(fig)
        st.caption("Insight: A base possui uma distribuição balanceada entre os níveis de obesidade, garantindo que o modelo aprenda bem todas as classes.")

    with c2:
        st.subheader("Relação Peso x Altura (Clusters)")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Weight', y='Height', hue='Obesity', data=df, palette='viridis', alpha=0.6, ax=ax)
        ax.set_title("Dispersão: Peso vs Altura")
        st.pyplot(fig)
        st.caption("Insight: Note a clara separação das cores (classes) baseada na relação Peso/Altura, validando a importância do IMC.")

    st.markdown("---")
    
    # Linha 2 de Gráficos (Fatores Comportamentais)
    st.subheader("Impacto do Transporte e Tecnologia")
    
    c3, c4 = st.columns(2)
    
    with c3:
        fig, ax = plt.subplots()
        # Ordenar por obesidade para ver gradiente
        order_obesity = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
        sns.boxplot(x='TUE', y='Obesity', data=df, order=order_obesity, palette='coolwarm', ax=ax)
        ax.set_title("Tempo em Dispositivos Eletrônicos (TUE) por Nível")
        st.pyplot(fig)
        
    with c4:
        # Analise de Transporte
        transporte_obesity = df.groupby('MTRANS')['Obesity'].value_counts(normalize=True).unstack()
        st.bar_chart(transporte_obesity)
        st.caption("Distribuição percentual de obesidade por meio de transporte.")