import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# --- CONFIG DE DIRETÓRIOS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
ENCODER_CLASSES_PATH = MODELS_DIR / 'encoder_classes.pkl'
MODELO_OBESIDADE_PATH = MODELS_DIR / 'modelo_obesidade.pkl'
CSV_FILE = DATA_DIR / 'Obesity.csv'

# Setup básico da página
st.set_page_config(
    page_title="Obesity Predictor | Tech Challenge",
    layout="wide"
)

# --- MAPAS DE TRADUÇÃO (DE-PARA) ---
MAP_YES_NO = {'yes': 'Sim', 'no': 'Não'}
MAP_GENDER = {'Male': 'Masculino', 'Female': 'Feminino'}
MAP_FREQ = {
    'no': 'Não', 
    'Sometimes': 'Às Vezes', 
    'Frequently': 'Frequentemente', 
    'Always': 'Sempre'
}
MAP_MTRANS = {
    'Public_Transportation': 'Transporte Público', 
    'Walking': 'A Pé', 
    'Automobile': 'Automóvel', 
    'Motorbike': 'Motocicleta', 
    'Bike': 'Bicicleta'
}
MAP_OBESITY = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III (Mórbida)'
}

REV_YES_NO = {v: k for k, v in MAP_YES_NO.items()}
REV_GENDER = {v: k for k, v in MAP_GENDER.items()}
REV_FREQ = {v: k for k, v in MAP_FREQ.items()}
REV_MTRANS = {v: k for k, v in MAP_MTRANS.items()}


# --- CARGA DE DADOS E MODELO ---
# Cache pra não ficar recarregando o modelo pesado a cada clique
@st.cache_resource
def load_model():
    pipeline = joblib.load(MODELO_OBESIDADE_PATH)
    target_encoder = joblib.load(ENCODER_CLASSES_PATH)
    return pipeline, target_encoder

@st.cache_data
def load_data():
    """
    Lê o CSV e já traduz as colunas categóricas.
    Assim o Dashboard já mostra tudo em PT-BR direto.
    """
    df = pd.read_csv(CSV_FILE)
    
    cols_yes_no = ['family_history', 'FAVC', 'SMOKE', 'SCC']
    for col in cols_yes_no:
        df[col] = df[col].map(MAP_YES_NO).fillna(df[col])
        
    df['Gender'] = df['Gender'].map(MAP_GENDER).fillna(df['Gender'])
    df['CAEC'] = df['CAEC'].map(MAP_FREQ).fillna(df['CAEC'])
    df['CALC'] = df['CALC'].map(MAP_FREQ).fillna(df['CALC'])
    df['MTRANS'] = df['MTRANS'].map(MAP_MTRANS).fillna(df['MTRANS'])
    
    # Target também
    df['Obesity'] = df['Obesity'].map(MAP_OBESITY).fillna(df['Obesity'])
        
    return df

try:
    pipeline, target_encoder = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("Erro Crítico: Arquivos não encontrados. Roda o 'python src/train_final.py' antes pra gerar os pkl.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/clouds/200/health-check.png", width=150)
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Simulador de Diagnóstico", "Dashboard Analítico"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tech Challenge - Fase 4**\n\n"
    "Sistema de apoio à decisão médica.\n"
    "Tradução: PT-BR"
)

# --- TELA 1: SIMULADOR ---
if page == "Simulador de Diagnóstico":
    st.title("Sistema de Predição de Obesidade")
    st.markdown("Preencha os dados do paciente para obter o diagnóstico clínico.")

    with st.form("form_paciente"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Dados Biométricos")
            # Populando os selects com os valores em PT dos dicts
            gender = st.selectbox("Gênero", list(MAP_GENDER.values()))
            age = st.number_input("Idade", 10, 100, 25)
            height = st.number_input("Altura (m)", 1.00, 2.50, 1.70, 0.01)
            weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0, 0.1)
            family_history = st.radio("Histórico Familiar de Obesidade?", list(MAP_YES_NO.values()))

        with col2:
            st.subheader("Hábitos Alimentares")
            favc = st.radio("Consome alimentos calóricos frequentemente?", list(MAP_YES_NO.values()))
            fcvc = st.slider("Consumo de Vegetais (1=Nunca, 3=Sempre)", 1.0, 3.0, 2.0)
            ncp = st.slider("Refeições por dia", 1.0, 4.0, 3.0)
            caec = st.selectbox("Comer entre refeições", list(MAP_FREQ.values()), index=1)
            ch2o = st.slider("Consumo de Água (L/dia aprox)", 1.0, 3.0, 2.0)
            calc = st.selectbox("Consumo de Álcool", list(MAP_FREQ.values()), index=1)

        with col3:
            st.subheader("Estilo de Vida")
            smoke = st.radio("Fumante?", list(MAP_YES_NO.values()), index=1)
            scc = st.radio("Monitora calorias?", list(MAP_YES_NO.values()), index=1)
            faf = st.slider("Freq. Atividade Física (0=Sedentário, 3=Alto)", 0.0, 3.0, 1.0)
            tue = st.slider("Tempo em Dispositivos (Tecnologia)", 0.0, 2.0, 1.0)
            mtrans = st.selectbox("Meio de Transporte", list(MAP_MTRANS.values()))

        submit_btn = st.form_submit_button("Gerar Diagnóstico")

    if submit_btn:
        # DF temporário só pra visualização/log se precisar
        input_data = pd.DataFrame({
            'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history': [family_history], 'FAVC': [favc], 'FCVC': [fcvc],
            'NCP': [ncp], 'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o],
            'SCC': [scc], 'FAF': [faf], 'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
        })

        # Feature Engineering (reproduzindo a lógica do treino)
        input_data['BMI'] = input_data['Weight'] / (input_data['Height'] ** 2)
        input_data['Sedentary_Ratio'] = input_data['TUE'] - input_data['FAF']
        input_data['Hydration_Efficiency'] = input_data['CH2O'] / input_data['Weight']
        
        # Comparação direta com strings PT pra gerar o score
        input_data['Unhealthy_Score'] = (
            (input_data['SMOKE'] == 'Sim').astype(int) + 
            (input_data['FAVC'] == 'Sim').astype(int) + 
            (input_data['CALC'] != 'Não').astype(int)
        )

        try:
            model_input = input_data.copy()
            
            # Reverto tudo pra Inglês pro XGBoost entender
            model_input['Gender'] = model_input['Gender'].map(REV_GENDER)
            model_input['family_history'] = model_input['family_history'].map(REV_YES_NO)
            model_input['FAVC'] = model_input['FAVC'].map(REV_YES_NO)
            model_input['SMOKE'] = model_input['SMOKE'].map(REV_YES_NO)
            model_input['SCC'] = model_input['SCC'].map(REV_YES_NO)
            model_input['CAEC'] = model_input['CAEC'].map(REV_FREQ)
            model_input['CALC'] = model_input['CALC'].map(REV_FREQ)
            model_input['MTRANS'] = model_input['MTRANS'].map(REV_MTRANS)
            
            # Mando pro pipeline
            pred_encoded = pipeline.predict(model_input)
            pred_original_label = target_encoder.inverse_transform(pred_encoded)[0]
            
            # Traduzo a resposta final de volta pra PT
            pred_pt_label = MAP_OBESITY.get(pred_original_label, pred_original_label)
            
            st.markdown("---")
            st.subheader("Resultado da Análise:")
            
            # Cores de alerta
            color = "green"
            if "Sobrepeso" in pred_pt_label: color = "orange"
            if "Obesidade" in pred_pt_label: color = "red"
            
            st.markdown(f"### O modelo identificou: :{color}[{pred_pt_label}]")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("IMC Calculado", f"{input_data['BMI'][0]:.2f} kg/m²")
            c2.metric("Score de Hábitos Nocivos", f"{input_data['Unhealthy_Score'][0]}")
            c3.metric("Hidratação Relativa", f"{input_data['Hydration_Efficiency'][0]:.3f} L/kg")
            
            if color == "red":
                st.warning("Atenção: Resultado indica obesidade. Consulte um médico.")

        except Exception as e:
            st.error(f"Deu erro na predição: {e}")

# --- TELA 2: DASHBOARD ---
elif page == "Dashboard Analítico":
    st.title("Laboratório de Saúde Populacional")
    st.markdown("Utilize o painel para filtrar a população e analisar os dados demográficos e hábitos relacionados à obesidade.")

    # Filtros Expansíveis
    with st.expander("Filtros de Segmentação", expanded=True):
        f1, f2, f3 = st.columns(3)
        
        with f1:
            st.markdown("### Perfil")
            sel_gender = st.multiselect("Gênero", df['Gender'].unique(), df['Gender'].unique())
            min_a, max_a = int(df['Age'].min()), int(df['Age'].max())
            sel_age = st.slider("Faixa Etária", min_a, max_a, (min_a, max_a))
            sel_hist = st.multiselect("Histórico Familiar", df['family_history'].unique(), df['family_history'].unique())

        with f2:
            st.markdown("### Alimentação")
            sel_favc = st.multiselect("Consome Calóricos?", df['FAVC'].unique(), df['FAVC'].unique())
            sel_caec = st.multiselect("Comer entre ref.", df['CAEC'].unique(), df['CAEC'].unique())
            sel_calc = st.multiselect("Álcool", df['CALC'].unique(), df['CALC'].unique())
            sel_fcvc = st.slider("Vegetais (Score)", 1.0, 3.0, (1.0, 3.0))

        with f3:
            st.markdown("### Estilo de Vida")
            sel_smoke = st.multiselect("Fumante?", df['SMOKE'].unique(), df['SMOKE'].unique())
            sel_mtrans = st.multiselect("Transporte", df['MTRANS'].unique(), df['MTRANS'].unique())
            sel_faf = st.slider("Ativ. Física (Score)", 0.0, 3.0, (0.0, 3.0))

    # Aplicando os filtros no DF
    df_f = df.copy()
    df_f = df_f[
        (df_f['Gender'].isin(sel_gender)) & (df_f['family_history'].isin(sel_hist)) &
        (df_f['FAVC'].isin(sel_favc)) & (df_f['CAEC'].isin(sel_caec)) &
        (df_f['CALC'].isin(sel_calc)) & (df_f['SMOKE'].isin(sel_smoke)) &
        (df_f['MTRANS'].isin(sel_mtrans)) &
        (df_f['Age'].between(sel_age[0], sel_age[1])) &
        (df_f['FCVC'].between(sel_fcvc[0], sel_fcvc[1])) &
        (df_f['FAF'].between(sel_faf[0], sel_faf[1]))
    ]

    # Sanity check: Se não sobrou ninguém, para tudo.
    if df_f.empty:
        st.warning("Nenhum dado encontrado para este filtro.")
        st.stop()

    # KPIs principais
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    avg_bmi = (df_f['Weight'] / (df_f['Height']**2)).mean()
    k1.metric("Nº Pacientes", len(df_f))
    k2.metric("Idade Média", f"{df_f['Age'].mean():.1f}")
    k3.metric("Peso Médio", f"{df_f['Weight'].mean():.1f} kg")
    k4.metric("IMC Médio", f"{avg_bmi:.1f}")

    # Gráficos com Plotly
    g1, g2 = st.columns(2)
    with g1:
        df_count = df_f['Obesity'].value_counts().reset_index()
        df_count.columns = ['Diagnostico', 'Total']
        fig = px.bar(df_count, x='Diagnostico', y='Total', color='Diagnostico', title="Distribuição por Diagnóstico")
        st.plotly_chart(fig, use_container_width=True)
    
    with g2:
        df_f['IMC_Calc'] = df_f['Weight'] / (df_f['Height']**2)
        fig = px.scatter(df_f, x='Weight', y='FAF', color='Obesity', size='IMC_Calc', 
                         title="Peso x Atividade Física (Cor = Diagnóstico)")
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig = px.box(df_f, x='MTRANS', y='Weight', color='MTRANS', title="Peso por Transporte")
        st.plotly_chart(fig, use_container_width=True)
    with g4:
        # Hierarquia (Sunburst) pra ver correlação multinível
        try:
            fig = px.sunburst(df_f, path=['CALC', 'SMOKE', 'Obesity'], title="Álcool -> Fumo -> Obesidade")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Dados insuficientes pra montar a hierarquia.")
            
    with st.expander("Ver Dados Detalhados"):
        st.dataframe(df_f)