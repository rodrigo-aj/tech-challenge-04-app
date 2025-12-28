import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

"""
DOCUMENTAÇÃO DA VARIÁVEL ALVO (TARGET)
--------------------------------------
O modelo foi treinado para realizar uma classificação multiclasse na variável 'Obesity'.
As classes representam os diferentes graus de classificação do Índice de Massa Corporal (IMC)
e dividem-se em 7 categorias distintas, conforme identificado nos dados de treino:

1. Insufficient_Weight  (Abaixo do Peso)
2. Normal_Weight        (Peso Normal)
3. Overweight_Level_I   (Sobrepeso Nível I)
4. Overweight_Level_II  (Sobrepeso Nível II)
5. Obesity_Type_I       (Obesidade Tipo I)
6. Obesity_Type_II      (Obesidade Tipo II)
7. Obesity_Type_III     (Obesidade Tipo III - Mórbida)

Essas classes foram convertidas numericamente (0 a 6) pelo LabelEncoder durante o
pré-processamento e são revertidas para estes rótulos textuais na saída final.
"""

# ---------------------------------------------------------

BASE_DIR                        = Path(__file__).resolve().parent.parent
DATA_DIR                        = BASE_DIR / 'data'
MODELS_DIR                      = BASE_DIR / 'models'
CSV_FILE                        = DATA_DIR / 'Obesity.csv'
CSV_PREDICTIONS_OUTPUT          = DATA_DIR / 'historico_predicoes_treino.csv'
ENCODER_CLASSES_PATH            = MODELS_DIR / 'encoder_classes.pkl'
MODELO_OBESIDADE_PATH           = MODELS_DIR / 'modelo_obesidade.pkl'

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_final_model():
    print(">>> Iniciando Pipeline de Treinamento (XGBoost)...")
        
    # Foi carregado o dataset a partir do caminho especificado
    df = pd.read_csv(CSV_FILE)
    
    # Foram removidas as linhas duplicadas para garantir a qualidade dos dados
    df = df.drop_duplicates()
    
    # 2. Feature Engineering (Engenharia de Recursos)
    print(">>> Executando Engenharia de Features...")

    # ---------------------------------------------------------
    # Feature 1: BMI (Índice de Massa Corporal)
    # ---------------------------------------------------------
    # Lógica de Negócio: O fator clínico determinante para diagnóstico de obesidade segundo a OMS.
    # Fórmula: Peso (kg) / Altura (m)²
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # ---------------------------------------------------------
    # Feature 2: Sedentary_Ratio (Razão de Sedentarismo)
    # ---------------------------------------------------------
    # Lógica de Negócio: Identificado o "sedentarismo digital" cruzando o tempo excessivo de tela
    # com a baixa frequência de exercícios. Quanto maior o valor, maior o risco.
    # Fórmula: TUE (Tempo Tecnologia) - FAF (Freq. Atividade Física)
    df['Sedentary_Ratio'] = df['TUE'] - df['FAF']

    # ---------------------------------------------------------
    # Feature 3: Hydration_Efficiency (Eficiência de Hidratação)
    # ---------------------------------------------------------
    # Lógica de Negócio: Relativiza o consumo de água pelo peso corporal. Pessoas mais pesadas 
    # precisam de mais água; o consumo absoluto engana o modelo.
    # Fórmula: CH2O (Litros de Água) / Weight (Peso)
    df['Hydration_Efficiency'] = df['CH2O'] / df['Weight']

    # ---------------------------------------------------------
    # Feature 4: Unhealthy_Score (Score de Hábitos Nocivos)
    # ---------------------------------------------------------
    # Lógica de Negócio: Foi criado um "score de risco" somando comportamentos negativos.
    # Agrupa pequenos hábitos (fumar, comer calorias, álcool) em um indicador.
    # Fórmula: (Fuma? 1:0) + (Comida Calórica? 1:0) + (Bebe Álcool? 1:0)
    
    # Mapeamento manual temporário para cálculo aritmético (antes do OneHotEncoder)
    flag_smoke = (df['SMOKE'] == 'yes').astype(int)
    flag_favc = (df['FAVC'] == 'yes').astype(int)
    flag_calc = (df['CALC'] != 'no').astype(int) # Considera risco se beber "às vezes", "frequentemente" ou "sempre"
    
    df['Unhealthy_Score'] = flag_smoke + flag_favc + flag_calc

    # ---------------------------------------------------------

    # Foram separadas as variáveis preditoras (X) da variável alvo (y)
    X = df.drop(columns=['Obesity'])
    y = df['Obesity']
    
    # 3. Encoding do Target
    # Foi instanciado e ajustado o LabelEncoder para transformar as classes em números
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Foi salvo o encoder para permitir a decodificação das predições na aplicação final
    joblib.dump(le, MODELS_DIR / 'encoder_classes.pkl')
    
    # 4. Configuração do Pipeline
    # Foram identificadas automaticamente as colunas categóricas e numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    # Foi configurado o pré-processador com padronização para numéricos e OneHot para categóricos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Foi estruturado o pipeline final integrando pré-processamento e o classificador XGBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='multi:softmax',
            num_class=len(le.classes_), # Foi definido automaticamente o nº de classes (7)
            eval_metric='mlogloss',
            random_state=42
        ))
    ])
    
    # 5. Treinamento Full
    print(">>> Treinando modelo com 100% dos dados e features...")
    # Foi executado o treinamento do modelo com a totalidade dos dados disponíveis
    pipeline.fit(X, y_encoded)
    
    # 6. Geração de Resultados em CSV
    print(">>> Gerando arquivo de previsões para conferência...")
    
    # Foram geradas as predições usando os próprios dados de treino para validação final
    y_pred_encoded = pipeline.predict(X)
    
    # Foram revertidos os códigos numéricos para os rótulos originais
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    
    # Foi criado um DataFrame consolidado com os dados originais e a predição do modelo
    df_results = df.copy()
    df_results['Predicao_Modelo'] = y_pred_labels
    
    # Foi criada uma coluna booleana para verificar a assertividade de cada linha
    df_results['Previsao_Correta'] = df_results['Obesity'] == df_results['Predicao_Modelo']
    
    # Foi salvo o arquivo CSV contendo o histórico das predições e a validação
    df_results.to_csv(CSV_PREDICTIONS_OUTPUT, index=False)
    
    # 7. Serialização
    # Foi serializado e salvo o pipeline treinado em formato .pkl
    joblib.dump(pipeline, MODELO_OBESIDADE_PATH)
    
    print(">>> Concluído! Arquivos gerados:")
    print(f"    - {MODELO_OBESIDADE_PATH} (Pipeline Completo)")
    print(f"    - {ENCODER_CLASSES_PATH} (Tradutor de Classes)")
    print(f"    - {CSV_PREDICTIONS_OUTPUT} (Tabela com resultados e validação)")

if __name__ == "__main__":
    train_final_model()