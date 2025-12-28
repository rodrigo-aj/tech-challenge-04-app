

# Obesity Prediction System - Tech Challenge Fase 4

Este repositório contém a solução desenvolvida para o Tech Challenge da Fase 4 (Data Analytics). O projeto consiste em uma aplicação de **Machine Learning** end-to-end para auxiliar equipes médicas no diagnóstico de obesidade, baseada em dados clínicos e comportamentais.

## Visão Geral do Projeto

O objetivo é classificar pacientes em 7 níveis de obesidade (de "Abaixo do Peso" a "Obesidade Tipo III") com alta assertividade. O sistema entrega:
1.  **Pipeline de Treinamento:** Script automatizado de engenharia de dados e modelagem.
2.  **Web App (Streamlit):** Interface para médicos simularem diagnósticos em tempo real.
3.  **Dashboard Analítico:** TBD

---

## Como Executar o Projeto

Siga os passos abaixo para rodar a aplicação em seu ambiente local.

### 1. Pré-requisitos
* Python 3.8 ou superior.
* Arquivo `Obesity.csv` posicionado na pasta `data/` (necessário para o treino inicial).

#### Instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### 3. Treinamento do Modelo (Obrigatório na 1ª execução)

Antes de abrir o aplicativo, é necessário gerar os artefatos do modelo (`.pkl`). Execute o script de produção:

```bash
python src/train_final.py
```

> **O que este comando faz?**
> * Carrega os dados brutos de `data/Obesity.csv`.
> * Executa a engenharia de features (Cálculo de IMC, Scores de Hábitos, etc.).
> * Treina o algoritmo XGBoost com 100% do dataset.
> * Salva o modelo treinado e o encoder na pasta `models/`.
> * Gera um log de previsões em `data/historico_predicoes_treino.csv`.

### 4. Execução da Aplicação

Inicie o servidor do Streamlit:

```bash
streamlit run app.py
```

O navegador abrirá automaticamente no endereço `http://localhost:8501`.

---

## Lógica do Modelo e Decisões Técnicas

### O Algoritmo

Optamos pelo **XGBoost Classifier** devido à sua capacidade de lidar com relações não-lineares e alta performance em dados tabulares.

* **Acurácia Global:** ~98%
* **Validação:** Cross-Validation Estratificado (5-Folds).

### Features e Engenharia

O modelo não utiliza apenas os dados brutos. Foram criadas variáveis sintéticas para capturar comportamentos complexos:

* `BMI` (IMC): O preditor mestre, calculado matematicamente ().
* `Sedentary_Ratio`: Relação entre tempo de tela e atividade física.
* `Hydration_Efficiency`: Consumo de água relativo ao peso corporal.
* `Unhealthy_Score`: Índice somatório de hábitos nocivos (fumo, álcool, dieta calórica).

### Nota Importante: Diagnóstico vs. Hábitos

Ao utilizar o simulador, uma dúvida comum é: *"Mudar a alimentação ou exercícios altera o diagnóstico?"*

**A resposta é: Depende.**

1. **Diagnóstico (O "O Que"):** É determinado majoritariamente por **Peso** e **Altura** (IMC). Se um paciente tem IMC 40 (Obesidade III), o modelo não mudará o diagnóstico para "Normal" apenas porque ele come vegetais. Isso garante o alinhamento com os critérios clínicos da OMS.
2. **Fatores de Risco (O "Porquê"):** As variáveis de estilo de vida (`SMOKE`, `FAVC`, etc.) atuam como "critérios de desempate" em casos de **fronteira (borderline)**. Elas ajudam o modelo a decidir entre categorias vizinhas (ex: *Sobrepeso II* vs *Obesidade I*) quando o IMC está no limiar de transição.