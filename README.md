# 🧠 Machine Learning Aplicado à Saúde

Este projeto tem como objetivo demonstrar a aplicação prática de **técnicas de Aprendizagem Supervisionada e Não Supervisionada** no contexto da saúde, utilizando o **dataset de Câncer de Mama** disponível na biblioteca `scikit-learn`.  

A aplicação foi desenvolvida com **Streamlit**, permitindo interação direta com os resultados e visualizações de maneira intuitiva e acessível.

---

## 🩺 Contexto do Problema

O câncer de mama é uma das doenças mais estudadas e diagnosticadas do mundo. O dataset utilizado contém informações clínicas e laboratoriais de pacientes, com o objetivo de **classificar tumores como benignos ou malignos** com base em diversas características numéricas das células analisadas.

Este projeto busca:
- Explorar estatísticas descritivas e padrões nos dados;
- Aplicar **modelos supervisionados** para prever o diagnóstico;
- Utilizar **técnicas não supervisionadas** para identificar grupos de comportamento similares;
- Visualizar resultados e insights de forma interativa.

---

## 🚀 Tecnologias Utilizadas

| Categoria | Tecnologias |
|------------|--------------|
| Linguagem | **Python 3.10+** |
| Interface | **Streamlit** |
| Machine Learning | **Scikit-learn** |
| Análise e Visualização | **Pandas**, **Seaborn**, **Matplotlib**, **NumPy** |

---

## 📊 Modelos Implementados

### 🔹 Aprendizagem Supervisionada
**Algoritmo:** `RandomForestClassifier`  
- Objetivo: prever o diagnóstico (benigno/maligno)  
- Métricas apresentadas: **Acurácia**, **Relatório de Classificação**, **Matriz de Confusão**  
- Justificativa: o Random Forest é robusto, lida bem com dados numéricos e permite interpretar a importância das variáveis.

### 🔹 Aprendizagem Não Supervisionada
**Algoritmo:** `KMeans` com **Redução de Dimensionalidade via PCA**  
- Objetivo: identificar agrupamentos naturais de pacientes com base nas características das células  
- Visualização: gráfico 2D dos componentes principais mostrando a separação dos clusters  
- Interpretação: compara os clusters encontrados com as classes reais do diagnóstico.

---

## 📈 Principais Etapas do Projeto

1. **Carregamento e Exploração dos Dados (EDA)**  
   - Estatísticas descritivas e distribuição das classes.  
2. **Pré-processamento e Normalização**  
   - Escalonamento com `StandardScaler`.  
3. **Treinamento do Modelo Supervisionado**  
   - Divisão entre treino e teste e cálculo de métricas.  
4. **Agrupamento Não Supervisionado (KMeans + PCA)**  
   - Redução de dimensionalidade e visualização dos clusters.  
5. **Interface Interativa no Streamlit**  
   - Visualização dinâmica dos resultados, gráficos e métricas.

---

## 💡 Como Executar Localmente

Clone este repositório e instale as dependências:

```bash
git clone https://github.com/pedrxzz/ml-saude-pedro-lucas.git
cd ml-saude-pedro-lucas
pip install -r requirements.txt
