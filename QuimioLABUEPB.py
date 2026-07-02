import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="QuimioLab UEPB",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 QuimioLab UEPB")
st.markdown("### Plataforma Interativa para Ensino de Quimiometria")

perfil = st.sidebar.radio(
    "Perfil",
    ["Aprendiz", "Mestre"]
)

st.sidebar.success(f"Modo selecionado: {perfil}")
st.sidebar.markdown("---")

modulo = st.sidebar.selectbox(
    "Escolha um módulo",
    [
        "Explorador de Dados",
        "Pré-processamento",
        "Reconhecimento de Padrões",
        "Classificação",
        "Calibração",
        "Escape Room"
    ]
)

def carregar_dados(chave):
    arquivo = st.file_uploader(
        "Carregue CSV ou XLSX",
        type=["csv", "xlsx"],
        key=chave
    )

    if arquivo is not None:
        if arquivo.name.endswith(".xlsx"):
            dados = pd.read_excel(arquivo)
        else:
            dados = pd.read_csv(arquivo)

        X = dados.select_dtypes(include=np.number)

        if X.shape[1] < 2:
            st.error("A matriz precisa ter pelo menos duas variáveis numéricas.")
            return None, None

        return dados, X

    return None, None


def aplicar_preprocessamento(X, metodo):
    if metodo == "Nenhum":
        return X.copy()

    elif metodo == "Centralização":
        return X - X.mean()

    elif metodo == "Autoescalamento":
        return (X - X.mean()) / X.std()

    elif metodo == "SNV":
        return X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1), axis=0)

    return X.copy()


#####################################
# EXPLORADOR DE DADOS
#####################################

if modulo == "Explorador de Dados":

    st.header("📊 Explorador de Dados")

    dados, X = carregar_dados("explorador")

    if X is not None:

        st.subheader("Matriz carregada")
        st.dataframe(dados)

        col1, col2 = st.columns(2)
        col1.metric("Amostras", X.shape[0])
        col2.metric("Variáveis numéricas", X.shape[1])

        st.subheader("Estatística descritiva")
        st.dataframe(X.describe())

        variavel = st.selectbox("Escolha uma variável", X.columns)

        st.subheader("Histograma")
        fig, ax = plt.subplots()
        ax.hist(X[variavel], bins=20)
        ax.set_xlabel(variavel)
        ax.set_ylabel("Frequência")
        st.pyplot(fig)

        st.subheader("Boxplot")
        fig2, ax2 = plt.subplots()
        ax2.boxplot(X[variavel])
        ax2.set_ylabel(variavel)
        st.pyplot(fig2)

        st.subheader("Matriz de correlação")
        corr = X.corr()
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        im = ax3.imshow(corr, aspect="auto")
        ax3.set_xticks(range(len(corr.columns)))
        ax3.set_yticks(range(len(corr.columns)))
        ax3.set_xticklabels(corr.columns, rotation=90)
        ax3.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig3)

        st.info(
            f"Foram carregadas {X.shape[0]} amostras e {X.shape[1]} variáveis numéricas."
        )

        st.success("🏅 Badge desbloqueada: Explorador Químico")


#####################################
# PRÉ-PROCESSAMENTO
#####################################

elif modulo == "Pré-processamento":

    st.header("🔬 Pré-processamento")

    dados, X = carregar_dados("prep")

    if X is not None:

        metodo = st.selectbox(
            "Método",
            ["Nenhum", "Centralização", "Autoescalamento", "SNV"]
        )

        Xproc = aplicar_preprocessamento(X, metodo)

        st.subheader("Dados originais")
        st.dataframe(X.head())

        st.subheader("Dados processados")
        st.dataframe(Xproc.head())

        variavel = st.selectbox("Escolha uma variável", X.columns)

        st.subheader("Comparação: Histograma")
        fig, ax = plt.subplots()
        ax.hist(X[variavel], alpha=0.5, label="Original")
        ax.hist(Xproc[variavel], alpha=0.5, label="Processado")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Comparação: Boxplot")
        fig2, ax2 = plt.subplots()
        ax2.boxplot([X[variavel], Xproc[variavel]])
        ax2.set_xticklabels(["Original", "Processado"])
        st.pyplot(fig2)

        st.subheader("Interpretação")

        if metodo == "Nenhum":
            st.info("Nenhum pré-processamento foi aplicado.")

        elif metodo == "Centralização":
            st.info("Os dados foram centralizados em torno da média.")

        elif metodo == "Autoescalamento":
            st.info("As variáveis foram padronizadas para média zero e desvio padrão igual a um.")

        elif metodo == "SNV":
            st.info("O SNV reduz efeitos de espalhamento e variações entre amostras.")

        st.success("🏅 Badge desbloqueada: Mestre do Pré-processamento")


#####################################
# RECONHECIMENTO DE PADRÕES
#####################################

elif modulo == "Reconhecimento de Padrões":

    st.header("📈 Reconhecimento de Padrões — PCA")

    dados, X = carregar_dados("pca")

    if X is not None:

        metodo = st.selectbox(
            "Pré-processamento",
            ["Nenhum", "Centralização", "Autoescalamento", "SNV"]
        )

        Xproc = aplicar_preprocessamento(X, metodo)

        pca = PCA()
        scores = pca.fit_transform(Xproc)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        st.subheader("Variância explicada")
        st.write(pd.DataFrame({
            "Componente": [f"PC{i+1}" for i in range(len(explained))],
            "Variância explicada (%)": explained
        }).head(10))

        fig, ax = plt.subplots()
        ax.plot(range(1, len(explained)+1), explained, marker="o")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Variância explicada (%)")
        st.pyplot(fig)

        st.subheader("Scores: PC1 × PC2")

        fig2, ax2 = plt.subplots()
        ax2.scatter(scores[:, 0], scores[:, 1])

        for i in range(scores.shape[0]):
            ax2.text(scores[i, 0], scores[i, 1], str(i+1))

        ax2.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax2.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        st.pyplot(fig2)

        st.subheader("Loadings: PC1 × PC2")

        fig3, ax3 = plt.subplots()
        ax3.scatter(loadings[:, 0], loadings[:, 1])

        for i, var in enumerate(X.columns):
            ax3.text(loadings[i, 0], loadings[i, 1], var)

        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        st.pyplot(fig3)

        st.subheader("Interpretação automática")

        st.info(f"""
PC1 explica {explained[0]:.2f}% da variância.  
PC2 explica {explained[1]:.2f}% da variância.  
Juntas, PC1 e PC2 explicam {explained[0] + explained[1]:.2f}% da variabilidade dos dados.
""")

        if perfil == "Aprendiz":
            st.success("🏅 Badge desbloqueada: Explorador Multivariado")

        elif perfil == "Mestre":
            st.subheader("Tabela completa de loadings")

            tabela_loadings = pd.DataFrame(
                loadings,
                index=X.columns,
                columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
            )

            st.dataframe(tabela_loadings)

       st.subheader("Quiz")

variancia_acumulada = np.cumsum(explained)
n_pcs_95 = np.argmax(variancia_acumulada >= 95) + 1

if n_pcs_95 <= 3:
    gabarito = str(n_pcs_95)
else:
    gabarito = "Mais de 3"

resposta = st.radio(
    "Quantos PCs são necessários para explicar pelo menos 95% da variância?",
    ["1", "2", "3", "Mais de 3"],
    key="quiz_pca"
)

if st.button("Verificar resposta"):

    if resposta == gabarito:
        st.success(f"✅ Correto! Resposta: {gabarito}.")
    else:
        st.error(f"❌ Ainda não. A resposta correta é: {gabarito}.")

    st.info(
        f"Variância acumulada: {variancia_acumulada[n_pcs_95-1]:.2f}% "
        f"com {n_pcs_95} PC(s)."
    )


#####################################
# CLASSIFICAÇÃO
#####################################

elif modulo == "Classificação":

    st.header("🎯 Classificação")
    st.info("Módulo em desenvolvimento.")


#####################################
# CALIBRAÇÃO
#####################################

elif modulo == "Calibração":

    st.header("📐 Calibração")
    st.info("Módulo em desenvolvimento.")


#####################################
# ESCAPE ROOM
#####################################

elif modulo == "Escape Room":

    st.header("🔓 Escape Room Quimiométrico")
    st.info("Módulo em desenvolvimento.")