import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(

page_title="QuimioLab",

page_icon="🧪",

layout="wide"

)

st.title("🧪 QuimioLab UEPB")

st.markdown(
"### Plataforma Interativa para Ensino de Quimiometria"
)

perfil = st.sidebar.radio(

"Perfil",

["Aprendiz","Mestre"]

)

st.sidebar.success(

f"Modo selecionado: {perfil}"

)

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

#####################################

if modulo=="Explorador de Dados":

    st.header("📊 Explorador de Dados")

    arquivo = st.file_uploader(

        "Carregar CSV ou XLSX",

        type=["csv","xlsx"]

    )

    if arquivo:

        if arquivo.name.endswith(".xlsx"):

            dados = pd.read_excel(arquivo)

        else:

            dados = pd.read_csv(arquivo)

        st.subheader("Dados")

        st.dataframe(dados)

        X = dados.select_dtypes(include="number")

        n_amostras = X.shape[0]

        n_variaveis = X.shape[1]

        st.subheader("Resumo")

        col1,col2 = st.columns(2)

        col1.metric(

            "Amostras",

            n_amostras

        )

        col2.metric(

            "Variáveis",

            n_variaveis

        )

        st.subheader(

            "Estatística Descritiva"

        )

        st.dataframe(

            X.describe()

        )

        st.subheader(

            "Histograma"

        )

        var = st.selectbox(

            "Escolha uma variável",

            X.columns

        )

        fig,ax = plt.subplots()

        ax.hist(

            X[var],

            bins=20

        )

        st.pyplot(fig)

        st.subheader(

            "Boxplot"

        )

        fig2,ax2 = plt.subplots()

        ax2.boxplot(

            X[var]

        )

        st.pyplot(fig2)

        st.subheader(

            "Correlação"

        )

        corr = X.corr()

        fig3,ax3 = plt.subplots(

            figsize=(7,7)

        )

        im = ax3.imshow(

            corr,

            aspect='auto'

        )

        plt.colorbar(im)

        st.pyplot(fig3)

        st.subheader(

            "Interpretação"

        )

        st.info(

f"""

Foram carregadas {n_amostras} amostras

e {n_variaveis} variáveis.

Analise a dispersão dos dados

e avalie a necessidade

de pré-processamento.

"""

)

        st.success(

"🏅 Badge desbloqueada: Explorador Químico"

)

#####################################

elif modulo=="Pré-processamento":

    st.header("🔬 Pré-processamento")

    arquivo = st.file_uploader(
        "Carregue CSV ou XLSX",
        type=["csv","xlsx"],
        key="prep"
    )

    if arquivo:

        if arquivo.name.endswith(".xlsx"):
            dados = pd.read_excel(arquivo)
        else:
            dados = pd.read_csv(arquivo)

        X = dados.select_dtypes(include=np.number)

        metodo = st.selectbox(
            "Método",
            ["Nenhum","Centralização","Autoescalamento","SNV"]
        )

        if metodo=="Nenhum":
            Xproc = X.copy()
        elif metodo=="Centralização":
            Xproc = X - X.mean()
        elif metodo=="Autoescalamento":
            Xproc = (X - X.mean())/X.std()
        elif metodo=="SNV":
            Xproc = (X.sub(X.mean(axis=1), axis=0)
                        .div(X.std(axis=1), axis=0))

        st.subheader("Dados Processados")
        st.dataframe(Xproc.head())

        variavel = st.selectbox("Escolha uma variável", X.columns)

        fig,ax = plt.subplots()
        ax.hist(X[variavel], alpha=0.5, label="Original")
        ax.hist(Xproc[variavel], alpha=0.5, label="Processado")
        ax.legend()
        st.pyplot(fig)

        fig2,ax2 = plt.subplots()
        ax2.boxplot([X[variavel], Xproc[variavel]])
        ax2.set_xticklabels(["Original","Processado"])
        st.pyplot(fig2)

        st.subheader("Interpretação")

        if metodo=="Centralização":
            st.info("Os dados foram centralizados em torno da média.")
        elif metodo=="Autoescalamento":
            st.info("As variáveis foram padronizadas.")
        elif metodo=="SNV":
            st.info("SNV reduz efeitos de espalhamento.")

        st.success("🏅 Badge desbloqueada: Mestre do Pré-processamento")


# 🔥 CORRETO: bloco separado do PCA
elif modulo=="Reconhecimento de Padrões":

    st.header("📈 PCA")

    arquivo = st.file_uploader(
        "Carregue CSV ou XLSX",
        type=["csv","xlsx"],
        key="pca"
    )

    if arquivo:

        if arquivo.name.endswith(".xlsx"):
            dados = pd.read_excel(arquivo)
        else:
            dados = pd.read_csv(arquivo)

        X = dados.select_dtypes(include=np.number)

        metodo = st.selectbox(
            "Pré-processamento",
            ["Nenhum","Centralização","Autoescalamento"]
        )

        if metodo=="Nenhum":
            Xproc = X.copy()
        elif metodo=="Centralização":
            Xproc = X - X.mean()
        elif metodo=="Autoescalamento":
            Xproc = (X - X.mean())/X.std()

        from sklearn.decomposition import PCA

        pca = PCA()
        scores = pca.fit_transform(Xproc)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        st.subheader("Variância Explicada")
        st.write(explained[:5])

        fig,ax = plt.subplots()
        ax.plot(explained, marker='o')
        ax.set_xlabel("PC")
        ax.set_ylabel("%")
        st.pyplot(fig)

	st.subheader("Scores")

        fig2,ax2 = plt.subplots()

        ax2.scatter(scores[:,0], scores[:,1])

        ax2.set_xlabel(f"PC1 ({explained[0]:.1f}%)")

        ax2.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        st.pyplot(fig2)

#####################################

elif modulo=="Classificação":

    st.header(

"🎯 Em desenvolvimento"

)

#####################################

elif modulo=="Calibração":

    st.header(

"📐 Em desenvolvimento"

)

#####################################

elif modulo=="Escape Room":

    st.header(

"🔓 Em desenvolvimento"

)