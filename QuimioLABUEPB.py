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

        fig, ax = plt.subplots()
        ax.plot(explained, marker='o')
        ax.set_xlabel("PC")
        ax.set_ylabel("%")
        st.pyplot(fig)

        # 🔥 SCORES (corrigido)
        st.subheader("Scores")

        fig2, ax2 = plt.subplots()
        ax2.scatter(scores[:,0], scores[:,1])
        ax2.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax2.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        st.pyplot(fig2)

        # 🔥 LOADINGS
        st.subheader("Loadings")

        fig3, ax3 = plt.subplots()
        ax3.scatter(loadings[:,0], loadings[:,1])

        for i, var in enumerate(X.columns):
            ax3.text(loadings[i,0], loadings[i,1], var)

        st.pyplot(fig3)

        # 🔥 INTERPRETAÇÃO
        st.subheader("Interpretação")

        st.info(f"""
PC1 explica {explained[0]:.2f}% da variância.  
PC2 explica {explained[1]:.2f}%.  
PC1+PC2 explicam {explained[0]+explained[1]:.2f}%.
""")

        # 🔥 PERFIL
        perfil = st.sidebar.radio(
            "Perfil",
            ["Aprendiz","Mestre"]
        )

        if perfil == "Aprendiz":
            st.success("🏅 Badge desbloqueada: Explorador Multivariado")

        if perfil == "Mestre":
            st.subheader("Tabela de Loadings")

            tabela = pd.DataFrame(
                loadings,
                index=X.columns,
                columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
            )

            st.dataframe(tabela)

        # 🔥 QUIZ
        st.subheader("Quiz")

        st.radio(
            "Quantos PCs explicam mais de 95% da variância?",
            ["1","2","3","4"]
        )

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