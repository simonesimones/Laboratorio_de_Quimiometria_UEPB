from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="QuimiLAB UEPB",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 QuimiLAB UEPB")
st.markdown("### Plataforma gamificada para ensino de Quimiometria")

# =========================
# CONFIGURAÇÕES INICIAIS
# =========================

if "fase_liberada" not in st.session_state:
    st.session_state.fase_liberada = 1

if "badges" not in st.session_state:
    st.session_state.badges = []

perfil = st.sidebar.radio(
    "Perfil",
    ["Aprendiz", "Mestre"]
)

st.sidebar.success(f"Modo selecionado: {perfil}")

fases = {
    1: "Explorador de Dados",
    2: "Pré-processamento",
    3: "Reconhecimento de Padrões",
    4: "Classificação",
    5: "Calibração",
    6: "Escape Room"
}

st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Jornada")

for numero, nome in fases.items():
    if numero <= st.session_state.fase_liberada:
        st.sidebar.write(f"🔓 Fase {numero}: {nome}")
    else:
        st.sidebar.write(f"🔒 Fase {numero}: {nome}")

fase_escolhida = st.sidebar.selectbox(
    "Escolha a fase",
    list(fases.keys()),
    format_func=lambda x: f"Fase {x} - {fases[x]}"
)

if fase_escolhida > st.session_state.fase_liberada:
    st.error("🔒 Esta fase ainda está bloqueada. Obtenha nota mínima 8 na fase anterior.")
    st.stop()


# =========================
# FUNÇÕES AUXILIARES
# =========================

def carregar_dados(chave):
    arquivo = st.file_uploader(
        "Carregue um arquivo CSV ou XLSX",
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


def preprocessar(X, metodo):
    if metodo == "Nenhum":
        return X.copy()

    elif metodo == "Centralização":
        return X - X.mean()

    elif metodo == "Autoescalamento":
        return (X - X.mean()) / X.std()

    elif metodo == "SNV":
        return X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1), axis=0)

    return X.copy()


def quiz(titulo, perguntas, gabarito, proxima_fase, badge):
    st.markdown("---")
    st.subheader(f"🧩 Quiz - {titulo}")
    st.write("Responda às 5 perguntas. Cada questão vale 2 pontos. Nota mínima para avançar: **8,0**.")

    respostas = []

    for i, pergunta in enumerate(perguntas):
        resp = st.radio(
            pergunta["pergunta"],
            pergunta["opcoes"],
            key=f"{titulo}_{i}"
        )
        respostas.append(resp)

    if st.button("✅ Verificar nota", key=f"botao_{titulo}"):

        acertos = 0

        for r, g in zip(respostas, gabarito):
            if r == g:
                acertos += 1

        nota = acertos * 2

        st.subheader(f"Nota: {nota}/10")

        if nota >= 8:
            st.success("🎉 Parabéns! Fase concluída. Próxima fase liberada.")

            if st.session_state.fase_liberada < proxima_fase:
                st.session_state.fase_liberada = proxima_fase

            if badge not in st.session_state.badges:
                st.session_state.badges.append(badge)

            st.success(f"🏅 Badge desbloqueada: {badge}")

        else:
            st.error("🔒 Você precisa obter pelo menos nota 8 para avançar.")
            st.info("Revise o conteúdo da fase e tente novamente.")


# =========================
# FASE 1 - EXPLORADOR
# =========================

if fase_escolhida == 1:

    st.header("📊 Fase 1 — Explorador de Dados")

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

        perguntas = [
            {
                "pergunta": "1. Em quimiometria, o que representa uma linha da matriz X?",
                "opcoes": ["Uma variável", "Uma amostra", "Um erro", "Um modelo"]
            },
            {
                "pergunta": "2. O que representa uma coluna da matriz X?",
                "opcoes": ["Uma amostra", "Uma variável", "Uma classe", "Uma validação"]
            },
            {
                "pergunta": "3. Para que serve a estatística descritiva?",
                "opcoes": ["Avaliar distribuição dos dados", "Criar moléculas", "Eliminar PCA", "Gerar amostras"]
            },
            {
                "pergunta": "4. O boxplot pode ajudar a identificar:",
                "opcoes": ["Outliers", "Reagentes", "Solventes", "Equipamentos"]
            },
            {
                "pergunta": "5. Um histograma mostra principalmente:",
                "opcoes": ["Distribuição dos valores", "Estrutura molecular", "Número de PCs", "RMSEP"]
            }
        ]

        gabarito = [
            "Uma amostra",
            "Uma variável",
            "Avaliar distribuição dos dados",
            "Outliers",
            "Distribuição dos valores"
        ]

        quiz(
            "Explorador de Dados",
            perguntas,
            gabarito,
            proxima_fase=2,
            badge="Explorador Químico"
        )


# =========================
# FASE 2 - PRÉ-PROCESSAMENTO
# =========================

elif fase_escolhida == 2:

    st.header("🔬 Fase 2 — Pré-processamento")

    dados, X = carregar_dados("preprocessamento")

    if X is not None:

        metodo = st.selectbox(
            "Escolha o pré-processamento",
            ["Nenhum", "Centralização", "Autoescalamento", "SNV"]
        )

        Xproc = preprocessar(X, metodo)

        st.subheader("Dados originais")
        st.dataframe(X.head())

        st.subheader("Dados processados")
        st.dataframe(Xproc.head())

        variavel = st.selectbox("Escolha uma variável", X.columns)

        st.subheader("Comparação - Histograma")
        fig, ax = plt.subplots()
        ax.hist(X[variavel], alpha=0.5, label="Original")
        ax.hist(Xproc[variavel], alpha=0.5, label="Processado")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Comparação - Boxplot")
        fig2, ax2 = plt.subplots()
        ax2.boxplot([X[variavel], Xproc[variavel]])
        ax2.set_xticklabels(["Original", "Processado"])
        st.pyplot(fig2)

        if metodo == "Centralização":
            st.info("A centralização desloca os dados para média zero.")

        elif metodo == "Autoescalamento":
            st.info("O autoescalamento coloca as variáveis em escala comparável.")

        elif metodo == "SNV":
            st.info("O SNV reduz variações de espalhamento entre amostras.")

        perguntas = [
            {
                "pergunta": "1. A centralização na média faz com que:",
                "opcoes": ["A média fique zero", "O RMSEP aumente", "As amostras desapareçam", "A PCA seja proibida"]
            },
            {
                "pergunta": "2. O autoescalamento ajusta:",
                "opcoes": ["Média e desvio padrão", "Somente a cor", "Somente o nome das amostras", "A massa molar"]
            },
            {
                "pergunta": "3. O SNV é muito usado em:",
                "opcoes": ["Espectroscopia", "Titulação clássica apenas", "Pesagem simples", "Filtração"]
            },
            {
                "pergunta": "4. O pré-processamento deve ser escolhido com base:",
                "opcoes": ["No problema analítico", "Na sorte", "Na ordem alfabética", "Na cor do gráfico"]
            },
            {
                "pergunta": "5. Um pré-processamento inadequado pode:",
                "opcoes": ["Alterar a interpretação dos padrões", "Melhorar sempre", "Eliminar a química", "Impedir qualquer gráfico"]
            }
        ]

        gabarito = [
            "A média fique zero",
            "Média e desvio padrão",
            "Espectroscopia",
            "No problema analítico",
            "Alterar a interpretação dos padrões"
        ]

        quiz(
            "Pré-processamento",
            perguntas,
            gabarito,
            proxima_fase=3,
            badge="Mestre do Pré-processamento"
        )


# =========================
# FASE 3 - PCA
# =========================

elif fase_escolhida == 3:

    st.header("📈 Fase 3 — Reconhecimento de Padrões")

    dados, X = carregar_dados("pca")

    if X is not None:

        metodo = st.selectbox(
            "Pré-processamento para PCA",
            ["Nenhum", "Centralização", "Autoescalamento", "SNV"]
        )

        Xproc = preprocessar(X, metodo)

        pca = PCA()
        scores = pca.fit_transform(Xproc)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100
        acumulada = np.cumsum(explained)

        st.subheader("Variância explicada")
        tabela_var = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(explained))],
            "Variância (%)": explained,
            "Variância acumulada (%)": acumulada
        })

        st.dataframe(tabela_var.head(10))

        st.subheader("Scree Plot")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(explained)+1), explained, marker="o")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Variância explicada (%)")
        st.pyplot(fig)

        st.subheader("Scores PC1 × PC2")
        fig2, ax2 = plt.subplots()
        ax2.scatter(scores[:, 0], scores[:, 1])

        for i in range(scores.shape[0]):
            ax2.text(scores[i, 0], scores[i, 1], str(i+1))

        ax2.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax2.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        st.pyplot(fig2)

        st.subheader("Loadings PC1 × PC2")
        fig3, ax3 = plt.subplots()
        ax3.scatter(loadings[:, 0], loadings[:, 1])

        for i, var in enumerate(X.columns):
            ax3.text(loadings[i, 0], loadings[i, 1], var)

        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        st.pyplot(fig3)

        st.info(
            f"PC1 explica {explained[0]:.2f}% da variância. "
            f"PC2 explica {explained[1]:.2f}%. "
            f"Juntas, PC1 e PC2 explicam {explained[0] + explained[1]:.2f}%."
        )

        if perfil == "Mestre":
            st.subheader("Tabela completa de loadings")
            tabela_loadings = pd.DataFrame(
                loadings,
                index=X.columns,
                columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
            )
            st.dataframe(tabela_loadings)
        
        st.markdown("---")
        st.subheader("🌳 HCA — Análise Hierárquica de Agrupamentos")

        metodo_ligacao = st.selectbox(
            "Método de ligação",
            ["ward", "complete", "average", "single"]
        )

        metrica = st.selectbox(
            "Métrica de distância",
            ["euclidean", "cityblock", "cosine"]
        )

        if metodo_ligacao == "ward" and metrica != "euclidean":
            st.warning("O método Ward é recomendado apenas com distância Euclidiana. A métrica foi ajustada para Euclidean.")
            metrica = "euclidean"

        distancias = pdist(Xproc, metric=metrica)
        Z = linkage(distancias, method=metodo_ligacao)

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        dendrogram(
            Z,
            labels=[str(i+1) for i in range(Xproc.shape[0])],
            ax=ax4
        )

        ax4.set_xlabel("Amostras")
        ax4.set_ylabel("Distância")
        ax4.set_title("Dendrograma")
        st.pyplot(fig4)

        st.info(
            "O dendrograma permite visualizar a similaridade entre as amostras. "
            "Amostras que se unem em menores distâncias são mais semelhantes."
        )

        perguntas = [
            {
                "pergunta": "1. PCA é uma técnica usada principalmente para:",
                "opcoes": ["Explorar padrões", "Medir pH", "Calcular massa molar", "Fazer titulação"]
            },
            {
                "pergunta": "2. O gráfico de scores mostra:",
                "opcoes": ["Relação entre amostras", "Somente reagentes", "Somente variáveis", "RMSEP"]
            },
            {
                "pergunta": "3. O gráfico de loadings ajuda a interpretar:",
                "opcoes": ["Contribuição das variáveis", "Cor da solução", "Número de alunos", "Temperatura ambiente"]
            },
            {
                "pergunta": "4. No HCA, o dendrograma mostra:",
                "opcoes": ["Similaridade entre amostras", "Somente calibração", "A fórmula molecular", "A concentração real"]
            },
            {
                "pergunta": "5. Amostras que se unem em menor distância no dendrograma são:",
                "opcoes": ["Mais semelhantes", "Sempre erradas", "Mais concentradas obrigatoriamente", "Excluídas automaticamente"]
            }
        ]

        gabarito = [
            "Explorar padrões",
            "Relação entre amostras",
            "Contribuição das variáveis",
            "Similaridade entre amostras",
            "Mais semelhantes"
        ]

        quiz(
            "Reconhecimento de Padrões",
            perguntas,
            gabarito,
            proxima_fase=4,
            badge="Detetive Quimiométrico"
        )


# =========================
# FASES FUTURAS
# =========================

elif fase_escolhida == 4:

    st.header("🎯 Fase 4 — Classificação")
    st.info("Módulo em desenvolvimento. Será liberado na próxima versão.")


elif fase_escolhida == 5:

    st.header("📐 Fase 5 — Calibração")
    st.info("Módulo em desenvolvimento. Será liberado na próxima versão.")


elif fase_escolhida == 6:

    st.header("🔓 Fase Final — Escape Room")
    st.info("Módulo em desenvolvimento. Será liberado na próxima versão.")


# =========================
# BADGES
# =========================

st.sidebar.markdown("---")
st.sidebar.subheader("🏅 Badges conquistadas")

if st.session_state.badges:
    for b in st.session_state.badges:
        st.sidebar.write(f"🏅 {b}")
else:
    st.sidebar.write("Nenhuma badge ainda.")