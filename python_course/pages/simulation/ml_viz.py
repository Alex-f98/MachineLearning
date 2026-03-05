# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    roc_auc_score
)

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="ML Visualizer",
    layout="wide",
    page_icon="🧠"
)

# ---------- Defaults ----------
DEFAULTS = {
    "algoritmo": "Regresión logística",
    "dataset": "Lineal separable",
    "n_muestras": 500,
    "ruido_features": 0.20,   # desviación estándar de ruido gaussiano en X
    "flip_y": 0.02,           # fracción de etiquetas volteadas
    "test_size": 0.25,
    "semilla": 42,

    # Por modelo
    "C": 1.0,
    "max_iter": 500,
    "gamma": 0.10,
    "k": 15,
    "max_depth": 5,
}

ALGORITHMS_INFO = {
    "Regresión logística": {
        "desc": (
            "Modelo lineal para clasificación binaria/multiclase. "
            "Aprende un hiperplano que separa clases maximizando la verosimilitud."
        ),
        "cuando": "Bueno con datos aproximadamente linealmente separables y con regularización L2.",
        "hipers": ["C (inversa de la regularización)", "max_iter"]
    },
    "SVM lineal": {
        "desc": (
            "Máquinas de Vectores de Soporte con kernel lineal. "
            "Busca el margen máximo entre clases."
        ),
        "cuando": "Útil cuando las clases son casi lineales y hay alta dimensionalidad.",
        "hipers": ["C (controla margen vs errores)"]
    },
    "SVM RBF": {
        "desc": (
            "SVM con kernel gaussiano (RBF). "
            "Captura fronteras no lineales mediante mapeo implícito a alta dimensión."
        ),
        "cuando": "Excelente para fronteras complejas cuando el dataset no es lineal.",
        "hipers": ["C", "gamma (alcance del kernel)"]
    },
    "KNN": {
        "desc": (
            "Clasificador de vecinos más cercanos. "
            "Predice según las clases de los k vecinos más próximos."
        ),
        "cuando": "Bueno como baseline y en datos con estructuras locales.",
        "hipers": ["k (número de vecinos)"]
    },
    "Árbol de decisión": {
        "desc": (
            "Modelo basado en divisiones jerárquicas del espacio de características."
        ),
        "cuando": "Útil por interpretabilidad y para capturar relaciones no lineales.",
        "hipers": ["max_depth (profundidad máxima)"]
    },
}

# ---------- Helpers de estado ----------
def ensure_session_state():
    if "initialized" not in st.session_state:
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state["initialized"] = True

def reset_params():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

ensure_session_state()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("ML viz")

    st.subheader("⚙️ Algoritmo")
    algoritmo = st.selectbox(
        "Seleccioná un algoritmo",
        list(ALGORITHMS_INFO.keys()),
        index=list(ALGORITHMS_INFO.keys()).index(st.session_state["algoritmo"]),
        key="algoritmo"
    )

    st.subheader("📦 Dataset")
    dataset_name = st.selectbox(
        "Tipo de dataset",
        ["Lineal separable", "Lunas", "Círculos", "Blobs"],
        index=["Lineal separable", "Lunas", "Círculos", "Blobs"].index(st.session_state["dataset"]),
        key="dataset"
    )

    n_muestras = st.slider("Cantidad de muestras", 100, 3000, st.session_state["n_muestras"], step=50, key="n_muestras")
    test_size = st.slider("Proporción test", 0.1, 0.5, st.session_state["test_size"], step=0.05, key="test_size")
    semilla = st.number_input("Semilla aleatoria", value=st.session_state["semilla"], step=1, key="semilla")

    st.markdown("---")
    st.subheader("🧪 Ruido")
    ruido_features = st.slider("Ruido gaussiano en features (σ)", 0.0, 1.0, st.session_state["ruido_features"], 0.01, key="ruido_features")
    flip_y = st.slider("Fracción de etiquetas volteadas", 0.0, 0.5, st.session_state["flip_y"], 0.01, key="flip_y")

    st.markdown("---")
    st.subheader("🛠️ Hiperparámetros")

    # Controles dinámicos por algoritmo
    if algoritmo in ["Regresión logística", "SVM lineal", "SVM RBF"]:
        C = st.slider("C (inversa de regularización)", 0.01, 10.0, float(st.session_state["C"]), 0.01, key="C")
    if algoritmo == "Regresión logística":
        max_iter = st.slider("Max iteraciones", 100, 2000, int(st.session_state["max_iter"]), 50, key="max_iter")
    if algoritmo == "SVM RBF":
        gamma = st.slider("gamma (kernel RBF)", 0.001, 2.0, float(st.session_state["gamma"]), 0.001, key="gamma")
    if algoritmo == "KNN":
        k = st.slider("k (vecinos)", 1, 50, int(st.session_state["k"]), 1, key="k")
    if algoritmo == "Árbol de decisión":
        max_depth = st.slider("Profundidad máxima", 1, 20, int(st.session_state["max_depth"]), 1, key="max_depth")

    st.markdown("---")
    st.button("🔄 Resetear parámetros", on_click=reset_params)

    st.markdown("---")
    st.subheader("ℹ️ Información del algoritmo")
    info = ALGORITHMS_INFO[algoritmo]
    st.markdown(f"**Descripción:** {info['desc']}")
    st.markdown(f"**Cuándo usarlo:** {info['cuando']}")
    st.markdown("**Hiperparámetros clave:**")
    for h in info["hipers"]:
        st.markdown(f"- {h}")

# ---------- Generación de datos ----------
def generar_datos(nombre, n, ruido_x, flip_y, seed):
    rng = np.random.RandomState(seed)
    if nombre == "Lineal separable":
        X, y = make_classification(
            n_samples=n, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, class_sep=2.0, flip_y=flip_y, random_state=seed
        )
    elif nombre == "Lunas":
        X, y = make_moons(n_samples=n, noise=0.15, random_state=seed)
        # Aplicamos flip_y manual:
        if flip_y > 0:
            m = int(flip_y * n)
            idx = rng.choice(n, m, replace=False)
            y[idx] = 1 - y[idx]
    elif nombre == "Círculos":
        X, y = make_circles(n_samples=n, noise=0.1, factor=0.5, random_state=seed)
        if flip_y > 0:
            m = int(flip_y * n)
            idx = rng.choice(n, m, replace=False)
            y[idx] = 1 - y[idx]
    else:  # Blobs
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=2.0, random_state=seed)
        if flip_y > 0:
            m = int(flip_y * n)
            idx = rng.choice(n, m, replace=False)
            y[idx] = 1 - y[idx]

    # Ruido gaussiano en features
    if ruido_x > 0:
        X = X + rng.normal(0, ruido_x, size=X.shape)

    return X, y

X, y = generar_datos(dataset_name, n_muestras, ruido_features, flip_y, semilla)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=semilla, stratify=y
)

# ---------- Modelo ----------
def construir_modelo(nombre):
    if nombre == "Regresión logística":
        clf = LogisticRegression(C=st.session_state["C"], max_iter=st.session_state["max_iter"])
        model = make_pipeline(StandardScaler(), clf)
        proba_ok = True  # tiene predict_proba
    elif nombre == "SVM lineal":
        clf = SVC(kernel="linear", C=st.session_state["C"], probability=True)  # probability=True para AUC
        model = make_pipeline(StandardScaler(), clf)
        proba_ok = True
    elif nombre == "SVM RBF":
        clf = SVC(kernel="rbf", C=st.session_state["C"], gamma=st.session_state["gamma"], probability=True)
        model = make_pipeline(StandardScaler(), clf)
        proba_ok = True
    elif nombre == "KNN":
        clf = KNeighborsClassifier(n_neighbors=st.session_state["k"])
        model = make_pipeline(StandardScaler(), clf)
        proba_ok = True
    else:  # Árbol de decisión
        clf = DecisionTreeClassifier(max_depth=st.session_state["max_depth"], random_state=st.session_state["semilla"])
        model = clf
        proba_ok = True

    return model, proba_ok

model, proba_ok = construir_modelo(algoritmo)
model.fit(X_train, y_train)

# Predicciones y métricas
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

auc = None
try:
    if proba_ok:
        y_score = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
except Exception:
    auc = None

cm = confusion_matrix(y_test, y_pred)

# ---------- Visualización principal ----------
st.markdown("### Resultado del modelo")
plot_col, metrics_col = st.columns([4, 1.4], gap="medium")

def plot_decision_boundary(ax, model, X, y, padding=0.7, grid_steps=300):
    # Generamos malla
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )
    XY = np.c_[xx.ravel(), yy.ravel()]

    # Scores para colorear el fondo
    try:
        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(XY)[:, 1]
        else:
            Z = model.decision_function(XY)
    except Exception:
        # fallback a predicción dura
        Z = model.predict(XY)
    Z = Z.reshape(xx.shape)

    # Fondo con la frontera
    cmap_bg = plt.cm.RdBu
    ax.contourf(xx, yy, Z, levels=25, cmap=cmap_bg, alpha=0.35)

    # Puntos
    colors = np.array(["#1f77b4", "#d62728"])  # azul, rojo
    ax.scatter(X[:, 0], X[:, 1], c=colors[y], s=24, edgecolor="white", linewidth=0.6, alpha=0.9)

    ax.set_xlabel("X₁")
    ax.set_ylabel("X₂")
    ax.set_title(f"Frontera de decisión — {algoritmo}  |  Dataset: {dataset_name}")
    ax.grid(True, alpha=0.15)

with plot_col:
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_decision_boundary(ax, model, X_train, y_train)
    st.pyplot(fig, use_container_width=True)

with metrics_col:
    st.markdown("#### Métricas")
    st.write(f"**Accuracy:** {acc*100:.2f}%")
    st.write(f"**Precision:** {prec*100:.2f}%")
    st.write(f"**Recall:** {rec*100:.2f}%")
    if auc is not None:
        st.write(f"**AUC:** {auc*100:.2f}%")
    else:
        st.write("**AUC:** N/D")

    st.markdown("---")
    st.markdown("**Matriz de confusión**")
    fig_cm, ax_cm = plt.subplots(figsize=(3.8, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")
    st.pyplot(fig_cm, use_container_width=True)


st.info("idea original: https://www.youtube.com/watch?v=W-aZ0ey64Ms")
