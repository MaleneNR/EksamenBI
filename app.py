import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


#Titel

st.set_page_config(page_title="Prediction", layout="wide")

st.title("Forudsigelse af bolig")
st.markdown("Indtast boligdata og få estimeret pris, segment og kategori.")


#Load data
@st.cache_data
def load_data():
    path = Path("DKHousingPrices.parquet")
    df = pd.read_parquet(path)

    # Opret house_age (VIGTIGT)
    if "year_build" in df.columns:
        df["house_age"] = 2024 - df["year_build"]

    return df

df = load_data()


#Prediction + interaktiv grafer
st.divider()
st.markdown('<div class="section-title">Forudsigelse & interaktiv grafer</div>', unsafe_allow_html=True)


#Forbered data (med house_age)
if "year_build" in df.columns:
    df["house_age"] = 2024 - df["year_build"]


#Load models (med dropna)

@st.cache_resource
def load_models(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier

    #Regression
    df_reg = df[["sqm", "house_age", "sqm_price"]].dropna()
    model = LinearRegression().fit(df_reg[["sqm", "house_age"]], df_reg["sqm_price"])

    #Clustering
    df_cluster = df[["sqm", "purchase_price", "no_rooms"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df_cluster)

    #Classification (valgfri)
    tree = None
    if "category" in df.columns:
        df_class = df[["sqm", "no_rooms", "year_build", "category"]].dropna()
        tree = DecisionTreeClassifier().fit(
            df_class[["sqm", "no_rooms", "year_build"]],
            df_class["category"]
        )

    return model, kmeans, tree

model, kmeans, tree = load_models(df)

#INPUT FRA BRUGER

col1, col2, col3 = st.columns(3)

with col1:
    sqm = st.slider("Størrelse (m²)", 50, 300, 120)

with col2:
    rooms = st.slider("Antal rum", 1, 8, 3)

with col3:
    age = st.slider("Boligens alder", 0, 200, 20)

year = 2024 - age

#Prediction
if st.button("Beregn forudsigelse"):

    #Regression
    input_df = pd.DataFrame([[sqm, age]], columns=["sqm", "house_age"])
    y_pred = model.predict(input_df)
    total_price = y_pred[0] * sqm

    st.subheader("Estimeret pris")
    st.success(f"{int(total_price):,} kr.".replace(",", "."))

    #Clustering
    cluster_df = pd.DataFrame(
        [[sqm, total_price, rooms]],
        columns=["sqm", "purchase_price", "no_rooms"]
    )

    cluster = kmeans.predict(cluster_df)[0]

    labels = {
        0: "Lavpris",
        1: "Mellemklasse",
        2: "LÅGsus med 'åg'"
    }

    st.subheader("Segment")
    st.info(labels.get(cluster, f"Cluster {cluster}"))

    #Classification
    if tree is not None:
        class_df = pd.DataFrame(
            [[sqm, rooms, year]],
            columns=["sqm", "no_rooms", "year_build"]
        )

        category = tree.predict(class_df)[0]

        st.subheader("Kategori")
        st.warning(category)

 
       # VISUALISERING AF INDTASTEDE BOLIG
    st.subheader("Din bolig i datasættet")

    import matplotlib.ticker as mticker

    fig, ax = plt.subplots()

    ax.scatter(df["sqm"], df["purchase_price"], alpha=0.3)
    ax.scatter(sqm, total_price, s=100)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "."))
    )

    ax.set_xlabel("Størrelse (m²)")
    ax.set_ylabel("Pris")

    st.pyplot(fig)


# Interaktiv graf
st.markdown("""
<div style="
    background-color: #f5f7fa;
    padding: 5px;
    border-radius: 5px;
    margin-top: 5px;
">
""", unsafe_allow_html=True)

st.subheader("Udforsk selv vores data")

x_col = st.selectbox("Vælg X-akse", df.columns, key="x_pred")
y_col = st.selectbox("Vælg Y-akse", df.columns, key="y_pred")

fig, ax = plt.subplots()
ax.scatter(df[x_col], df[y_col], alpha=0.5)

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "."))
)

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)


st.pyplot(fig)

