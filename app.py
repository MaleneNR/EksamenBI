# ---------------------------------------------------------
# Home Page / Dashboard for boligdata
# ---------------------------------------------------------
# Denne side viser et flot dashboard med forskellige grafer
# baseret på datasættet DKHousingPrices.parquet.
# ---------------------------------------------------------

from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Streamlit sideopsætning
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bolig Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #666666;
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    possible_paths = [
        Path("DKHousingPrices.parquet"),
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_parquet(path)

    st.error("Kunne ikke finde DKHousingPrices.parquet. Tjek filstien.")
    st.stop()

df = load_data()

# Her laver vi filtered_df, så resten af koden virker
filtered_df = df.copy()

# ---------------------------------------------------------
# Sidebar - vælg hvilke grafer der skal vises
# ---------------------------------------------------------
st.sidebar.header(" Vælg grafer")

selected_plots = st.sidebar.multiselect(
    "Hvilke visualiseringer vil du se?",
    [
        "Fordeling af pris pr. m²",
        "Størrelse vs pris",
        "Pris pr. m² efter region",
        "Boligtyper (antal)",
        "Pris pr. m² efter boligtype",
        "Korrelationsmatrix",
        "Boligsegmenter (KMeans)"
    ],
    default=[
        "Fordeling af pris pr. m²",
        "Størrelse vs pris"
    ]
)

# ---------------------------------------------------------
# Header / introduktion
# ---------------------------------------------------------
st.markdown('<div class="main-title"> Bolig Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Et interaktivt overblik over boligdata i Danmark med fokus på pris, størrelse, region og segmentering.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Nøgletal
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Antal boliger", f"{len(filtered_df):,}".replace(",", "."))

with col2:
    if "purchase_price" in filtered_df.columns:
        avg_price = filtered_df["purchase_price"].mean()
        st.metric("Gns. boligpris", f"{avg_price:,.0f} kr.".replace(",", "."))

with col3:
    if "sqm_price" in filtered_df.columns:
        avg_sqm_price = filtered_df["sqm_price"].mean()
        st.metric("Gns. pris pr. m²", f"{avg_sqm_price:,.0f} kr.".replace(",", "."))

with col4:
    if "sqm" in filtered_df.columns:
        avg_sqm = filtered_df["sqm"].mean()
        st.metric("Gns. størrelse", f"{avg_sqm:,.1f} m²".replace(",", "."))

st.divider()

# ---------------------------------------------------------
# Preview af data
# ---------------------------------------------------------
with st.expander("Se de første rækker i datasættet"):
    st.dataframe(filtered_df.head(), use_container_width=True)

st.title("Interaktiv graf 📊")

# Brugeren vælger kolonner
x_col = st.selectbox("Vælg X-akse", df.columns)
y_col = st.selectbox("Vælg Y-akse", df.columns)

# Plot
fig, ax = plt.subplots()
ax.scatter(df[x_col], df[y_col])

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)

st.pyplot(fig)

# ---------------------------------------------------------
# Dynamisk visning af grafer
# ---------------------------------------------------------

if "Fordeling af pris pr. m²" in selected_plots:
    st.markdown('<div class="section-title">Fordeling af pris pr. m²</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(filtered_df["sqm_price"].dropna(), bins=30)
    ax.set_xlabel("Pris pr. m²")
    ax.set_ylabel("Antal boliger")
    ax.set_title("Distribution af pris pr. m²")
    st.pyplot(fig)

if "Størrelse vs pris" in selected_plots:
    st.markdown('<div class="section-title">Sammenhæng mellem størrelse og pris</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(filtered_df["sqm"], filtered_df["purchase_price"], alpha=0.6)
    ax.set_xlabel("Boligstørrelse (m²)")
    ax.set_ylabel("Boligpris")
    ax.set_title("Sammenhæng mellem boligstørrelse og boligpris")
    st.pyplot(fig)

if "Pris pr. m² efter region" in selected_plots:
    st.markdown('<div class="section-title">Pris pr. m² efter region</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    filtered_df.boxplot(column="sqm_price", by="region", ax=ax)
    plt.suptitle("")
    ax.set_title("Pris pr. m² fordelt på region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Pris pr. m²")
    plt.xticks(rotation=45)
    st.pyplot(fig)

if "Boligtyper (antal)" in selected_plots:
    st.markdown('<div class="section-title">Antal boliger efter boligtype</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    filtered_df["house_type"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Boligtype")
    ax.set_ylabel("Antal")
    ax.set_title("Fordeling af boligtyper")
    plt.xticks(rotation=45)
    st.pyplot(fig)

if "Pris pr. m² efter boligtype" in selected_plots:
    st.markdown('<div class="section-title">Pris pr. m² efter boligtype</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    filtered_df.boxplot(column="sqm_price", by="house_type", ax=ax)
    plt.suptitle("")
    ax.set_title("Pris pr. m² efter boligtype")
    ax.set_xlabel("Boligtype")
    ax.set_ylabel("Pris pr. m²")
    plt.xticks(rotation=45)
    st.pyplot(fig)

if "Korrelationsmatrix" in selected_plots:
    st.markdown('<div class="section-title">Korrelationsmatrix</div>', unsafe_allow_html=True)

    numeric_df = filtered_df.select_dtypes(include="number").copy()

    if "year_build" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["year_build"])

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        cax = ax.imshow(corr, cmap="coolwarm")
        fig.colorbar(cax)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        ax.set_title("Korrelationsmatrix for numeriske variable")
        st.pyplot(fig)
    else:
        st.warning("Der er ikke nok numeriske kolonner til at beregne en korrelationsmatrix.")

if "Boligsegmenter (KMeans)" in selected_plots:
    st.markdown('<div class="section-title">Boligsegmenter</div>', unsafe_allow_html=True)

    st.write(
        "Her grupperes boliger i segmenter ud fra størrelse, pris og antal rum ved hjælp af KMeans clustering."
    )

    cluster_columns = ["sqm", "purchase_price", "no_rooms"]
    available_cols = [col for col in cluster_columns if col in filtered_df.columns]

    if len(available_cols) == 3:
        cluster_data = filtered_df[cluster_columns].dropna()

        if len(cluster_data) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

            df_cluster = cluster_data.copy()
            df_cluster["cluster"] = kmeans.fit_predict(cluster_data)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                df_cluster["sqm"],
                df_cluster["purchase_price"],
                c=df_cluster["cluster"],
                cmap="viridis",
                alpha=0.7
            )

            ax.set_xlabel("Størrelse (m²)")
            ax.set_ylabel("Pris")
            ax.set_title("Boligsegmenter baseret på størrelse, pris og antal rum")
            st.pyplot(fig)

            st.dataframe(df_cluster.head(), use_container_width=True)
        else:
            st.warning("Der er for få rækker til at lave clustering.")
    else:
        st.warning("De nødvendige kolonner til clustering findes ikke i datasættet.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("Udviklet i Streamlit • Boligdata dashboard")