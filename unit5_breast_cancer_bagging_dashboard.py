"""Unit 5: Breast Cancer Classification — Decision Tree vs Bagging Dashboard

Run:
    pip install streamlit pandas numpy plotly scikit-learn
    streamlit run unit5_breast_cancer_bagging_dashboard.py

Upload Kaggle's Breast Cancer Wisconsin (Diagnostic) data.csv, or use the
built-in scikit-learn copy of the same diagnostic dataset as a fallback.
This is an educational machine-learning demonstration, NOT a medical tool.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Unit 5 | Breast Cancer Model Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling: editorial clinical-lab look (different from Units 3 & 4) ----------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f6f5f2; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1450px; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #252b31; letter-spacing: -0.035em; }
      .hero { background: linear-gradient(115deg,#252b31 0%,#46323e 58%,#743f55 100%);
              color: #fff; border-radius: 20px; padding: 1.65rem 1.9rem; margin-bottom: 1.25rem; }
      .hero-kicker { color: #eacbd7; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
      .hero h1 { color: white; font-size: clamp(2rem,4vw,3.2rem); margin: .35rem 0 .4rem; }
      .hero p { color: #eee4e9; margin: 0; max-width: 800px; }
      .metric-card { background: white; border: 1px solid #e7e4e0; border-radius: 15px; padding: 1rem 1.1rem; min-height: 105px; }
      .metric-label { color: #777b80; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }
      .metric-value { color: #292e34; font-family: 'Manrope',sans-serif; font-size: 1.9rem; font-weight: 800; margin-top: .22rem; }
      .metric-note { color: #85888c; font-size: .75rem; margin-top: .1rem; }
      .section-note { color: #72767b; font-size: .9rem; }
      .pill { display:inline-block; background:#f2e7eb; color:#753d55; padding:.25rem .6rem; border-radius:99px; font-size:.72rem; font-weight:700; }
      .callout { background:#fff; border-left:4px solid #9a5c73; padding:.9rem 1rem; border-radius:8px; color:#45494e; }
      div[data-testid="stTabs"] button { font-weight: 700; }
      [data-testid="stSidebar"] { background: #eeece8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data loading and normalization ----------
EXPECTED_TARGETS = {"M": 1, "B": 0, "malignant": 1, "benign": 0}


def load_builtin_dataset() -> pd.DataFrame:
    """Return sklearn's copy in the familiar Kaggle/Wisconsin column format."""
    bunch = load_breast_cancer()
    frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    # sklearn target: 0=malignant, 1=benign. Convert to Kaggle convention M/B.
    frame.insert(0, "diagnosis", np.where(bunch.target == 0, "M", "B"))
    frame.insert(0, "id", np.arange(1, len(frame) + 1))
    # Convert sklearn names to the common Kaggle naming convention.
    rename = {}
    for col in frame.columns:
        new = col.replace("mean ", "").replace(" ", "_")
        new = new.replace("_error", "_se")
        if new.startswith("worst_"):
            pass
        elif col.endswith(" error"):
            new = col.replace(" error", "_se").replace(" ", "_")
        rename[col] = new
    frame = frame.rename(columns=rename)
    return frame


def prepare_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop common non-feature columns when present.
    for col in list(df.columns):
        if col.lower().startswith("unnamed") or col.lower() in {"id", "index"}:
            df = df.drop(columns=col)
    target_candidates = [c for c in df.columns if c.lower() in {"diagnosis", "target", "class", "label"}]
    if not target_candidates:
        raise ValueError("Could not find the target column. Expected a column named 'diagnosis'.")
    target_col = target_candidates[0]
    y_raw = df.pop(target_col)
    if y_raw.isna().any():
        keep = y_raw.notna()
        df, y_raw = df.loc[keep], y_raw.loc[keep]
    if pd.api.types.is_numeric_dtype(y_raw):
        values = set(pd.Series(y_raw).dropna().unique().tolist())
        if values <= {0, 1}:
            y = y_raw.astype(int)
            target_note = "Numeric binary target (0/1)"
        else:
            raise ValueError("Numeric target must contain only 0 and 1, or use M/B labels.")
    else:
        labels = y_raw.astype(str).str.strip().str.lower()
        mapped = labels.map(EXPECTED_TARGETS)
        if mapped.isna().any():
            unknown = sorted(labels[mapped.isna()].unique().tolist())
            raise ValueError(f"Unrecognized diagnosis labels: {unknown}. Use M/B or malignant/benign.")
        y = mapped.astype(int)
        target_note = "Diagnosis encoded as M=1 (malignant), B=0 (benign)"
    X = df.apply(pd.to_numeric, errors="coerce")
    # Remove completely empty or non-numeric columns, then median-impute numeric missing values.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns were found in the CSV.")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    valid = y.index.intersection(X.index)
    X, y = X.loc[valid], y.loc[valid]
    if y.nunique() != 2:
        raise ValueError("The dataset must contain both benign and malignant classes.")
    return X.reset_index(drop=True), y.reset_index(drop=True), raw.copy(), target_note


def make_bagging(n_estimators: int, random_state: int) -> BaggingClassifier:
    tree = DecisionTreeClassifier(random_state=random_state)
    params = inspect.signature(BaggingClassifier).parameters
    if "estimator" in params:
        return BaggingClassifier(estimator=tree, n_estimators=n_estimators,
                                 random_state=random_state, n_jobs=-1)
    return BaggingClassifier(base_estimator=tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=-1)


def positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    pos_index = classes.index(1)
    return probs[:, pos_index]


def metric_pack(y_true, predictions, probabilities, threshold: float) -> Dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision (malignant)": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "Recall (malignant)": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "F1 (malignant)": f1_score(y_true, predictions, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

# ---------- Sidebar controls ----------
st.sidebar.markdown("## 🧬 Model Lab")
st.sidebar.caption("UNIT 5 · DECISION TREE + BAGGING")
upload = st.sidebar.file_uploader("Upload breast cancer data.csv", type=["csv"])
source = "Uploaded CSV" if upload is not None else "Built-in Wisconsin diagnostic dataset"
if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    raw_df = load_builtin_dataset()

try:
    X, y, original_df, target_note = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Dataset issue: {exc}")
    st.info("Expected Kaggle Breast Cancer Wisconsin (Diagnostic) data.csv with a diagnosis column containing M and B.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Experiment settings")
test_size = st.sidebar.slider("Test-set proportion", 0.15, 0.40, 0.20, 0.05)
n_estimators = st.sidebar.slider("Number of trees in Bagging", 10, 200, 100, 10)
threshold = st.sidebar.slider("Malignant probability threshold", 0.10, 0.90, 0.50, 0.05)
seed = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
run_cv = st.sidebar.checkbox("Run repeated 5-fold cross-validation", value=True)

# Split and fit each time the controls change.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(seed), stratify=y
)
tree_model = DecisionTreeClassifier(random_state=int(seed))
bag_model = make_bagging(int(n_estimators), int(seed))
tree_model.fit(X_train, y_train)
bag_model.fit(X_train, y_train)

p_tree = positive_probability(tree_model, X_test)
p_bag = positive_probability(bag_model, X_test)
pred_tree = (p_tree >= threshold).astype(int)
pred_bag = (p_bag >= threshold).astype(int)
metrics_tree = metric_pack(y_test, pred_tree, p_tree, threshold)
metrics_bag = metric_pack(y_test, pred_bag, p_bag, threshold)

# ---------- Header ----------
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">Machine Learning · Unit 5</div>
      <h1>Breast Cancer Classification</h1>
      <p>Can a group of decision trees make more stable predictions than one tree?
      Explore the effect of Bagging, compare model performance, and inspect the features.</p>
      <div style="margin-top:1rem"><span class="pill">{source}</span>
      <span style="color:#e6dce1;font-size:.78rem;margin-left:.5rem">Educational classification experiment</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dataset context KPIs
n_total = len(X)
n_malignant = int((y == 1).sum())
n_benign = int((y == 0).sum())
col1, col2, col3, col4 = st.columns(4)
with col1: metric_card("Samples", f"{n_total:,}", "Rows after basic cleaning")
with col2: metric_card("Numeric features", f"{X.shape[1]}", "Measurements used by the models")
with col3: metric_card("Benign", f"{n_benign:,}", f"{n_benign/n_total:.1%} of records")
with col4: metric_card("Malignant", f"{n_malignant:,}", f"{n_malignant/n_total:.1%} of records")

st.markdown("<div class='section-note'>Positive class: malignant (1). Negative class: benign (0). " + target_note + ".</div>", unsafe_allow_html=True)

# ---------- Navigation tabs ----------
tab_overview, tab_compare, tab_explain, tab_data = st.tabs([
    "01 · Overview", "02 · Model comparison", "03 · Explainability", "04 · Data studio"
])

with tab_overview:
    st.subheader("The experiment at a glance")
    st.markdown("<p class='section-note'>The project uses the Breast Cancer Wisconsin (Diagnostic) dataset and compares a single CART decision tree with a Bagging ensemble of decision trees.</p>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        counts = pd.DataFrame({"Diagnosis": ["Benign", "Malignant"], "Samples": [n_benign, n_malignant]})
        fig = px.pie(counts, names="Diagnosis", values="Samples", hole=.62,
                     title="Class distribution", color="Diagnosis",
                     color_discrete_map={"Benign": "#9bb9b0", "Malignant": "#a85d78"})
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(height=340, margin=dict(t=55,b=10,l=10,r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### What is being compared?")
        st.markdown("**Single Decision Tree**  \nOne tree learns decision rules by splitting the feature space. A deep tree can fit the training data very closely and become sensitive to small changes in the data.")
        st.markdown("**Bagging (100 trees by default)**  \nMany trees are trained on different bootstrap samples. Their votes are combined, often reducing variance and making predictions more stable.")
        st.markdown("<div class='callout'><b>Key idea:</b> different training samples create different trees; combining their votes can reduce the effect of any one tree's mistakes.</div>", unsafe_allow_html=True)
    st.markdown("#### Current run · test-set snapshot")
    a, b, c = st.columns(3)
    with a: metric_card("Decision Tree accuracy", f"{metrics_tree['Accuracy']:.1%}", "Current test split")
    with b: metric_card("Bagging accuracy", f"{metrics_bag['Accuracy']:.1%}", f"{n_estimators} trees · current test split")
    delta = metrics_bag["Accuracy"] - metrics_tree["Accuracy"]
    with c: metric_card("Accuracy change", f"{delta:+.1%}", "Bagging minus single tree")
    st.caption("These scores are recalculated for the selected CSV, split, random state, tree count, and probability threshold. Your results may differ from the repository's reported run.")

with tab_compare:
    st.subheader("Model comparison")
    st.markdown("<p class='section-note'>Use the probability threshold in the sidebar to change how aggressively the models label a case as malignant.</p>", unsafe_allow_html=True)
    metric_names = ["Accuracy", "Balanced accuracy", "Precision (malignant)", "Recall (malignant)", "F1 (malignant)", "ROC-AUC"]
    comparison = pd.DataFrame({
        "Metric": metric_names,
        "Decision Tree": [metrics_tree[m] for m in metric_names],
        "Bagging": [metrics_bag[m] for m in metric_names],
    })
    melted = comparison.melt(id_vars="Metric", var_name="Model", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group", text_auto=".2f",
                 color_discrete_map={"Decision Tree": "#9b9da3", "Bagging": "#9a5c73"},
                 title="Test-set metric comparison")
    fig.update_layout(yaxis=dict(range=[0,1.05], title="Score"), xaxis_title="", height=420,
                      legend_title_text="", margin=dict(t=60,b=80,l=20,r=20))
    fig.update_xaxes(tickangle=-18)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Confusion matrices")
    cm1, cm2 = st.columns(2)
    for container, model_name, pred in [(cm1, "Decision Tree", pred_tree), (cm2, f"Bagging ({n_estimators} trees)", pred_bag)]:
        with container:
            st.markdown(f"**{model_name}**")
            cm = confusion_matrix(y_test, pred, labels=[0,1])
            fig_cm = px.imshow(cm, text_auto=True, x=["Predicted benign", "Predicted malignant"],
                               y=["Actual benign", "Actual malignant"], color_continuous_scale="PuRd",
                               aspect="auto")
            fig_cm.update_layout(height=330, coloraxis_showscale=False,
                                 margin=dict(t=15,b=10,l=10,r=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("Top-left = correctly identified benign; bottom-right = correctly identified malignant; off-diagonal cells = classification errors.")

    st.markdown("#### ROC curves")
    fpr_t, tpr_t, _ = roc_curve(y_test, p_tree)
    fpr_b, tpr_b, _ = roc_curve(y_test, p_bag)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_t, y=tpr_t, mode="lines", name=f"Decision Tree (AUC {metrics_tree['ROC-AUC']:.3f})", line=dict(width=3, color="#8d9298")))
    fig_roc.add_trace(go.Scatter(x=fpr_b, y=tpr_b, mode="lines", name=f"Bagging (AUC {metrics_bag['ROC-AUC']:.3f})", line=dict(width=3, color="#9a5c73")))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line=dict(dash="dash", color="#b7b5b2")))
    fig_roc.update_layout(title="Receiver Operating Characteristic", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=400, margin=dict(t=55,b=25,l=25,r=20))
    st.plotly_chart(fig_roc, use_container_width=True)

    if run_cv:
        st.markdown("#### Repeated stratified cross-validation")
        with st.spinner("Running 25 cross-validation fits for each model..."):
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=int(seed))
            cv_tree = cross_val_score(DecisionTreeClassifier(random_state=int(seed)), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_bag = cross_val_score(make_bagging(int(n_estimators), int(seed)), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_frame = pd.DataFrame({
            "Model": ["Decision Tree", "Bagging"] * len(cv_tree),
            "Accuracy": np.r_[cv_tree, cv_bag],
        })
        fig_cv = px.box(cv_frame, x="Model", y="Accuracy", color="Model", points="all",
                        color_discrete_map={"Decision Tree": "#9b9da3", "Bagging": "#9a5c73"},
                        title="Accuracy across 25 stratified folds")
        fig_cv.update_layout(height=380, showlegend=False, margin=dict(t=55,b=20,l=20,r=20))
        st.plotly_chart(fig_cv, use_container_width=True)
        cv_a, cv_b = st.columns(2)
        with cv_a: metric_card("Decision Tree CV mean", f"{cv_tree.mean():.1%}", f"Std. dev. {cv_tree.std():.1%}")
        with cv_b: metric_card("Bagging CV mean", f"{cv_bag.mean():.1%}", f"Std. dev. {cv_bag.std():.1%}")
    else:
        st.info("Enable repeated cross-validation in the sidebar to compare stability across 25 folds.")

with tab_explain:
    st.subheader("Why did the model make its decision?")
    st.markdown("<p class='section-note'>Feature importance indicates which measurements the fitted model used most for splitting. It does not mean a feature causes cancer.</p>", unsafe_allow_html=True)
    imp_tree = pd.Series(tree_model.feature_importances_, index=X.columns, name="Decision Tree")
    bag_importances = np.mean([est.feature_importances_ for est in bag_model.estimators_], axis=0)
    imp_bag = pd.Series(bag_importances, index=X.columns, name="Bagging average")
    top_n = st.slider("Number of features to display", 5, min(20, X.shape[1]), min(12, X.shape[1]))
    top_features = imp_bag.sort_values(ascending=False).head(top_n).index
    imp_df = pd.concat([imp_tree.loc[top_features], imp_bag.loc[top_features]], axis=1).reset_index().rename(columns={"index":"Feature"})
    imp_long = imp_df.melt(id_vars="Feature", var_name="Model", value_name="Importance")
    fig_imp = px.bar(imp_long, x="Importance", y="Feature", color="Model", barmode="group", orientation="h",
                     color_discrete_map={"Decision Tree": "#9b9da3", "Bagging average": "#9a5c73"},
                     title="Most important features in the current fitted models")
    fig_imp.update_layout(height=max(420, top_n * 31),
                          yaxis={"categoryorder":"total ascending"}, margin=dict(t=55,b=20,l=20,r=20), legend_title_text="")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### How Bagging works")
    steps = st.columns(4)
    descriptions = [
        ("01 · Resample", "Create bootstrap samples by drawing training rows with replacement."),
        ("02 · Train", "Fit a decision tree independently on each bootstrap sample."),
        ("03 · Vote", "Each tree votes benign or malignant for a new sample."),
        ("04 · Combine", "The ensemble combines the tree votes into a final class probability."),
    ]
    for box, (heading, body) in zip(steps, descriptions):
        with box:
            st.markdown(f"**{heading}**")
            st.write(body)

    st.markdown("#### Feature distributions")
    selected_feature = st.selectbox("Choose a measurement", list(X.columns),
                                    index=list(X.columns).index(top_features[0]) if len(top_features) else 0)
    dist = pd.DataFrame({selected_feature: X[selected_feature], "Diagnosis": np.where(y == 1, "Malignant", "Benign")})
    fig_dist = px.violin(dist, x="Diagnosis", y=selected_feature, color="Diagnosis", box=True, points="outliers",
                         color_discrete_map={"Benign":"#8eafa5", "Malignant":"#a85d78"},
                         title=f"{selected_feature}: distribution by diagnosis")
    fig_dist.update_layout(height=390, showlegend=False, margin=dict(t=55,b=20,l=20,r=20))
    st.plotly_chart(fig_dist, use_container_width=True)

with tab_data:
    st.subheader("Dataset studio")
    st.markdown("<p class='section-note'>Inspect the uploaded data, basic quality checks, and numeric relationships.</p>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1: metric_card("Missing values after preparation", f"{int(X.isna().sum().sum())}", "Numeric missing values median-imputed")
    with d2: metric_card("Duplicate rows", f"{int(raw_df.duplicated().sum())}", "Count in original uploaded/built-in table")
    with d3: metric_card("Features retained", f"{X.shape[1]}", "ID and empty unnamed columns excluded")
    with st.expander("Preview prepared feature table", expanded=True):
        preview = X.copy()
        preview.insert(0, "diagnosis", np.where(y == 1, "M", "B"))
        st.dataframe(preview.head(30), use_container_width=True, height=360)
        st.download_button("Download prepared data as CSV", preview.to_csv(index=False).encode("utf-8"),
                           file_name="unit5_prepared_breast_cancer_data.csv", mime="text/csv")
    with st.expander("Descriptive statistics"):
        st.dataframe(X.describe().T, use_container_width=True)
    corr = X.corr(numeric_only=True)
    # Keep the heatmap useful and legible for this 30-feature dataset.
    st.markdown("#### Correlation heatmap")
    st.caption("Highly correlated measurements can carry overlapping information. Correlation does not imply causation.")
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                         title="Feature-to-feature Pearson correlation")
    fig_corr.update_layout(height=760, margin=dict(t=55,b=20,l=20,r=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------- Repository benchmark and safety note ----------
st.markdown("---")
st.subheader("Repository benchmark (reference values)")
st.markdown("The project's README reports the following results for its own experiment. They are shown separately from the live run above so they are not confused with scores calculated from the selected data and settings.")
ref = pd.DataFrame([
    {"Model":"Single Decision Tree", "Test accuracy":0.9005847953216374, "CV mean":0.930747, "CV std. dev.":0.025258},
    {"Model":"Bagging (100 trees)", "Test accuracy":0.9590643274853801, "CV mean":0.957476, "CV std. dev.":0.022555},
])
st.dataframe(ref.style.format({"Test accuracy":"{:.2%}", "CV mean":"{:.2%}", "CV std. dev.":"{:.2%}"}), use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="callout"><b>Important medical disclaimer:</b> This is a classroom machine-learning project using a historical benchmark dataset. It is not validated for clinical use, must not be used to diagnose or rule out cancer, and must not guide treatment decisions. A qualified healthcare professional must interpret medical findings.</div>
    """,
    unsafe_allow_html=True,
)
st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic).")
