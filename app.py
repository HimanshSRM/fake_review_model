import streamlit as st
import joblib
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import os
import plotly.express as px
from db_utils import get_reviews, get_dashboard_stats, get_connection, get_product_id_or_insert

# Paths
FAST_MODEL_PATH = "fast_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DEEP_MODEL_PATH = "deep_model"
DATA_DIR = "data"

# Product list
PRODUCTS = ["Phone", "Laptop", "Shoes", "Headphones", "Washing Machine", "Camera"]

# Load fast model
@st.cache_resource
def load_fast_model():
    model = joblib.load(FAST_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Load deep model
@st.cache_resource
def load_deep_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(DEEP_MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(DEEP_MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer, device

# Fast Model Prediction
def predict_fast(review, model, vectorizer):
    vec = vectorizer.transform([review])
    result = model.predict(vec)[0]
    return result  # 0 = Real, 1 = Fake

# Deep Model Prediction with confidence
def predict_deep(review, model, tokenizer, device):
    inputs = tokenizer(review, truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        return prediction.item(), confidence.item()

# Insert review into MySQL from DataFrame row
def insert_review_to_mysql(row, label_column):
    conn = get_connection()
    cursor = conn.cursor()
    product_id = get_product_id_or_insert(row["Product"])
    cursor.execute("""
        INSERT INTO reviews (review_text, product_id, model_used, prediction, confidence)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        row["Review"], product_id, row["Model"], label_column, row["Confidence"]
    ))
    conn.commit()
    conn.close()

# Streamlit UI
def main():
    st.set_page_config(page_title="Fake Review Detector", layout="wide")
    st.title("🕵️‍♂️ Fake Review Detector")

    review_text = st.text_area("Enter a review:", height=150)
    product = st.selectbox("Select product:", PRODUCTS)
    model_option = st.radio("Choose model option:", ["Compare Both Models", "Fast Model Only", "Deep Model Only"])

    if st.button("Detect"):
        if not review_text.strip():
            st.warning("⚠️ Please enter a review.")
            return

        fast_model, vectorizer = load_fast_model()
        deep_model, tokenizer, device = load_deep_model()

        # Predictions
        fast_result = predict_fast(review_text, fast_model, vectorizer)
        deep_result, deep_conf = predict_deep(review_text, deep_model, tokenizer, device)

        # Format predictions
        fast_label = "🟢 Real" if fast_result == 0 else "🔴 Fake"
        deep_label = "🟢 Real" if deep_result == 0 else "🔴 Fake"
        deep_label += f" ({deep_conf * 100:.2f}%)"

        st.subheader("🔍 Results")

        if model_option == "Fast Model Only":
            st.write(f"**Fast Model Prediction:** {fast_label}")
            final_result = fast_result
            model_used = "Fast Model"

        elif model_option == "Deep Model Only":
            st.write(f"**Deep Model Prediction:** {deep_label}")
            final_result = deep_result
            model_used = "Deep Model"

        else:
            st.write(f"**Fast Model:** {fast_label}")
            st.write(f"**Deep Model:** {deep_label}")
            final_result = deep_result
            model_used = "Both"

        # Save to MySQL and CSV
        row = {
            "Review": review_text,
            "Product": product,
            "Model": model_used,
            "Prediction": "Genuine" if final_result == 0 else "Fake",
            "Confidence": f"{deep_conf * 100:.2f}%" if model_used != "Fast Model" else "N/A"
        }

        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, "genuine_reviews.csv" if final_result == 0 else "fake_reviews.csv")
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

        insert_review_to_mysql(row, row["Prediction"])
        st.success(f"✅ Review saved and marked as {row['Prediction']}.")

    st.markdown("---")
    st.subheader("📋 Filter & Explore Reviews")

    selected_product = st.selectbox("Filter by Product:", ["All"] + PRODUCTS)
    selected_model = st.selectbox("Filter by Model:", ["All", "Fast Model", "Deep Model", "Both"])
    page = st.number_input("Page", min_value=1, step=1, value=1)
    limit = 5

    product_filter = selected_product if selected_product != "All" else None
    model_filter = selected_model if selected_model != "All" else None

    results = get_reviews(filter_product=product_filter, filter_model=model_filter, page=page, page_size=limit)
    if results:
        for row in results:
            st.markdown(f"""
            **Review**: {row['review_text']}  
            **Product**: {row['product']}  
            **Model**: {row['model_used']}  
            **Prediction**: {row['prediction']}  
            **Confidence**: {row['confidence']}  
            ---""")
    else:
        st.info("No reviews found.")

    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")
    stats = get_dashboard_stats()
    st.metric("Total Reviews", stats["total"])
    st.metric("Genuine Reviews", stats["genuine"])
    st.metric("Fake Reviews", stats["fake"])

    model_data = stats["model_breakdown"]
    if model_data:
        df_model = pd.DataFrame(model_data, columns=["Model", "Count"])
        fig = px.pie(df_model, names="Model", values="Count", title="Model Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
