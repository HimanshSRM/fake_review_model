import mysql.connector

# MySQL Connection
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1Pplolqppr@",
        database="review_db"
    )

# Get product_id if exists, else insert and return new id
def get_product_id_or_insert(product_name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM products WHERE name = %s", (product_name,))
    result = cursor.fetchone()
    if result:
        product_id = result[0]
    else:
        cursor.execute("INSERT INTO products (name) VALUES (%s)", (product_name,))
        conn.commit()
        product_id = cursor.lastrowid

    conn.close()
    return product_id

# Get paginated, filtered reviews
def get_reviews(filter_product=None, filter_model=None, page=1, page_size=10):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    offset = (page - 1) * page_size
    query = """
        SELECT r.review_text, p.name AS product, r.model_used, r.prediction, r.confidence
        FROM reviews r
        JOIN products p ON r.product_id = p.id
        WHERE (%s IS NULL OR p.name = %s)
          AND (%s IS NULL OR r.model_used = %s)
        ORDER BY r.id DESC
        LIMIT %s OFFSET %s
    """
    cursor.execute(query, (filter_product, filter_product, filter_model, filter_model, page_size, offset))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Get total number of reviews (with filters)
def get_total_review_count(filter_product=None, filter_model=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT COUNT(*)
        FROM reviews r
        JOIN products p ON r.product_id = p.id
        WHERE (%s IS NULL OR p.name = %s)
          AND (%s IS NULL OR r.model_used = %s)
    """
    cursor.execute(query, (filter_product, filter_product, filter_model, filter_model))
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Dashboard stats
def get_dashboard_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM reviews")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reviews WHERE prediction = 'Genuine'")
    genuine = cursor.fetchone()[0]

    cursor.execute("SELECT model_used, COUNT(*) FROM reviews GROUP BY model_used")
    model_breakdown = cursor.fetchall()

    conn.close()
    return {
        "total": total,
        "genuine": genuine,
        "fake": total - genuine,
        "model_breakdown": model_breakdown
    }

# Filtered reviews with pagination for Streamlit
def filter_reviews_paginated(filter_product=None, filter_model=None, limit=10, offset=0):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT r.review_text, p.name, r.model_used, r.prediction, r.confidence, r.created_at
        FROM reviews r
        JOIN products p ON r.product_id = p.id
        WHERE (%s IS NULL OR p.name = %s)
          AND (%s IS NULL OR r.model_used = %s)
        ORDER BY r.id DESC
        LIMIT %s OFFSET %s
    """
    cursor.execute(query, (filter_product, filter_product, filter_model, filter_model, limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Review statistics for dashboard
def get_review_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM reviews")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reviews WHERE prediction = 'Genuine'")
    genuine = cursor.fetchone()[0]

    cursor.execute("SELECT model_used, COUNT(*) FROM reviews GROUP BY model_used")
    model_data = cursor.fetchall()

    conn.close()
    return {
        "total": total,
        "genuine_percent": (genuine / total * 100) if total > 0 else 0,
        "model_usage": model_data
    }
