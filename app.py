"""
app.py - Dashboard Tương Tác cho Hệ Thống Khuyến Nghị Điện Thoại
Tác giả: Hệ thống gợi ý dựa trên Content-Based Filtering + Cosine Similarity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="PhoneRec · Smartphone Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS CUSTOM - Giao diện tối hiện đại, accent xanh lam điện
# ============================================================
st.markdown("""
<style>
/* ── Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Nền tối toàn trang ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background-color: #0d0f14; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #7c8fa8;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.2rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8;
    border-bottom: 2px solid #38bdf8;
}

/* ── Metric card ── */
.metric-card {
    background: linear-gradient(135deg, #141820 0%, #1a2030 100%);
    border: 1px solid #232d3f;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c8fa8;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
}
.metric-sub {
    font-size: 0.7rem;
    color: #4a5568;
    margin-top: 0.4rem;
}

/* ── Section header ── */
.section-header {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8;
    font-weight: 600;
    margin-bottom: 0.8rem;
    margin-top: 1.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2a3a;
}

/* ── Result card ── */
.result-card {
    background: #141820;
    border: 1px solid #1e2a3a;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-left-color: #7dd3fc; }
.result-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #38bdf8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0.25rem 0;
    text-transform: capitalize;
}
.result-meta {
    font-size: 0.78rem;
    color: #7c8fa8;
    margin-bottom: 0.5rem;
}
.score-bar-bg {
    background: #1e2a3a;
    border-radius: 4px;
    height: 5px;
    margin-top: 0.5rem;
}
.score-bar-fill {
    background: linear-gradient(90deg, #1d4ed8, #38bdf8);
    border-radius: 4px;
    height: 5px;
}
.tag-chip {
    display: inline-block;
    background: #1e3a52;
    color: #7dd3fc;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    margin-right: 0.3rem;
    letter-spacing: 0.05em;
}

/* ── Filter panel ── */
.filter-panel {
    background: #141820;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.4rem;
}

/* ── Selectbox, slider label override ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stTextInput"] label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    letter-spacing: 0.03em;
}

/* ── Divider ── */
hr { border-color: #1e2a3a; margin: 1.5rem 0; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2e 0%, #0d0f14 60%);
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}
.hero-accent { color: #38bdf8; }
.hero-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# NẠP DỮ LIỆU - Dùng cache để tránh đọc lại file mỗi lần
# ============================================================
@st.cache_data
def load_data():
    """Nạp 3 file dữ liệu chính và trả về dưới dạng tuple."""
    df_names    = pd.read_csv("phone_names.csv")
    df_tags     = pd.read_csv("tag_matrix.csv")
    df_features = pd.read_csv("feature_matrix.csv")
    return df_names, df_tags, df_features

try:
    df_names, df_tags, df_features = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file dữ liệu: {e}\n\nĐặt 3 file CSV cùng thư mục với app.py rồi chạy lại.")
    st.info("📂 Đảm bảo các file `phone_names.csv`, `tag_matrix.csv`, `feature_matrix.csv` nằm **cùng thư mục** với `app.py`.")
    st.stop()

# Lưu ý: tất cả code bên dưới chỉ chạy khi load_data() thành công.
# st.stop() ở trên đảm bảo Streamlit dừng script ngay lập tức nếu có lỗi.

# ============================================================
# HẰNG SỐ & ÁNH XẠ NHÃN NHU CẦU
# ============================================================
else: 
    TAG_COLS = df_tags.columns.tolist()

    # Ánh xạ tên cột kỹ thuật → nhãn tiếng Việt thân thiện
    TAG_LABELS = {
        "Gaming_Need":         "🎮 Gaming",
        "Battery_Need":        "🔋 Pin trâu",
        "Photography_Need":    "📷 Chụp ảnh",
        "Performance_Need":    "⚡ Hiệu năng",
        "Large_Display_Need":  "📺 Màn hình lớn",
        "HighRes_Need":        "🖥️ Độ phân giải cao",
        "Multitask_Need":      "🔄 Đa nhiệm",
        "Budget_King_Need":    "💰 Giá rẻ",
        "Midrange_Value_Need": "🏷️ Tầm trung",
        "Premium_Luxury_Need": "👑 Cao cấp",
    }

    PRICE_MIN = int(df_names["Price"].min())
    PRICE_MAX = int(df_names["Price"].max())

    def fmt_vnd(val: float) -> str:
        """Định dạng số thành chuỗi VND dễ đọc."""
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M ₫"
        return f"{val/1_000:.0f}K ₫"


    @st.cache_data
    def load_eda_data():
        """Nạp file dữ liệu thô sau tiền xử lý lần 1 (preprocess_first_result.csv) dùng cho EDA lần 2."""
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        return pd.read_csv(os.path.join(base, "preprocess_first_result.csv"))

    try:
        df_eda = load_eda_data()
    except FileNotFoundError:
        df_eda = None  # Tab EDA sẽ hiển thị cảnh báo nếu thiếu file


    # ============================================================
    # HERO HEADER
    # ============================================================
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📱 Phone<span class="hero-accent">Rec</span> · Smartphone Intelligence</div>
        <div class="hero-sub">Content-Based Filtering · Cosine Similarity · 445 sản phẩm · 40 chiều đặc trưng</div>
    </div>
    """, unsafe_allow_html=True)


    # ============================================================
    # TABS CHÍNH
    # ============================================================
    tab_data, tab_eda, tab_rec = st.tabs([
        "📊  Tổng quan kiểm định dữ liệu",
        "🔬  EDA Insights",
        "🔍  Hộp thử nghiệm gợi ý",
    ])


    # ════════════════════════════════════════════════════════════
    # TAB 1 — TỔNG QUAN KIỂM ĐỊNH DỮ LIỆU
    # ════════════════════════════════════════════════════════════
    with tab_data:

        # ── Metric cards ──────────────────────────────────────
        st.markdown('<div class="section-header">Chỉ số toàn hệ thống</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            (c1, "445",  "Tổng sản phẩm",       "điện thoại thông minh"),
            (c2, "40",   "Chiều đặc trưng",      "feature_matrix.csv"),
            (c3, "0%",   "Tỷ lệ khuyết thiếu",   "sau tiền xử lý"),
            (c4, "0.602","Cosine trung bình",     "toàn bộ cặp sản phẩm"),
        ]
        for col, val, label, sub in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Biểu đồ thị phần thương hiệu ─────────────────────
        st.markdown('<div class="section-header">Thị phần thương hiệu</div>', unsafe_allow_html=True)

        # Trích xuất tên brand từ cột Name (lấy từ feature_matrix brand dummies)
        brand_cols = [c for c in df_features.columns if c.startswith("Brand_Raw_")]
        if brand_cols:
            # Tái tạo tên brand từ one-hot
            def get_brand(row):
                for bc in brand_cols:
                    if row[bc] == 1:
                        return bc.replace("Brand_Raw_", "").replace("_", " ").title()
                return "Other"
            brand_series = df_features[brand_cols].apply(get_brand, axis=1)
        else:
            # Fallback: trích từ tên máy
            brand_series = df_names["Name"].str.split().str[0].str.title()

        brand_counts = brand_series.value_counts().reset_index()
        brand_counts.columns = ["Brand", "Count"]
        brand_counts["Pct"] = (brand_counts["Count"] / brand_counts["Count"].sum() * 100).round(1)

        fig_brand = px.bar(
            brand_counts,
            x="Brand", y="Count",
            text=brand_counts["Pct"].astype(str) + "%",
            color="Count",
            color_continuous_scale=[[0, "#1e3a52"], [1, "#38bdf8"]],
            template="plotly_dark",
        )
        fig_brand.update_traces(
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
            marker_line_width=0,
        )
        fig_brand.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,15,20,0.9)",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2a3a", title=None),
            yaxis=dict(gridcolor="#1e2a3a", title="Số lượng"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=10),
            height=340,
        )
        st.plotly_chart(fig_brand, use_container_width=True)

        # ── Biểu đồ phân phối nhãn nhu cầu ──────────────────
        st.markdown('<div class="section-header">Phân phối nhãn nhu cầu (Need Tags)</div>', unsafe_allow_html=True)

        tag_counts = df_tags[TAG_COLS].sum().reset_index()
        tag_counts.columns = ["Tag", "Count"]
        tag_counts["Label"] = tag_counts["Tag"].map(TAG_LABELS)
        tag_counts["Pct"] = (tag_counts["Count"] / len(df_tags) * 100).round(1)
        tag_counts = tag_counts.sort_values("Count", ascending=True)

        fig_tags = px.bar(
            tag_counts,
            x="Count", y="Label",
            orientation="h",
            text=tag_counts["Pct"].astype(str) + "%",
            color="Count",
            color_continuous_scale=[[0, "#1d4ed8"], [0.5, "#38bdf8"], [1, "#7dd3fc"]],
            template="plotly_dark",
            hover_data={"Tag": True, "Count": True, "Pct": True, "Label": False},
        )
        fig_tags.update_traces(
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
            marker_line_width=0,
        )
        fig_tags.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,15,20,0.9)",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2a3a", title="Số máy đạt nhãn"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10),
            height=360,
        )
        st.plotly_chart(fig_tags, use_container_width=True)

        # ── Phân phối giá ─────────────────────────────────────
        st.markdown('<div class="section-header">Phân phối giá sản phẩm</div>', unsafe_allow_html=True)

        price_m = df_names["Price"] / 1_000_000  # đổi sang triệu VND
        fig_price = px.histogram(
            price_m, nbins=30,
            labels={"value": "Giá (triệu ₫)", "count": "Số lượng"},
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_price.update_traces(marker_line_width=0, opacity=0.85)
        fig_price.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,15,20,0.9)",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2a3a", title="Giá (triệu ₫)"),
            yaxis=dict(gridcolor="#1e2a3a", title="Số sản phẩm"),
            showlegend=False,
            margin=dict(t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_price, use_container_width=True)



    # ════════════════════════════════════════════════════════════
    # TAB 2 — EDA INSIGHTS
    # ════════════════════════════════════════════════════════════
    with tab_eda:

        if df_eda is None:
            st.warning(
                "⚠️ Không tìm thấy file `preprocess_first_result.csv`. "
                "Đặt file này cùng thư mục với `app.py` để xem EDA Insights."
            )
        else:
            # ── Chuẩn bị dữ liệu dùng chung trong tab ──────────
            NUMERIC_VARS = [
                'Price', 'antutu_11', 'Battery', 'RAM_min', 'ROM_min',
                'Screen Size', 'PPI', 'rear_mp_max', 'front_mp'
            ]
            NUMERIC_LABELS = {
                'Price': 'Giá (VND)', 'antutu_11': 'AnTuTu v11', 'Battery': 'Pin (mAh)',
                'RAM_min': 'RAM (GB)', 'ROM_min': 'ROM (GB)', 'Screen Size': 'Màn hình (inch)',
                'PPI': 'PPI', 'rear_mp_max': 'Camera sau (MP)', 'front_mp': 'Camera trước (MP)'
            }

            # ── [1] Phân phối giá theo nguồn dữ liệu ────────────
            st.markdown('<div class="section-header">1 · Phân phối giá theo nguồn (Price Source)</div>', unsafe_allow_html=True)

            price_src_map = {0: "Giá thực (Observed)", 1: "Giá ước tính (Imputed)"}
            df_eda['Price_source_label'] = df_eda['Price_source'].map(price_src_map)

            fig_ps = px.histogram(
                df_eda, x='Price', color='Price_source_label',
                nbins=40, barmode='overlay', opacity=0.7,
                color_discrete_map={
                    "Giá thực (Observed)": "#38bdf8",
                    "Giá ước tính (Imputed)": "#f59e0b"
                },
                labels={'Price': 'Giá (VND)', 'count': 'Số sản phẩm', 'Price_source_label': 'Nguồn giá'},
                template="plotly_dark",
            )
            fig_ps.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,20,0.9)",
                font=dict(family="Inter", color="#94a3b8"),
                xaxis=dict(gridcolor="#1e2a3a"),
                yaxis=dict(gridcolor="#1e2a3a", title="Số sản phẩm"),
                legend=dict(title="Nguồn giá", bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10), height=280,
            )
            n_obs = (df_eda['Price_source'] == 0).sum()
            n_imp = (df_eda['Price_source'] == 1).sum()
            ci1, ci2, ci3 = st.columns(3)
            ci1.metric("Giá thực (Observed)", n_obs, f"{n_obs/len(df_eda)*100:.1f}%")
            ci2.metric("Giá ước tính (Imputed - RF)", n_imp, f"{n_imp/len(df_eda)*100:.1f}%")
            ci3.metric("Tổng cộng", len(df_eda))
            st.plotly_chart(fig_ps, use_container_width=True)
            st.caption(
                "💡 **Insight:** Phân phối giá của nhóm Imputed (Random Forest) bám sát nhóm Observed, "
                "xác nhận chiến lược imputation bằng RF log-price là hợp lý và không gây lệch phân phối."
            )

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── [2] Univariate distributions (9 biến số) ────────
            st.markdown('<div class="section-header">2 · Univariate Analysis — Phân phối 9 biến số chính</div>', unsafe_allow_html=True)

            selected_var = st.selectbox(
                "Chọn biến để xem phân phối",
                options=NUMERIC_VARS,
                format_func=lambda x: NUMERIC_LABELS[x],
                label_visibility="collapsed",
            )

            col_u1, col_u2 = st.columns([2, 1])
            with col_u1:
                series = df_eda[selected_var].dropna()
                skew_val = series.skew()
                mean_val = series.mean()
                med_val  = series.median()

                fig_uni = px.histogram(
                    df_eda, x=selected_var, nbins=35,
                    labels={selected_var: NUMERIC_LABELS[selected_var]},
                    template="plotly_dark",
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_uni.add_vline(x=mean_val, line_dash="dash", line_color="#ef4444",
                                annotation_text=f"Mean: {mean_val:.1f}", annotation_position="top right")
                fig_uni.add_vline(x=med_val, line_dash="solid", line_color="#22c55e",
                                annotation_text=f"Median: {med_val:.1f}", annotation_position="top left")
                fig_uni.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,20,0.9)",
                    font=dict(family="Inter", color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e2a3a"),
                    yaxis=dict(gridcolor="#1e2a3a", title="Số sản phẩm"),
                    margin=dict(t=30, b=10), height=300,
                )
                st.plotly_chart(fig_uni, use_container_width=True)

            with col_u2:
                skew_note = "Lệch phải nặng → Cần Log Transform" if skew_val > 1 else (
                            "Lệch trái → Phân phối tương đối cân" if skew_val < -0.5 else
                            "Phân phối tương đối đối xứng")
                st.markdown(f"""
                <div style="background:#141820;border:1px solid #1e2a3a;border-radius:10px;padding:1.2rem;margin-top:0.5rem;">
                    <div class="metric-label">Skewness</div>
                    <div class="metric-value" style="font-size:1.6rem;color:{'#f59e0b' if abs(skew_val)>1 else '#38bdf8'};">{skew_val:.2f}</div>
                    <div style="font-size:0.75rem;color:#64748b;margin-top:0.5rem;">{skew_note}</div>
                    <hr style="border-color:#1e2a3a;margin:0.8rem 0;">
                    <div class="metric-label">Mean</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e2e8f0;">{mean_val:,.1f}</div>
                    <div class="metric-label" style="margin-top:0.6rem;">Median</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e2e8f0;">{med_val:,.1f}</div>
                    <div class="metric-label" style="margin-top:0.6rem;">Std Dev</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e2e8f0;">{series.std():,.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Bảng tổng hợp skewness tất cả biến
            skew_df = pd.DataFrame([
                {"Biến": NUMERIC_LABELS[v], "Skewness": round(df_eda[v].skew(), 2),
                "Cần Log Transform": "✅" if df_eda[v].skew() > 1 else "—"}
                for v in NUMERIC_VARS
            ]).sort_values("Skewness", ascending=False)
            with st.expander("📋 Bảng tổng hợp Skewness tất cả biến"):
                st.dataframe(skew_df, use_container_width=True, hide_index=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── [3] Correlation Heatmaps (Pearson + Spearman) ──
            st.markdown('<div class="section-header">3 · Bivariate Analysis — Correlation Heatmaps</div>', unsafe_allow_html=True)

            pearson_features = [
                'Price', 'antutu_11', 'Battery', 'Screen Size', 'PPI',
                'max_freq_ghz', 'min_freq_ghz', 'clock', 'weighted_mean',
                'rear_mp_max', 'front_mp'
            ]
            spearman_features = [
                'Refresh Rate', 'OS_Version', 'total_cores', 'rear_count', 'SIM_total',
                'RAM_min', 'RAM_max', 'ROM_min', 'ROM_max',
                'rear_ois', 'rear_telephoto', 'rear_wide'
            ]

            corr_tab = st.radio(
                "Chọn loại tương quan",
                ["Pearson (Biến liên tục)", "Spearman (Biến rời rạc/thứ bậc)"],
                horizontal=True, label_visibility="collapsed"
            )

            if corr_tab == "Pearson (Biến liên tục)":
                feats = pearson_features
                method = "pearson"
                subtitle = "Pearson r — đo mối quan hệ tuyến tính giữa các biến liên tục"
            else:
                feats = spearman_features
                method = "spearman"
                subtitle = "Spearman ρ — đo mối quan hệ đơn điệu giữa các biến rời rạc / thứ bậc"

            # Tính correlation matrix, chỉ lấy lower triangle
            avail_feats = [f for f in feats if f in df_eda.columns]
            corr_mat = df_eda[avail_feats].corr(method=method)
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)  # upper triangle mask để ẩn
            corr_lower = corr_mat.where(~mask)  # chỉ giữ lower triangle + diagonal

            fig_heat = px.imshow(
                corr_lower,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                template="plotly_dark",
                aspect="auto",
            )
            fig_heat.update_traces(textfont_size=10)
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,20,0.9)",
                font=dict(family="Inter", color="#94a3b8", size=11),
                coloraxis_colorbar=dict(title="r / ρ", tickfont=dict(size=10)),
                margin=dict(t=20, b=10, l=10, r=10),
                height=480,
            )
            st.caption(f"📐 {subtitle}")
            st.plotly_chart(fig_heat, use_container_width=True)

            # Highlight cặp tương quan cao (|r| > 0.85) → các biến bị loại
            THRESHOLD = 0.85
            upper_tri = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
            high_pairs = (
                upper_tri.stack()
                .reset_index()
                .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"})
            )
            high_pairs = high_pairs[high_pairs["Correlation"].abs() > THRESHOLD].sort_values(
                "Correlation", key=abs, ascending=False
            )
            if not high_pairs.empty:
                high_pairs["Correlation"] = high_pairs["Correlation"].round(3)
                with st.expander(f"⚠️ {len(high_pairs)} cặp tương quan cao (|r| > {THRESHOLD}) → đã loại khỏi feature_matrix"):
                    st.dataframe(high_pairs, use_container_width=True, hide_index=True)
            else:
                st.success(f"✅ Không có cặp nào vượt ngưỡng |r| > {THRESHOLD}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── [4] Brand Premium Score ──────────────────────────
            st.markdown('<div class="section-header">4 · Brand Premium Score — Phân tích định giá vượt hiệu năng</div>', unsafe_allow_html=True)

            df_bp = df_eda.dropna(subset=['antutu_11', 'Price']).copy()
            df_bp = df_bp[df_bp['antutu_11'] > 0]
            df_bp['log_antutu'] = np.log10(df_bp['antutu_11'])
            df_bp['log_price']  = np.log10(df_bp['Price'])
            reg = stats.linregress(df_bp['log_antutu'], df_bp['log_price'])
            r2  = reg.rvalue ** 2
            df_bp['residual'] = df_bp['log_price'] - (reg.intercept + reg.slope * df_bp['log_antutu'])
            brand_res = df_bp.groupby('Brand')['residual'].mean().sort_values(ascending=False).reset_index()
            brand_res.columns = ['Brand', 'Premium Score']
            brand_res['Brand'] = brand_res['Brand'].str.title()

            col_bp1, col_bp2 = st.columns([3, 1])
            with col_bp1:
                fig_bp = px.bar(
                    brand_res, x='Premium Score', y='Brand', orientation='h',
                    color='Premium Score',
                    color_continuous_scale=[[0, "#1d4ed8"], [0.5, "#e2e8f0"], [1, "#ef4444"]],
                    template="plotly_dark",
                )
                fig_bp.add_vline(x=0, line_color="#64748b", line_dash="dash")
                fig_bp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,20,0.9)",
                    font=dict(family="Inter", color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e2a3a", title="Brand Premium Score (Residual)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed", title=None),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10), height=380,
                )
                st.plotly_chart(fig_bp, use_container_width=True)
            with col_bp2:
                st.markdown(f"""
                <div style="background:#141820;border:1px solid #1e2a3a;border-radius:10px;padding:1.2rem;margin-top:0.5rem;">
                    <div class="metric-label">R² (log-log fit)</div>
                    <div class="metric-value" style="font-size:1.5rem;">{r2:.3f}</div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:0.4rem;">Hiệu năng giải thích {r2*100:.1f}% biến thiên giá</div>
                    <hr style="border-color:#1e2a3a;margin:0.8rem 0;">
                    <div class="metric-label">👑 Premium nhất</div>
                    <div style="font-size:0.85rem;color:#f59e0b;font-weight:600;">{brand_res.iloc[0]['Brand']}</div>
                    <div style="font-size:0.72rem;color:#64748b;">Score: {brand_res.iloc[0]['Premium Score']:.3f}</div>
                    <div class="metric-label" style="margin-top:0.8rem;">💰 Value nhất</div>
                    <div style="font-size:0.85rem;color:#22c55e;font-weight:600;">{brand_res.iloc[-1]['Brand']}</div>
                    <div style="font-size:0.72rem;color:#64748b;">Score: {brand_res.iloc[-1]['Premium Score']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.caption(
                "💡 **Cách đọc:** Score > 0 → hãng định giá cao hơn mức hiệu năng dự đoán (Premium Pricing). "
                "Score < 0 → hãng cung cấp hiệu năng tốt hơn so với giá tiền (High Value/Cost Ratio)."
            )

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── [5] Chipset AnTuTu Distribution ─────────────────
            st.markdown('<div class="section-header">5 · Chipset Performance Distribution — AnTuTu v11</div>', unsafe_allow_html=True)

            df_chip = df_eda.dropna(subset=['Chipset_name', 'antutu_11']).copy()
            chip_order = (
                df_chip.groupby('Chipset_name')['antutu_11']
                .median().sort_values(ascending=False).index.tolist()
            )

            # Box plot mỗi chipset maker
            fig_chip = px.box(
                df_chip, x='Chipset_name', y='antutu_11',
                category_orders={'Chipset_name': chip_order},
                color='Chipset_name',
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_dark",
                labels={'Chipset_name': 'Hãng Chipset', 'antutu_11': 'Điểm AnTuTu v11'},
            )
            fig_chip.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,20,0.9)",
                font=dict(family="Inter", color="#94a3b8"),
                xaxis=dict(gridcolor="#1e2a3a"),
                yaxis=dict(gridcolor="#1e2a3a", tickformat=","),
                showlegend=False,
                margin=dict(t=10, b=10), height=360,
            )
            st.plotly_chart(fig_chip, use_container_width=True)

            # Bảng median AnTuTu theo nhóm chipset
            chip_stats = (
                df_chip.groupby('Chipset_name')['antutu_11']
                .agg(["median", "mean", "count"])
                .rename(columns={"median": "Median AnTuTu", "mean": "Mean AnTuTu", "count": "Số máy"})
                .sort_values("Median AnTuTu", ascending=False)
                .reset_index()
            )
            chip_stats["Median AnTuTu"] = chip_stats["Median AnTuTu"].map("{:,.0f}".format)
            chip_stats["Mean AnTuTu"]   = chip_stats["Mean AnTuTu"].map("{:,.0f}".format)
            chip_stats.rename(columns={"Chipset_name": "Chipset"}, inplace=True)

            with st.expander("📋 Thống kê AnTuTu theo hãng chipset"):
                st.dataframe(chip_stats, use_container_width=True, hide_index=True)

            st.caption(
                "💡 **Insight:** Apple dẫn đầu tuyệt đối về hiệu năng trung vị. "
                "Snapdragon và Exynos cạnh tranh ở phân khúc cao cấp. "
                "MediaTek trải dài từ budget → flagship (Dimensity). "
                "Unisoc tập trung phân khúc giá rẻ."
            )


    # ════════════════════════════════════════════════════════════
    # TAB 3 — HỘP THỬ NGHIỆM GỢI Ý
    # ════════════════════════════════════════════════════════════
    with tab_rec:

        col_filter, col_result = st.columns([1, 2], gap="large")

        # ── CỘT BỘ LỌC (Inputs) ──────────────────────────────
        with col_filter:
            st.markdown('<div class="section-header" style="margin-top:0.5rem">Bộ lọc nhu cầu</div>', unsafe_allow_html=True)

            # 1. Chọn nhu cầu cốt lõi
            selected_tags_labels = st.multiselect(
                "Chọn nhu cầu (có thể chọn nhiều)",
                options=list(TAG_LABELS.values()),
                default=[],
                placeholder="Chọn ít nhất 1 nhu cầu…",
            )
            # Ánh xạ nhãn đẹp → tên cột gốc
            label_to_col = {v: k for k, v in TAG_LABELS.items()}
            selected_tag_cols = [label_to_col[lbl] for lbl in selected_tags_labels]

            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Khoảng giá
            st.markdown("**Khoảng giá (VND)**", unsafe_allow_html=False)
            price_range = st.slider(
                "Khoảng giá",
                min_value=PRICE_MIN,
                max_value=PRICE_MAX,
                value=(PRICE_MIN, PRICE_MAX),
                step=500_000,
                format="%d",
                label_visibility="collapsed",
            )
            pc1, pc2 = st.columns(2)
            pc1.caption(f"Từ: **{fmt_vnd(price_range[0])}**")
            pc2.caption(f"Đến: **{fmt_vnd(price_range[1])}**")

            st.markdown("<br>", unsafe_allow_html=True)

            # 3. Chọn sản phẩm gốc làm mốc
            st.markdown('<div class="section-header" style="margin-top:0rem">Sản phẩm mục tiêu</div>', unsafe_allow_html=True)
            phone_list = df_names["Name"].str.title().tolist()
            search_input = st.text_input(
                "Tìm tên điện thoại",
                placeholder="Ví dụ: Samsung, iPhone, Xiaomi...",
                label_visibility="collapsed",
            )

            # Lọc danh sách theo từ khóa tìm kiếm
            filtered_phones = [p for p in phone_list if search_input.lower() in p.lower()] if search_input else phone_list
            if not filtered_phones:
                filtered_phones = phone_list

            target_phone = st.selectbox(
                "Chọn sản phẩm gốc",
                options=filtered_phones,
                label_visibility="collapsed",
            )
            target_idx = phone_list.index(target_phone)

            # Nút tìm kiếm
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🔍  Tìm sản phẩm tương đồng", use_container_width=True, type="primary")

            # Thông tin sản phẩm mục tiêu
            st.markdown('<div class="section-header">Thông tin mục tiêu</div>', unsafe_allow_html=True)
            target_info = df_names.iloc[target_idx]
            target_tags = df_tags.iloc[target_idx]
            active_tags = [TAG_LABELS[c] for c in TAG_COLS if target_tags[c] == 1]

            st.markdown(f"""
            <div style="background:#141820;border:1px solid #1e2a3a;border-radius:10px;padding:1rem;">
                <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;text-transform:capitalize;">{target_phone}</div>
                <div style="font-size:0.82rem;color:#38bdf8;margin:0.3rem 0;">{fmt_vnd(target_info['Price'])}</div>
                <div style="font-size:0.75rem;color:#64748b;">RAM {int(target_info['RAM_min'])}GB · ROM {int(target_info['ROM_min'])}GB</div>
                <div style="margin-top:0.6rem;">{''.join(f'<span class="tag-chip">{t}</span>' for t in active_tags) if active_tags else '<span style="color:#4a5568;font-size:0.75rem;">Không có nhãn nhu cầu</span>'}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── CỘT KẾT QUẢ (Outputs) ────────────────────────────
        with col_result:
            st.markdown('<div class="section-header" style="margin-top:0.5rem">Kết quả gợi ý</div>', unsafe_allow_html=True)

            if not run_btn:
                st.markdown("""
                <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
                    <div style="font-size:2.5rem;margin-bottom:1rem;">🔍</div>
                    <div style="font-size:0.9rem;">Thiết lập bộ lọc và nhấn <strong style="color:#38bdf8">Tìm sản phẩm tương đồng</strong> để bắt đầu.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── TẦNG 1: HARD FILTER (Boolean Mask) ──
                # Lọc theo khoảng giá
                price_mask = (df_names["Price"] >= price_range[0]) & (df_names["Price"] <= price_range[1])

                # Lọc theo nhu cầu (AND logic: máy phải đạt TẤT CẢ nhu cầu đã chọn)
                if selected_tag_cols:
                    tag_mask = df_tags[selected_tag_cols].all(axis=1)
                else:
                    tag_mask = pd.Series([True] * len(df_tags))

                combined_mask = price_mask & tag_mask

                # Loại bỏ sản phẩm gốc khỏi tập kết quả
                combined_mask.iloc[target_idx] = False

                filtered_indices = combined_mask[combined_mask].index.tolist()

                # Xử lý edge case: không có sản phẩm nào sau lọc
                if len(filtered_indices) == 0:
                    st.warning(
                        "⚠️ **Không tìm thấy sản phẩm nào** phù hợp với bộ lọc hiện tại.\n\n"
                        "Hãy thử:\n"
                        "- Mở rộng khoảng giá\n"
                        "- Bỏ bớt một số nhu cầu đã chọn\n"
                        "- Chọn lại kết hợp nhu cầu khác nhau"
                    )
                else:
                    # ── TẦNG 2: SOFT FILTER (Cosine Similarity) ──
                    # Vector đặc trưng của sản phẩm mục tiêu
                    target_vec = df_features.iloc[[target_idx]].values

                    # Cắt lấy phần feature_matrix tương ứng tập đã lọc
                    subset_features = df_features.iloc[filtered_indices].values

                    # Tính Cosine Similarity
                    scores = cosine_similarity(target_vec, subset_features)[0]

                    # Đóng gói kết quả
                    results_df = pd.DataFrame({
                        "original_idx": filtered_indices,
                        "score": scores,
                    })
                    results_df = results_df.sort_values("score", ascending=False).head(10)

                    # ── Hiển thị tóm tắt ──
                    n_filtered = len(filtered_indices)
                    n_shown = len(results_df)
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("Máy qua lọc cứng", n_filtered)
                    col_s2.metric("Kết quả hiển thị", n_shown)
                    top_score = results_df["score"].max()
                    col_s3.metric("Điểm cao nhất", f"{top_score:.3f}")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Danh sách kết quả ──
                    for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
                        idx = int(row["original_idx"])
                        score = row["score"]
                        info = df_names.iloc[idx]
                        tags_row = df_tags.iloc[idx]
                        phone_name = df_names.iloc[idx]["Name"].title()
                        active = [TAG_LABELS[c] for c in TAG_COLS if tags_row[c] == 1]
                        bar_pct = int(score * 100)

                        tag_html = "".join(f'<span class="tag-chip">{t}</span>' for t in active)
                        if not active:
                            tag_html = '<span style="color:#4a5568;font-size:0.72rem;">Không có nhãn</span>'

                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-rank">#{rank:02d} · Độ tương đồng</div>
                            <div class="result-name">{phone_name}</div>
                            <div class="result-meta">
                                💰 {fmt_vnd(info['Price'])} &nbsp;|&nbsp;
                                🧠 RAM {int(info['RAM_min'])}GB &nbsp;|&nbsp;
                                💾 ROM {int(info['ROM_min'])}GB
                            </div>
                            <div>{tag_html}</div>
                            <div style="display:flex;align-items:center;gap:0.6rem;margin-top:0.5rem;">
                                <div class="score-bar-bg" style="flex:1;">
                                    <div class="score-bar-fill" style="width:{bar_pct}%;"></div>
                                </div>
                                <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#38bdf8;min-width:3rem;">{score:.4f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Biểu đồ điểm số ──
                    if len(results_df) >= 3:
                        st.markdown('<div class="section-header">Biểu đồ điểm Cosine Similarity</div>', unsafe_allow_html=True)
                        chart_names = [df_names.iloc[int(r["original_idx"])]["Name"].title()[:28] for _, r in results_df.iterrows()]
                        fig_scores = go.Figure(go.Bar(
                            x=results_df["score"].tolist(),
                            y=chart_names,
                            orientation="h",
                            marker=dict(
                                color=results_df["score"].tolist(),
                                colorscale=[[0, "#1d4ed8"], [1, "#38bdf8"]],
                                line_width=0,
                            ),
                            text=[f"{s:.4f}" for s in results_df["score"]],
                            textposition="outside",
                            textfont=dict(size=10, color="#94a3b8"),
                        ))
                        fig_scores.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(13,15,20,0.9)",
                            font=dict(family="Inter", color="#94a3b8", size=11),
                            xaxis=dict(gridcolor="#1e2a3a", range=[0, 1.05], title="Cosine Score"),
                            yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
                            margin=dict(t=10, b=10, l=10, r=60),
                            height=max(240, len(results_df) * 36),
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)