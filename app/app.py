import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title='Mirae Asset Analytics',
    page_icon='M',
    layout='wide',
    initial_sidebar_state='expanded',
)

COLOR_BLUE = '#23B7F2'
COLOR_TEAL = '#16E0B5'
COLOR_GREEN = '#68E27D'
COLOR_AMBER = '#FFB84D'
COLOR_RED = '#FF4D6D'
COLOR_PURPLE = '#9B5DE5'
COLOR_GRAY = '#8EA9C5'
PALETTE = [COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_AMBER, COLOR_PURPLE, COLOR_RED]

RISK_ORDER = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk', 'Not Observable']
RFM_ORDER = [
    'Champions',
    'Loyal Customers',
    'Potential Loyalists',
    'Recent Customers',
    'Needs Attention',
    'Cannot Lose Them',
    'At Risk',
    'Lost',
    'Non-Buyer',
]
CLUSTER_ORDER = [
    'VIP / Champions',
    'Engaged Regulars',
    'Casual Buyers',
    'Dormant / At-Risk',
    'Non-Buyer',
]


st.markdown(
    """
    <style>
    :root {
        --bg: #050b13;
        --panel: #081f35;
        --panel-2: #0b2740;
        --line: #174a70;
        --muted: #8ea9c5;
        --text: #eaf6ff;
        --cyan: #16e0b5;
        --blue: #23b7f2;
        --red: #ff4d6d;
        --amber: #ffb84d;
    }
    * {
        box-sizing: border-box;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }
    .stApp {
        background:
            linear-gradient(90deg, rgba(22,224,181,0.06) 1px, transparent 1px),
            linear-gradient(rgba(35,183,242,0.04) 1px, transparent 1px),
            #050b13;
        background-size: 36px 36px;
        color: var(--text);
    }
    .block-container {
        width: min(100%, 1520px);
        max-width: none;
        padding: 1rem clamp(1rem, 1.8vw, 2rem) 2rem clamp(1rem, 1.8vw, 2rem);
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="column"] {
        min-width: 0;
    }
    [data-testid="stHorizontalBlock"] {
        gap: clamp(0.9rem, 1.2vw, 1.25rem);
        align-items: stretch;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0.9rem;
    }
    section[data-testid="stSidebar"] {
        background: #090f1c;
        border-right: 1px solid #183a59;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text);
    }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: #081f35;
        border-color: #174a70;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: var(--text);
    }
    div[data-testid="stMetric"] {
        border: 1px solid #174a70;
        border-radius: 14px;
        padding: 16px 18px;
        background: linear-gradient(180deg, #0b2740 0%, #081f35 100%);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .06em;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 800;
    }
    div[data-testid="stMetricDelta"] {
        color: var(--cyan);
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        width: 100%;
        margin: 10px 0 24px 0;
    }
    .kpi-card {
        min-height: 106px;
        border: 1px solid #174a70;
        border-radius: 14px;
        padding: 17px 18px 14px 18px;
        background: linear-gradient(180deg, #0b2740 0%, #081f35 100%);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
    }
    .kpi-label {
        color: #b7c4d6;
        text-transform: uppercase;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        overflow-wrap: anywhere;
    }
    .kpi-value {
        color: #ffffff;
        font-size: clamp(1.55rem, 1.8vw, 2.25rem);
        font-weight: 900;
        line-height: 1.05;
        margin-top: 10px;
        overflow-wrap: anywhere;
    }
    .kpi-value.good {
        color: var(--cyan);
    }
    .kpi-value.warn {
        color: var(--amber);
    }
    .kpi-value.bad {
        color: var(--red);
    }
    .kpi-delta {
        margin-top: 8px;
        color: #8ea9c5;
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }
    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        overflow: hidden;
        border: 1px solid #174a70;
        border-radius: 14px;
        background: #081f35;
        padding: 0.35rem;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }
    div[data-testid="stPlotlyChart"] > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stPlotlyChart"] svg {
        max-width: 100%;
    }
    .js-plotly-plot, .plot-container, .svg-container {
        width: 100% !important;
        max-width: 100% !important;
    }
    @media (min-width: 1280px) {
        .kpi-grid {
            grid-template-columns: repeat(6, minmax(0, 1fr));
        }
    }
    @media (max-width: 1180px) {
        .kpi-grid {
            grid-template-columns: repeat(3, minmax(160px, 1fr));
        }
    }
    @media (max-width: 760px) {
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(150px, 1fr));
        }
    }
    @media (max-width: 640px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    div[data-testid="stTabs"] button {
        color: var(--muted);
        border-radius: 10px 10px 0 0;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff;
        background: #0b2740;
        border-bottom: 2px solid var(--cyan);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #174a70;
        border-radius: 12px;
        overflow: hidden;
    }
    .modebar {
        display: none !important;
    }
    .hero {
        border: 1px solid #174a70;
        border-radius: 18px;
        padding: 20px 24px 18px 24px;
        margin-bottom: 18px;
        background:
            radial-gradient(circle at top right, rgba(35,183,242,.22), transparent 34%),
            linear-gradient(135deg, #071b33 0%, #06111f 70%);
        box-shadow: 0 18px 45px rgba(0,0,0,.28);
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 850;
        line-height: 1.12;
        margin: 0;
        color: #ffffff;
    }
    .hero-subtitle {
        color: #b7c4d6;
        margin-top: .45rem;
        font-size: .96rem;
    }
    .hero-strip {
        display: flex;
        gap: 10px;
        margin-top: 14px;
    }
    .hero-strip span {
        display: inline-block;
        width: 72px;
        height: 7px;
        border-radius: 999px;
    }
    .section-note {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: -0.35rem;
        margin-bottom: 0.75rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #174a70;
        background: #0b2740;
        color: #ffffff;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--cyan);
        color: var(--cyan);
    }
    hr {
        border-color: #183a59;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #0b2740;
        color: #ffffff;
        border: 1px solid #174a70;
        max-width: 100%;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] {
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #081f35 !important;
        border-color: #174a70 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #eaf6ff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #8ea9c5 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] div {
        color: #eaf6ff !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"] {
        z-index: 999999 !important;
    }
    [data-baseweb="popover"] > div,
    [role="listbox"] {
        background: #081f35 !important;
        border: 1px solid #174a70 !important;
        color: #eaf6ff !important;
        box-shadow: 0 18px 45px rgba(0,0,0,.35) !important;
    }
    [data-baseweb="popover"] ul {
        background: #0b1220 !important;
        border: 1px solid #174a70 !important;
        color: #eaf6ff !important;
    }
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li,
    [role="option"] {
        color: #eaf6ff !important;
    }
    [data-baseweb="popover"] li:hover,
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background: #0b2740 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def project_base():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def processed_dir():
    return os.path.join(project_base(), 'data', 'processed')


def raw_dir():
    return os.path.join(project_base(), 'data', 'raw')


def format_rs(value):
    value = 0 if pd.isna(value) else float(value)
    if abs(value) >= 10_000_000:
        return f'Rs {value / 1_000_000:.1f}M'
    if abs(value) >= 100_000:
        return f'Rs {value / 100_000:.1f}L'
    return f'Rs {value:,.0f}'


def format_pct(value):
    value = 0 if pd.isna(value) else float(value)
    return f'{value * 100:.1f}%'


def format_num(value):
    value = 0 if pd.isna(value) else value
    return f'{int(value):,}'


def metric_delta(current, baseline, pct=False):
    if pd.isna(current) or pd.isna(baseline):
        return None
    diff = current - baseline
    return round(diff * 100, 2) if pct else round(diff, 0)


def render_kpi_grid(metrics):
    cards = []
    for label, value, delta, tone in metrics:
        tone_class = f' {tone}' if tone else ''
        delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ''
        cards.append(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value{tone_class}">{value}</div>{delta_html}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def apply_common_layout(fig, height=360):
    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=54, r=24, t=64, b=48),
        paper_bgcolor='#081f35',
        plot_bgcolor='#081f35',
        font=dict(family='Segoe UI, Arial', size=12, color='#eaf6ff'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(color='#b7c4d6', size=11),
            itemwidth=30,
        ),
        title=dict(font=dict(color='#ffffff', size=16)),
        hoverlabel=dict(bgcolor='#0b2740', bordercolor='#23b7f2', font=dict(color='#ffffff')),
        uniformtext_minsize=9,
        uniformtext_mode='hide',
    )
    for trace_type in ['bar', 'scatter', 'histogram']:
        fig.update_traces(cliponaxis=False, selector=dict(type=trace_type))
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor='#183a59', zeroline=False, automargin=True)
    fig.update_xaxes(color='#b7c4d6', title_font=dict(color='#8ea9c5'), tickfont=dict(color='#b7c4d6'))
    fig.update_yaxes(color='#b7c4d6', title_font=dict(color='#8ea9c5'), tickfont=dict(color='#b7c4d6'))
    return fig


@st.cache_data(ttl=600)
def load_data():
    proc = processed_dir()
    seg_path = os.path.join(proc, 'user_data_segmented.csv')
    data_path = seg_path if os.path.exists(seg_path) else os.path.join(proc, 'user_data.csv')
    header = pd.read_csv(data_path, nrows=0)
    parse_cols = [
        col
        for col in ['signup_date', 'last_active_date', 'first_purchase_date', 'last_purchase_date']
        if col in header.columns
    ]
    df = pd.read_csv(data_path, parse_dates=parse_cols)

    for col in ['churn', 'has_purchased', 'total_sessions', 'total_purchases', 'churn_eligible']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    for col in ['RFM_Segment', 'cluster_label', 'risk_tier']:
        if col not in df.columns:
            df[col] = 'Not Available'
        df[col] = df[col].fillna('Not Available').astype(str)

    df['age_group'] = pd.cut(
        df['age'],
        bins=[17, 24, 34, 44, 59],
        labels=['18-24', '25-34', '35-44', '45-59'],
    )
    df['signup_month'] = df['signup_date'].dt.to_period('M').astype(str)
    return df


@st.cache_data(ttl=600)
def load_project_metrics():
    path = os.path.join(processed_dir(), 'project_metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data(ttl=600)
def load_marketing_summary():
    path = os.path.join(processed_dir(), 'marketing_summary.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_funnel_summary():
    events_path = os.path.join(raw_dir(), 'events.csv')
    if not os.path.exists(events_path):
        return pd.DataFrame()
    events = pd.read_csv(events_path)
    stages = ['visit', 'signup', 'add_to_cart', 'purchase']
    rows = []
    previous = None
    for stage in stages:
        users = events.loc[events['event_type'].eq(stage), 'user_id'].nunique()
        rows.append({
            'stage': stage.replace('_', ' ').title(),
            'users': int(users),
            'stage_conversion': 1.0 if previous is None or previous == 0 else users / previous,
            'dropoff': 0 if previous is None else previous - users,
        })
        previous = users
    return pd.DataFrame(rows)


def filter_options(df, column):
    return sorted([x for x in df[column].dropna().unique().tolist()])


def summarize(df):
    buyers = df[df['has_purchased'].eq(1)]
    return {
        'users': len(df),
        'revenue': float(df['total_revenue'].sum()),
        'conversion': float(df['has_purchased'].mean()) if len(df) else 0,
        'churn': float(df['churn'].mean()) if len(df) else 0,
        'ltv': float(buyers['total_revenue'].mean()) if len(buyers) else 0,
        'aov': float(buyers['avg_order_value'].mean()) if len(buyers) else 0,
        'buyers': len(buyers),
    }


try:
    df_full = load_data()
    project_metrics = load_project_metrics()
    marketing_summary = load_marketing_summary()
    funnel_summary = load_funnel_summary()
except FileNotFoundError as exc:
    st.error(f'Data file not found: {exc}. Run python scripts/pipeline.py first.')
    st.stop()
except Exception as exc:
    st.error(f'Error loading data: {exc}')
    st.stop()


st.sidebar.title('Filters')
if st.sidebar.button('Reload data', use_container_width=True):
    st.cache_data.clear()
    st.rerun()

channels = st.sidebar.multiselect(
    'Acquisition Channel',
    options=filter_options(df_full, 'acquisition_channel'),
    default=filter_options(df_full, 'acquisition_channel'),
)
devices = st.sidebar.multiselect(
    'Device',
    options=filter_options(df_full, 'device'),
    default=filter_options(df_full, 'device'),
)
genders = st.sidebar.multiselect(
    'Gender',
    options=filter_options(df_full, 'gender'),
    default=filter_options(df_full, 'gender'),
)
risk_tiers = st.sidebar.multiselect(
    'Risk Tier',
    options=[r for r in RISK_ORDER if r in df_full['risk_tier'].unique()],
    default=[r for r in RISK_ORDER if r in df_full['risk_tier'].unique()],
)
segments = st.sidebar.multiselect(
    'RFM Segment',
    options=[s for s in RFM_ORDER if s in df_full['RFM_Segment'].unique()],
    default=[s for s in RFM_ORDER if s in df_full['RFM_Segment'].unique()],
)
churn_filter = st.sidebar.radio('Churn Status', ['All', 'Active', 'Churned'], index=0)
age_range = st.sidebar.slider(
    'Age Range',
    int(df_full['age'].min()),
    int(df_full['age'].max()),
    (int(df_full['age'].min()), int(df_full['age'].max())),
)

df = df_full[
    df_full['acquisition_channel'].isin(channels)
    & df_full['device'].isin(devices)
    & df_full['gender'].isin(genders)
    & df_full['risk_tier'].isin(risk_tiers)
    & df_full['RFM_Segment'].isin(segments)
    & df_full['age'].between(*age_range)
].copy()

if churn_filter == 'Active':
    df = df[df['churn'].eq(0)]
elif churn_filter == 'Churned':
    df = df[df['churn'].eq(1)]

baseline = summarize(df_full)
selected = summarize(df)
filters_active = len(df) != len(df_full)

st.sidebar.markdown('---')
st.sidebar.metric('Selected Users', format_num(selected['users']))
st.sidebar.metric('Selected Revenue', format_rs(selected['revenue']))
if project_metrics.get('model_auc') is not None:
    st.sidebar.metric('Model AUC', f'{project_metrics["model_auc"]:.3f}')
st.sidebar.caption(f'Last pipeline run: {project_metrics.get("generated_at", "not available")}')

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Mirae Asset Analytics Command Center</div>
        <div class="hero-subtitle">Executive intelligence for acquisition, revenue, churn risk, and customer segmentation.</div>
        <div class="hero-strip">
            <span style="background:#16e0b5"></span>
            <span style="background:#23b7f2"></span>
            <span style="background:#ffb84d"></span>
            <span style="background:#ff4d6d"></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.warning('No users match the selected filters.')
    st.stop()

tab_exec, tab_acq, tab_rev, tab_churn, tab_seg, tab_data = st.tabs([
    'Executive',
    'Acquisition',
    'Revenue',
    'Churn & Risk',
    'Segmentation',
    'Data',
])

with tab_exec:
    top_metrics = [
        ('Users', selected['users'], baseline['users'], 'num', False, ''),
        ('Revenue', selected['revenue'], baseline['revenue'], 'rs', False, 'good'),
        ('Conversion', selected['conversion'], baseline['conversion'], 'pct', False, 'good'),
        ('Churn', selected['churn'], baseline['churn'], 'pct', True, 'bad'),
        ('Avg LTV', selected['ltv'], baseline['ltv'], 'rs', False, 'warn'),
        ('Avg AOV', selected['aov'], baseline['aov'], 'rs', False, 'warn'),
    ]
    kpi_cards = []
    for label, current, base, kind, inverse, tone in top_metrics:
        if kind == 'rs':
            value = format_rs(current)
            delta = metric_delta(current, base)
        elif kind == 'pct':
            value = format_pct(current)
            delta = metric_delta(current, base, pct=True)
        else:
            value = format_num(current)
            delta = metric_delta(current, base)
        delta_text = None
        if filters_active and delta is not None:
            if kind == 'pct':
                delta_text = f"{delta:+.2f} pp vs full data"
            elif kind == 'rs':
                delta_text = f"{format_rs(delta)} vs full data"
            else:
                delta_text = f"{delta:+,.0f} vs full data"
            if inverse and delta > 0:
                delta_text = f"higher risk, {delta_text}"
        kpi_cards.append((label, value, delta_text, tone))
    render_kpi_grid(kpi_cards)

    st.markdown('---')
    c1, c2 = st.columns([1.15, 1])
    with c1:
        if not funnel_summary.empty:
            fig = go.Figure(go.Funnel(
                y=funnel_summary['stage'],
                x=funnel_summary['users'],
                textinfo='value+percent initial',
                marker=dict(color=[COLOR_BLUE, COLOR_TEAL, COLOR_AMBER, COLOR_GREEN]),
            ))
            fig.update_layout(title='Visitor Funnel')
            st.plotly_chart(apply_common_layout(fig, 360), use_container_width=True)
    with c2:
        st.subheader('Current Baseline')
        callout_rows = [
            ('Registered users', format_num(project_metrics.get('total_users', baseline['users']))),
            ('Total revenue', format_rs(project_metrics.get('total_revenue', baseline['revenue']))),
            ('Best paid CAC', f"{project_metrics.get('best_paid_cac_channel', 'n/a')} ({format_rs(project_metrics.get('best_paid_cac_per_buyer', 0))})"),
            ('Best paid ROAS', f"{project_metrics.get('best_roas_channel', 'n/a')} ({project_metrics.get('best_roas', 0):.1f}x)"),
            ('Model training rows', format_num(project_metrics.get('model_training_rows', 0))),
        ]
        st.dataframe(
            pd.DataFrame(callout_rows, columns=['Signal', 'Value']),
            use_container_width=True,
            hide_index=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        risk_counts = df['risk_tier'].value_counts().reindex(RISK_ORDER).dropna().reset_index()
        risk_counts.columns = ['risk_tier', 'users']
        fig = px.bar(
            risk_counts,
            x='risk_tier',
            y='users',
            color='risk_tier',
            color_discrete_map={
                'Low Risk': COLOR_GREEN,
                'Medium Risk': COLOR_AMBER,
                'High Risk': COLOR_RED,
                'Critical Risk': '#7A1E1E',
                'Not Observable': COLOR_GRAY,
            },
            title='Risk Tier Distribution',
        )
        st.plotly_chart(apply_common_layout(fig, 330), use_container_width=True)
    with c4:
        channel_rev = df.groupby('acquisition_channel', as_index=False).agg(
            revenue=('total_revenue', 'sum'),
            users=('user_id', 'count'),
        ).sort_values('revenue', ascending=False)
        fig = px.bar(
            channel_rev,
            x='acquisition_channel',
            y='revenue',
            color='acquisition_channel',
            color_discrete_sequence=PALETTE,
            title='Revenue by Acquisition Channel',
            text=channel_rev['revenue'].map(lambda x: f'{x / 1_000_000:.1f}M'),
        )
        fig.update_traces(textposition='outside')
        fig.update_yaxes(tickprefix='Rs ')
        st.plotly_chart(apply_common_layout(fig, 330), use_container_width=True)

with tab_acq:
    st.subheader('Acquisition & Paid Efficiency')
    if marketing_summary.empty:
        st.info('marketing_summary.csv is not available.')
    else:
        paid_summary = marketing_summary[marketing_summary['is_paid_channel'].astype(bool)].copy()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                paid_summary,
                x='CAC_per_buyer',
                y='ROAS',
                size='buyers',
                color='channel',
                color_discrete_sequence=PALETTE,
                hover_data=['users', 'buyers', 'total_spend', 'total_revenue'],
                title='Paid Channel Efficiency',
            )
            fig.update_xaxes(title='CAC per Buyer (Rs)')
            fig.update_yaxes(title='ROAS (x)')
            st.plotly_chart(apply_common_layout(fig, 390), use_container_width=True)
        with c2:
            fig = px.bar(
                marketing_summary.sort_values('conversion_rate', ascending=False),
                x='channel',
                y='conversion_rate',
                color='channel',
                color_discrete_sequence=PALETTE,
                title='Conversion Rate by Channel',
                text=marketing_summary.sort_values('conversion_rate', ascending=False)['conversion_rate'].map(lambda x: f'{x * 100:.1f}%'),
            )
            fig.update_traces(textposition='outside')
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(apply_common_layout(fig, 390), use_container_width=True)

        scorecard = marketing_summary.copy()
        scorecard['conversion_rate'] = (scorecard['conversion_rate'] * 100).round(1)
        scorecard['churn_rate'] = (scorecard['churn_rate'] * 100).round(1)
        scorecard['ROAS'] = scorecard['ROAS'].round(1)
        scorecard['CAC_per_buyer'] = scorecard['CAC_per_buyer'].round(0).astype(int)
        scorecard['total_revenue'] = scorecard['total_revenue'].round(0).astype(int)
        scorecard['total_spend'] = scorecard['total_spend'].round(0).astype(int)
        st.dataframe(
            scorecard[[
                'channel',
                'users',
                'buyers',
                'conversion_rate',
                'churn_rate',
                'total_spend',
                'CAC_per_buyer',
                'ROAS',
                'total_revenue',
            ]],
            use_container_width=True,
            hide_index=True,
        )

with tab_rev:
    st.subheader('Revenue Performance')
    c1, c2 = st.columns(2)
    with c1:
        monthly = df.groupby('signup_month', as_index=False).agg(
            revenue=('total_revenue', 'sum'),
            users=('user_id', 'count'),
        ).sort_values('signup_month')
        fig = px.line(monthly, x='signup_month', y='revenue', markers=True, title='Revenue by Signup Cohort')
        fig.update_yaxes(tickprefix='Rs ')
        st.plotly_chart(apply_common_layout(fig, 360), use_container_width=True)
    with c2:
        state_rev = df.groupby('state', as_index=False)['total_revenue'].sum().nlargest(10, 'total_revenue')
        fig = px.bar(
            state_rev.sort_values('total_revenue'),
            x='total_revenue',
            y='state',
            orientation='h',
            color='total_revenue',
            color_continuous_scale='Blues',
            title='Top 10 States by Revenue',
        )
        fig.update_xaxes(tickprefix='Rs ')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_common_layout(fig, 360), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        device_rev = df.groupby('device', as_index=False)['total_revenue'].sum()
        fig = px.pie(device_rev, names='device', values='total_revenue', color_discrete_sequence=PALETTE, title='Revenue by Device')
        st.plotly_chart(apply_common_layout(fig, 350), use_container_width=True)
    with c4:
        buyers_df = df[df['has_purchased'].eq(1)].sort_values('total_revenue', ascending=False).copy()
        if len(buyers_df):
            buyers_df['buyer_percentile'] = np.arange(1, len(buyers_df) + 1) / len(buyers_df) * 100
            buyers_df['cumulative_revenue_pct'] = buyers_df['total_revenue'].cumsum() / buyers_df['total_revenue'].sum() * 100
            fig = px.line(
                buyers_df,
                x='buyer_percentile',
                y='cumulative_revenue_pct',
                title='Revenue Concentration',
            )
            fig.add_hline(y=80, line_dash='dash', line_color=COLOR_RED)
            fig.update_xaxes(title='% of buyers ranked by revenue')
            fig.update_yaxes(title='Cumulative revenue %')
            st.plotly_chart(apply_common_layout(fig, 350), use_container_width=True)

with tab_churn:
    st.subheader('Churn & Risk')
    c1, c2, c3 = st.columns(3)
    high_risk = df[df['risk_tier'].isin(['High Risk', 'Critical Risk'])]
    critical = df[df['risk_tier'].eq('Critical Risk')]
    observable = df[df['risk_tier'].ne('Not Observable')]
    c1.metric('Observable Users', format_num(len(observable)))
    c2.metric('High+Critical Risk Revenue', format_rs(high_risk['total_revenue'].sum()))
    c3.metric('Critical Risk Users', format_num(len(critical)))

    c4, c5 = st.columns(2)
    with c4:
        churn_channel = df.groupby('acquisition_channel', as_index=False).agg(churn_rate=('churn', 'mean'))
        fig = px.bar(
            churn_channel.sort_values('churn_rate', ascending=False),
            x='acquisition_channel',
            y='churn_rate',
            color='churn_rate',
            color_continuous_scale='Reds',
            title='Churn Rate by Channel',
            text=churn_channel.sort_values('churn_rate', ascending=False)['churn_rate'].map(lambda x: f'{x * 100:.1f}%'),
        )
        fig.update_yaxes(tickformat='.0%')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_common_layout(fig, 360), use_container_width=True)
    with c5:
        if 'churn_proba' in df.columns:
            proba_df = df[df['churn_proba'].notna()].copy()
            fig = px.histogram(
                proba_df,
                x='churn_proba',
                color='churn',
                nbins=40,
                barmode='overlay',
                color_discrete_map={0: COLOR_BLUE, 1: COLOR_RED},
                title='Out-of-Fold Churn Probability',
            )
            fig.update_xaxes(tickformat='.0%')
            st.plotly_chart(apply_common_layout(fig, 360), use_container_width=True)

    risk_table = df.groupby('risk_tier', as_index=False).agg(
        users=('user_id', 'count'),
        actual_churn_rate=('churn', 'mean'),
        total_revenue=('total_revenue', 'sum'),
        avg_sessions=('total_sessions', 'mean'),
        avg_engagement=('engagement_score', 'mean'),
    )
    risk_table['risk_tier'] = pd.Categorical(risk_table['risk_tier'], categories=RISK_ORDER, ordered=True)
    risk_table = risk_table.sort_values('risk_tier')
    risk_table['actual_churn_rate'] = (risk_table['actual_churn_rate'] * 100).round(1)
    risk_table['total_revenue'] = risk_table['total_revenue'].round(0).astype(int)
    risk_table['avg_sessions'] = risk_table['avg_sessions'].round(1)
    risk_table['avg_engagement'] = risk_table['avg_engagement'].round(3)
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tab_seg:
    st.subheader('Segmentation')
    c1, c2 = st.columns(2)
    with c1:
        seg_counts = df['RFM_Segment'].value_counts().reindex([s for s in RFM_ORDER if s in df['RFM_Segment'].unique()]).dropna().reset_index()
        seg_counts.columns = ['segment', 'users']
        fig = px.bar(
            seg_counts.sort_values('users'),
            x='users',
            y='segment',
            orientation='h',
            color='segment',
            color_discrete_sequence=PALETTE,
            title='RFM Segment Size',
        )
        st.plotly_chart(apply_common_layout(fig, 380), use_container_width=True)
    with c2:
        seg_rev = df.groupby('RFM_Segment', as_index=False).agg(avg_revenue=('total_revenue', 'mean'))
        seg_rev['RFM_Segment'] = pd.Categorical(seg_rev['RFM_Segment'], categories=RFM_ORDER, ordered=True)
        seg_rev = seg_rev.sort_values('avg_revenue')
        fig = px.bar(
            seg_rev,
            x='avg_revenue',
            y='RFM_Segment',
            orientation='h',
            color='avg_revenue',
            color_continuous_scale='Blues',
            title='Average Revenue by Segment',
        )
        fig.update_xaxes(tickprefix='Rs ')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_common_layout(fig, 380), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        cluster_counts = df['cluster_label'].value_counts().reindex([c for c in CLUSTER_ORDER if c in df['cluster_label'].unique()]).dropna().reset_index()
        cluster_counts.columns = ['cluster', 'users']
        fig = px.bar(cluster_counts, x='cluster', y='users', color='cluster', color_discrete_sequence=PALETTE, title='Cluster Size')
        st.plotly_chart(apply_common_layout(fig, 350), use_container_width=True)
    with c4:
        cluster_revenue = df.groupby('cluster_label', as_index=False).agg(
            avg_revenue=('total_revenue', 'mean'),
            churn_rate=('churn', 'mean'),
            users=('user_id', 'count'),
        )
        fig = px.scatter(
            cluster_revenue,
            x='churn_rate',
            y='avg_revenue',
            size='users',
            color='cluster_label',
            color_discrete_sequence=PALETTE,
            title='Cluster Value vs Churn',
        )
        fig.update_xaxes(tickformat='.0%', title='Churn rate')
        fig.update_yaxes(tickprefix='Rs ', title='Avg revenue')
        st.plotly_chart(apply_common_layout(fig, 350), use_container_width=True)

    seg_table = df.groupby('RFM_Segment', as_index=False).agg(
        users=('user_id', 'count'),
        conversion=('has_purchased', 'mean'),
        churn_rate=('churn', 'mean'),
        avg_revenue=('total_revenue', 'mean'),
        avg_sessions=('total_sessions', 'mean'),
    )
    seg_table['RFM_Segment'] = pd.Categorical(seg_table['RFM_Segment'], categories=RFM_ORDER, ordered=True)
    seg_table = seg_table.sort_values('RFM_Segment')
    seg_table['conversion'] = (seg_table['conversion'] * 100).round(1)
    seg_table['churn_rate'] = (seg_table['churn_rate'] * 100).round(1)
    seg_table['avg_revenue'] = seg_table['avg_revenue'].round(0).astype(int)
    seg_table['avg_sessions'] = seg_table['avg_sessions'].round(1)
    st.dataframe(seg_table, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader('Filtered User Table')
    cols_show = [
        'user_id',
        'acquisition_channel',
        'device',
        'gender',
        'age',
        'risk_tier',
        'RFM_Segment',
        'cluster_label',
        'total_sessions',
        'total_revenue',
        'has_purchased',
        'churn',
        'engagement_score',
        'churn_proba',
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    sort_col = st.selectbox('Sort by', ['total_revenue', 'churn_proba', 'total_sessions', 'engagement_score'], index=0)
    ascending = st.toggle('Ascending', value=False)
    preview = df[cols_show].sort_values(sort_col, ascending=ascending).head(1000)
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.download_button(
        'Download filtered CSV',
        data=df[cols_show].to_csv(index=False),
        file_name='mirae_filtered_users.csv',
        mime='text/csv',
        use_container_width=True,
    )
