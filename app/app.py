import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title='Mirae Asset Analytics',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

BLUE   = '#4C72B0'
ORANGE = '#DD8452'
GREEN  = '#55A868'
RED    = '#C44E52'
PURPLE = '#8172B2'
TEAL   = '#64B5CD'
PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL]

def clean_chart(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = os.path.join(base, 'data', 'processed')
    df = pd.read_csv(
        os.path.join(proc, 'user_data.csv'),
        parse_dates=['signup_date', 'last_active_date', 'first_purchase_date']
    )
    df['churn']           = df['churn'].astype(int)
    df['has_purchased']   = df['has_purchased'].astype(int)
    df['total_sessions']  = df['total_sessions'].astype(int)
    df['total_purchases'] = df['total_purchases'].astype(int)
    df['age_group'] = pd.cut(
        df['age'], bins=[17, 24, 34, 44, 59],
        labels=['18-24', '25-34', '35-44', '45-59']
    )
    seg_path = os.path.join(proc, 'user_data_segmented.csv')
    if os.path.exists(seg_path):
        seg = pd.read_csv(seg_path, usecols=['user_id', 'RFM_Segment', 'cluster_label'])
        df = df.merge(seg, on='user_id', how='left')
    else:
        df['RFM_Segment']   = None
        df['cluster_label'] = None
    return df

df_full = load_data()

st.sidebar.title('🔽 Filters')
channels = st.sidebar.multiselect('Acquisition Channel',
    options=sorted(df_full['acquisition_channel'].unique()),
    default=sorted(df_full['acquisition_channel'].unique()))
devices  = st.sidebar.multiselect('Device',
    options=sorted(df_full['device'].unique()),
    default=sorted(df_full['device'].unique()))
genders  = st.sidebar.multiselect('Gender',
    options=sorted(df_full['gender'].unique()),
    default=sorted(df_full['gender'].unique()))
churn_f  = st.sidebar.radio('Churn Status', ['All', 'Active', 'Churned'], index=0)
age_r    = st.sidebar.slider('Age Range',
    int(df_full['age'].min()), int(df_full['age'].max()),
    (int(df_full['age'].min()), int(df_full['age'].max())))

df = df_full[
    df_full['acquisition_channel'].isin(channels) &
    df_full['device'].isin(devices) &
    df_full['gender'].isin(genders) &
    df_full['age'].between(*age_r)
].copy()
if churn_f == 'Active':    df = df[df['churn'] == 0]
elif churn_f == 'Churned': df = df[df['churn'] == 1]
st.sidebar.markdown(f'**{len(df):,}** users selected')
st.sidebar.markdown('---')
st.sidebar.markdown('**Phase coverage:** 23 / 27 ✅')
st.sidebar.markdown('**Model AUC:** 0.84 (GBM)')

st.title('📊 Mirae Asset Digital Platform — Analytics Dashboard')
st.markdown('*27 phases · 9 notebooks · Full analytics lifecycle · Synthetic fintech dataset*')
st.markdown('---')

tab1, tab2, tab3, tab4 = st.tabs([
    '📈 KPI Overview', '💰 Revenue Analysis',
    '📉 Churn Analysis', '🧠 User Segmentation'
])

with tab1:
    st.subheader('Key Performance Indicators')
    buyers = df[df['has_purchased'] == 1]
    bf     = df_full[df_full['has_purchased'] == 1]
    metrics = [
        ('Total Users',      len(df),                                         len(df_full),                   ''),
        ('Total Revenue',    df['total_revenue'].sum(),                        df_full['total_revenue'].sum(), 'Rs'),
        ('Churn Rate',       df['churn'].mean(),                               df_full['churn'].mean(),        'pct'),
        ('Conversion Rate',  df['has_purchased'].mean(),                       df_full['has_purchased'].mean(),'pct'),
        ('Avg LTV (buyers)', buyers['total_revenue'].mean() if len(buyers)>0 else 0, bf['total_revenue'].mean(),'Rs'),
        ('Avg AOV',          buyers['avg_order_value'].mean() if len(buyers)>0 else 0, bf['avg_order_value'].mean(),'Rs'),
    ]
    cols = st.columns(3)
    for i, (name, val, base, fmt) in enumerate(metrics):
        with cols[i % 3]:
            if fmt == 'pct':  st.metric(name, f'{val*100:.1f}%',  f'{(val-base)*100:+.2f}pp')
            elif fmt == 'Rs': st.metric(name, f'Rs {val:,.0f}',   f'Rs {val-base:+,.0f}')
            else:             st.metric(name, f'{val:,}',          f'{int(val-base):+,}')
    st.markdown('---')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Engagement Score Distribution**')
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df['engagement_score'], bins=40, color=BLUE, edgecolor='white', linewidth=0.4, alpha=0.85)
        ax.axvline(df['engagement_score'].mean(), color=RED, linewidth=2, linestyle='--',
                   label=f'Mean: {df["engagement_score"].mean():.3f}')
        ax.set_xlabel('Engagement Score'); ax.set_ylabel('Users'); ax.legend(fontsize=8)
        clean_chart(ax); st.pyplot(fig); plt.close()
    with c2:
        st.markdown('**Sessions vs Revenue (by Churn Status)**')
        fig, ax = plt.subplots(figsize=(6, 3))
        for lbl, col, name in [(0, BLUE, 'Active'), (1, RED, 'Churned')]:
            sub = df[df['churn'] == lbl]
            ax.scatter(sub['total_sessions'], sub['total_revenue'],
                       alpha=0.15, s=8, color=col, label=f'{name} ({len(sub):,})')
        ax.set_xlabel('Total Sessions'); ax.set_ylabel('Total Revenue (Rs)')
        ax.legend(fontsize=8); clean_chart(ax); st.pyplot(fig); plt.close()

with tab2:
    st.subheader('Revenue Analysis')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Revenue by Signup Cohort**')
        mr = df.groupby(df['signup_date'].dt.to_period('M').astype(str))['total_revenue'].sum().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(mr)), mr.values / 1e6, color=BLUE, edgecolor='white', linewidth=0.4)
        ax.set_xticks(range(len(mr)))
        ax.set_xticklabels(mr.index, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Rs M'); clean_chart(ax); st.pyplot(fig); plt.close()
    with c2:
        st.markdown('**Revenue by Acquisition Channel**')
        cr = df.groupby('acquisition_channel')['total_revenue'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(cr.index, cr.values / 1e6, color=PALETTE[:len(cr)], edgecolor='white', linewidth=0.4)
        for i, v in enumerate(cr.values / 1e6):
            ax.text(i, v + 0.02, f'Rs{v:.1f}M', ha='center', fontsize=7, fontweight='bold')
        ax.set_ylabel('Rs M'); ax.set_xticklabels(cr.index, rotation=15, ha='right', fontsize=8)
        clean_chart(ax); st.pyplot(fig); plt.close()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('**Revenue by Device**')
        dr = df.groupby('device')['total_revenue'].sum()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(dr, labels=dr.index, autopct='%1.1f%%', colors=[BLUE, ORANGE],
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        st.pyplot(fig); plt.close()
    with c4:
        st.markdown('**Top 10 States by Revenue**')
        sr = df.groupby('state')['total_revenue'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(sr.index[::-1], sr.values[::-1] / 1e6, color=BLUE, edgecolor='white', linewidth=0.4)
        ax.set_xlabel('Rs M'); clean_chart(ax); st.pyplot(fig); plt.close()
    st.markdown('---')
    st.markdown('**Revenue Concentration — Pareto Analysis**')
    buyers_df = df[df['has_purchased'] == 1].copy()
    if len(buyers_df) > 0:
        buyers_df = buyers_df.sort_values('total_revenue', ascending=False)
        buyers_df['cum_pct']  = buyers_df['total_revenue'].cumsum() / buyers_df['total_revenue'].sum() * 100
        buyers_df['user_pct'] = np.arange(1, len(buyers_df) + 1) / len(buyers_df) * 100
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(buyers_df['user_pct'], buyers_df['cum_pct'], color=BLUE, linewidth=2)
        ax.axhline(80, color=RED, linestyle='--', linewidth=1, label='80% revenue threshold')
        ax.fill_between(buyers_df['user_pct'], buyers_df['cum_pct'], alpha=0.1, color=BLUE)
        ax.set_xlabel('% of Buyers (ranked by revenue)'); ax.set_ylabel('Cumulative % of Revenue')
        ax.legend(fontsize=8); clean_chart(ax); st.pyplot(fig); plt.close()

with tab3:
    st.subheader('Churn Analysis')
    avg_churn = df['churn'].mean()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Churn Rate by Channel**')
        cc = df.groupby('acquisition_channel')['churn'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [RED if v > avg_churn else GREEN for v in cc]
        ax.bar(cc.index, cc.values * 100, color=colors, edgecolor='white', linewidth=0.4)
        ax.axhline(avg_churn * 100, color='black', linestyle='--', linewidth=1.2,
                   label=f'Avg: {avg_churn*100:.1f}%')
        ax.set_ylabel('Churn Rate (%)'); ax.legend(fontsize=8)
        ax.set_xticklabels(cc.index, rotation=15, ha='right', fontsize=8)
        clean_chart(ax); st.pyplot(fig); plt.close()
    with c2:
        st.markdown('**Churn Rate by Age Group**')
        ac = df.groupby('age_group', observed=True)['churn'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(ac.index.astype(str), ac.values * 100, color=PURPLE, edgecolor='white', linewidth=0.4)
        ax.axhline(avg_churn * 100, color='black', linestyle='--', linewidth=1.2)
        for i, v in enumerate(ac.values * 100):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('Churn Rate (%)'); clean_chart(ax); st.pyplot(fig); plt.close()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('**Days Since Last Active (Active vs Churned)**')
        fig, ax = plt.subplots(figsize=(5, 3))
        for lbl, col, name in [(0, BLUE, 'Active'), (1, RED, 'Churned')]:
            s = df[df['churn'] == lbl]['days_since_last_active'].dropna()
            ax.hist(s, bins=30, alpha=0.6, color=col,
                    label=f'{name} ({len(s):,})', edgecolor='none', density=True)
        ax.set_xlabel('Days Since Last Active'); ax.legend(fontsize=8)
        clean_chart(ax); st.pyplot(fig); plt.close()
    with c4:
        st.markdown('**Churn Rate by Device**')
        dc = df.groupby('device')['churn'].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(dc.index, dc.values * 100, color=[BLUE, ORANGE], edgecolor='white', linewidth=0.4)
        for i, v in enumerate(dc.values * 100):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Churn Rate (%)'); clean_chart(ax); st.pyplot(fig); plt.close()
    st.markdown('---')
    st.markdown('**Engagement Score: Active vs Churned**')
    fig, ax = plt.subplots(figsize=(10, 2.5))
    for lbl, col, name in [(0, BLUE, 'Active'), (1, RED, 'Churned')]:
        s = df[df['churn'] == lbl]['engagement_score']
        ax.hist(s, bins=40, alpha=0.65, color=col,
                label=f'{name} — mean: {s.mean():.3f}', edgecolor='none', density=True)
    ax.set_xlabel('Engagement Score'); ax.legend(fontsize=8)
    clean_chart(ax); st.pyplot(fig); plt.close()

with tab4:
    st.subheader('User Segmentation — RFM & K-Means Clusters')
    has_seg = ('RFM_Segment' in df.columns) and df['RFM_Segment'].notna().any()
    if has_seg:
        st.markdown('### 📌 RFM Segments')
        rfm_order = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                     'Recent Customers', 'Needs Attention',
                     'Cannot Lose Them', 'At Risk', 'Lost', 'Non-Buyer']
        seg_present = [s for s in rfm_order if s in df['RFM_Segment'].values]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Segment Size**')
            seg_counts = df['RFM_Segment'].value_counts().reindex(seg_present).dropna()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            seg_colors = [GREEN if s in ['Champions', 'Loyal Customers', 'Potential Loyalists']
                          else ORANGE if s in ['Recent Customers', 'Needs Attention']
                          else RED for s in seg_counts.index]
            ax.barh(seg_counts.index[::-1], seg_counts.values[::-1],
                    color=seg_colors[::-1], edgecolor='white', linewidth=0.4)
            ax.set_xlabel('Users'); clean_chart(ax); st.pyplot(fig); plt.close()
        with c2:
            st.markdown('**Avg Revenue by Segment**')
            seg_rev = df.groupby('RFM_Segment')['total_revenue'].mean().reindex(seg_present).dropna()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(seg_rev.index[::-1], seg_rev.values[::-1],
                    color=BLUE, edgecolor='white', linewidth=0.4)
            ax.set_xlabel('Avg Revenue (Rs)'); clean_chart(ax); st.pyplot(fig); plt.close()
        st.markdown('**Segment Summary Table**')
        seg_stats = df.groupby('RFM_Segment').agg(
            Users        = ('user_id',       'count'),
            Churn_Rate   = ('churn',         'mean'),
            Conversion   = ('has_purchased', 'mean'),
            Avg_Revenue  = ('total_revenue', 'mean'),
            Avg_Sessions = ('total_sessions','mean'),
        ).reindex(seg_present).dropna().reset_index()
        seg_stats['Churn_Rate']   = (seg_stats['Churn_Rate']  * 100).round(1)
        seg_stats['Conversion']   = (seg_stats['Conversion']  * 100).round(1)
        seg_stats['Avg_Revenue']  = seg_stats['Avg_Revenue'].round(0).astype(int)
        seg_stats['Avg_Sessions'] = seg_stats['Avg_Sessions'].round(1)
        seg_stats.columns = ['Segment', 'Users', 'Churn Rate (%)',
                              'Conversion (%)', 'Avg Revenue (Rs)', 'Avg Sessions']
        st.dataframe(seg_stats, use_container_width=True, hide_index=True)
        st.markdown('---')
        st.markdown('### 🔵 K-Means Cluster Profiles')
        cl_order = ['VIP / Champions', 'Engaged Regulars', 'Casual Buyers',
                    'Dormant / At-Risk', 'Non-Buyer']
        cl_present = [c for c in cl_order if c in df['cluster_label'].values]
        c3, c4 = st.columns(2)
        with c3:
            st.markdown('**Cluster Size**')
            cl_counts = df['cluster_label'].value_counts().reindex(cl_present).dropna()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(cl_counts.index, cl_counts.values,
                   color=PALETTE[:len(cl_counts)], edgecolor='white', linewidth=0.4)
            for i, v in enumerate(cl_counts.values):
                ax.text(i, v + 15, f'{v:,}', ha='center', fontsize=8, fontweight='bold')
            ax.set_ylabel('Users')
            ax.set_xticklabels(cl_counts.index, rotation=15, ha='right', fontsize=8)
            clean_chart(ax); st.pyplot(fig); plt.close()
        with c4:
            st.markdown('**Avg Revenue by Cluster**')
            cl_rev = df.groupby('cluster_label')['total_revenue'].mean().reindex(cl_present).dropna()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(cl_rev.index, cl_rev.values,
                   color=PALETTE[:len(cl_rev)], edgecolor='white', linewidth=0.4)
            ax.set_ylabel('Avg Revenue (Rs)')
            ax.set_xticklabels(cl_rev.index, rotation=15, ha='right', fontsize=8)
            clean_chart(ax); st.pyplot(fig); plt.close()
        st.markdown('**Cluster Deep-Dive Table**')
        cl_stats = df.groupby('cluster_label').agg(
            Users         = ('user_id',         'count'),
            Churn_Rate    = ('churn',            'mean'),
            Conversion    = ('has_purchased',    'mean'),
            Avg_Revenue   = ('total_revenue',    'mean'),
            Avg_Sessions  = ('total_sessions',   'mean'),
            Avg_Engagement= ('engagement_score', 'mean'),
        ).reindex(cl_present).dropna().reset_index()
        cl_stats['Churn_Rate']     = (cl_stats['Churn_Rate']     * 100).round(1)
        cl_stats['Conversion']     = (cl_stats['Conversion']     * 100).round(1)
        cl_stats['Avg_Revenue']    = cl_stats['Avg_Revenue'].round(0).astype(int)
        cl_stats['Avg_Sessions']   = cl_stats['Avg_Sessions'].round(1)
        cl_stats['Avg_Engagement'] = cl_stats['Avg_Engagement'].round(3)
        cl_stats.columns = ['Cluster', 'Users', 'Churn Rate (%)', 'Conversion (%)',
                             'Avg Revenue (Rs)', 'Avg Sessions', 'Avg Engagement']
        st.dataframe(cl_stats, use_container_width=True, hide_index=True)
    else:
        st.info('Segmented data not found. Run NB06 to generate user_data_segmented.csv, then relaunch.')
    st.markdown('---')
    st.markdown('**📥 Download Filtered Data**')
    cols_show = ['user_id', 'acquisition_channel', 'device', 'gender', 'age',
                 'total_sessions', 'total_revenue', 'has_purchased', 'churn',
                 'engagement_score', 'RFM_Segment', 'cluster_label']
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(
        df[cols_show].sort_values('total_revenue', ascending=False).head(500),
        use_container_width=True
    )
    st.download_button(
        '⬇ Download filtered data as CSV',
        data=df[cols_show].to_csv(index=False),
        file_name='mirae_filtered.csv',
        mime='text/csv'
    )
