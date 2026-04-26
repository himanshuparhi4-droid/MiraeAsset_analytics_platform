import json
import os
from datetime import datetime

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')
MODELS = os.path.join(BASE, 'models')
MODEL_ARTIFACTS_PATH = os.path.join(MODELS, 'model_artifacts.pkl')

CHANNEL_ORDER = ['Facebook Ads', 'Google Ads', 'Organic', 'Referral']
FUNNEL_STAGES = ['visit', 'signup', 'add_to_cart', 'purchase']
CAMPAIGN_CHANNEL_MAP = {
    'Facebook Ads': 'Facebook Ads',
    'Google Ads': 'Google Ads',
    'Organic': 'Organic',
    'Referral': 'Referral',
}

GBM_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_model_artifacts(**updates):
    """Persist all reusable model objects in one versioned bundle."""
    os.makedirs(MODELS, exist_ok=True)
    if os.path.exists(MODEL_ARTIFACTS_PATH):
        artifacts = load(MODEL_ARTIFACTS_PATH)
    else:
        artifacts = {}
    artifacts.update(updates)
    artifacts['updated_at'] = datetime.now().isoformat(timespec='seconds')
    dump(artifacts, MODEL_ARTIFACTS_PATH)


def validate_funnel_events(events):
    first_events = events.pivot_table(
        index='user_id',
        columns='event_type',
        values='event_date',
        aggfunc='min',
    )
    missing_stages = [stage for stage in FUNNEL_STAGES if stage not in first_events.columns]
    if missing_stages:
        raise ValueError(f'events.csv is missing funnel stages: {missing_stages}')

    stage_masks = {}
    prior_mask = pd.Series(True, index=first_events.index)
    prior_date = None
    for stage in FUNNEL_STAGES:
        stage_date = first_events[stage]
        mask = prior_mask & stage_date.notna()
        if prior_date is not None:
            mask &= stage_date >= prior_date
        stage_masks[stage] = mask
        prior_mask = mask
        prior_date = stage_date

    counts = {stage: int(mask.sum()) for stage, mask in stage_masks.items()}
    if any(counts[FUNNEL_STAGES[i]] < counts[FUNNEL_STAGES[i + 1]] for i in range(len(FUNNEL_STAGES) - 1)):
        raise ValueError(f'Funnel counts are not monotonic: {counts}')

    invalid_purchase = first_events['purchase'].notna() & ~stage_masks['add_to_cart']
    if invalid_purchase.any():
        raise ValueError(
            f'events.csv has {int(invalid_purchase.sum()):,} purchase users without a prior funnel path'
        )

    return counts


def load_raw_data():
    users = pd.read_csv(os.path.join(RAW, 'users.csv'), parse_dates=['signup_date'])
    sessions = pd.read_csv(os.path.join(RAW, 'sessions.csv'), parse_dates=['session_date'])
    txns = pd.read_csv(os.path.join(RAW, 'transactions.csv'), parse_dates=['transaction_date'])
    events = pd.read_csv(os.path.join(RAW, 'events.csv'), parse_dates=['event_date'])
    campaigns = pd.read_csv(os.path.join(RAW, 'campaigns.csv'), parse_dates=['start_date'])
    return users, sessions, txns, events, campaigns


def validate_raw_data(users, sessions, txns, events, campaigns):
    for name, df in [
        ('users', users),
        ('sessions', sessions),
        ('transactions', txns),
        ('events', events),
        ('campaigns', campaigns),
    ]:
        nulls = int(df.isnull().sum().sum())
        if nulls > 0:
            log(f'  WARNING: {name} has {nulls:,} null values')

    if users['user_id'].duplicated().any():
        raise ValueError('users.csv has duplicate user_id values')

    user_dates = users[['user_id', 'signup_date']]

    sessions_checked = sessions.merge(user_dates, on='user_id', how='left')
    txns_checked = txns.merge(user_dates, on='user_id', how='left')
    events_checked = events.merge(user_dates, on='user_id', how='left')

    for name, df in [
        ('sessions', sessions_checked),
        ('transactions', txns_checked),
        ('events', events_checked),
    ]:
        missing = int(df['signup_date'].isna().sum())
        if missing:
            raise ValueError(f'{name}.csv contains {missing:,} rows with unknown user_id')

    invalid_sessions = sessions_checked['session_date'] < sessions_checked['signup_date']
    invalid_txns = txns_checked['transaction_date'] < txns_checked['signup_date']
    invalid_events = events_checked['event_date'] < events_checked['signup_date']

    if invalid_sessions.any():
        log(f'  WARNING: dropping {int(invalid_sessions.sum()):,} pre-signup sessions')
    if invalid_txns.any():
        log(f'  WARNING: dropping {int(invalid_txns.sum()):,} pre-signup transactions')
    if invalid_events.any():
        log(f'  WARNING: dropping {int(invalid_events.sum()):,} pre-signup events')

    sessions = sessions_checked.loc[~invalid_sessions, sessions.columns].copy()
    txns = txns_checked.loc[~invalid_txns, txns.columns].copy()
    events = events_checked.loc[~invalid_events, events.columns].copy()
    funnel_counts = validate_funnel_events(events)
    log(
        '  Funnel validation: '
        + ' -> '.join(f'{stage}={funnel_counts[stage]:,}' for stage in FUNNEL_STAGES)
    )

    campaigns = campaigns.copy()
    campaigns['channel'] = campaigns['channel'].map(CAMPAIGN_CHANNEL_MAP).fillna(campaigns['channel'])
    campaigns.loc[campaigns['channel'].eq('Organic'), 'cost'] = 0

    unknown_campaign_channels = sorted(set(campaigns['channel']) - set(CHANNEL_ORDER))
    if unknown_campaign_channels:
        raise ValueError(f'Unknown campaign channels: {unknown_campaign_channels}')

    return sessions, txns, events, campaigns


def build_user_data(users, sessions, txns):
    os.makedirs(MODELS, exist_ok=True)

    if 'is_registered' in users.columns:
        registered_users = users[users['is_registered'].fillna(1).astype(int).eq(1)].copy()
    else:
        registered_users = users.copy()

    sess_agg = sessions.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        avg_session_duration=('duration_minutes', 'mean'),
        total_pages_viewed=('pages_viewed', 'sum'),
        last_active_date=('session_date', 'max'),
    ).reset_index()

    txn_agg = txns.groupby('user_id').agg(
        total_revenue=('amount', 'sum'),
        total_purchases=('transaction_id', 'count'),
        avg_order_value=('amount', 'mean'),
        first_purchase_date=('transaction_date', 'min'),
        last_purchase_date=('transaction_date', 'max'),
    ).reset_index()

    ud = registered_users.copy()
    ud = ud.merge(sess_agg, on='user_id', how='left')
    ud = ud.merge(txn_agg, on='user_id', how='left')

    fill_zero_cols = [
        'total_sessions',
        'total_pages_viewed',
        'total_revenue',
        'total_purchases',
        'avg_order_value',
        'avg_session_duration',
    ]
    for col in fill_zero_cols:
        ud[col] = ud[col].fillna(0)

    latest = sessions['session_date'].max()
    ud['days_since_signup'] = (latest - ud['signup_date']).dt.days
    ud['days_since_last_active'] = (
        (latest - ud['last_active_date']).dt.days.fillna(ud['days_since_signup'])
    )
    ud['avg_revenue_per_purchase'] = np.where(
        ud['total_purchases'] > 0,
        ud['total_revenue'] / ud['total_purchases'],
        0,
    )
    ud['revenue_per_session'] = np.where(
        ud['total_sessions'] > 0,
        ud['total_revenue'] / ud['total_sessions'],
        0,
    )
    ud['has_purchased'] = (ud['total_purchases'] > 0).astype(int)

    ud['engagement_score_raw'] = (
        ud['total_sessions'] * 0.5
        + ud['avg_session_duration'] * 0.3
        + ud['total_purchases'] * 0.2
    )
    engagement_scaler = MinMaxScaler()
    ud['engagement_score'] = engagement_scaler.fit_transform(ud[['engagement_score_raw']])
    save_model_artifacts(engagement_score_scaler=engagement_scaler)

    ud['churn_eligible'] = (ud['days_since_signup'] >= 30).astype(int)
    inactive_over_30d = ud['last_active_date'].isna() | ((latest - ud['last_active_date']).dt.days > 30)
    ud['churn'] = np.where(ud['churn_eligible'].eq(1) & inactive_over_30d, 1, 0)

    int_cols = [
        'total_sessions',
        'total_pages_viewed',
        'total_purchases',
        'has_purchased',
        'churn_eligible',
        'churn',
    ]
    for col in int_cols:
        ud[col] = ud[col].astype(int)

    assert len(ud) == len(registered_users)
    assert 0.1 < ud['churn'].mean() < 0.9
    return ud


def quantile_score(series, larger_is_better=True):
    """Return stable 1-5 quantile scores, breaking ties before qcut."""
    if series.empty:
        return pd.Series(dtype=int, index=series.index)
    if len(series) == 1:
        return pd.Series(3, index=series.index, dtype=int)

    ranked = series.rank(method='first', ascending=larger_is_better)
    q = min(5, len(series))
    raw_score = pd.qcut(ranked, q, labels=False, duplicates='drop') + 1
    if q < 5:
        raw_score = 1 + (raw_score - 1) * (4 / max(q - 1, 1))
    return raw_score.round().clip(1, 5).astype(int)


def assign_rfm_segment(row):
    """Assign the first matching priority-ordered RFM segment."""
    r, f, m, total = row['R'], row['F'], row['M'], row['RFM_Total']
    if total >= 13:
        return 'Champions'
    elif r >= 4 and f >= 3:
        return 'Loyal Customers'
    elif r >= 3 and total >= 9:
        return 'Potential Loyalists'
    elif r >= 4 and f <= 2:
        return 'Recent Customers'
    elif r <= 2 and f >= 4:
        return 'Cannot Lose Them'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif total <= 6:
        return 'Lost'
    else:
        return 'Needs Attention'


def build_user_segments(user_data, txns):
    os.makedirs(MODELS, exist_ok=True)

    if txns.empty:
        user_enriched = user_data.copy()
        user_enriched['rfm_recency'] = -1
        user_enriched['rfm_frequency'] = 0
        user_enriched['rfm_monetary'] = 0
        user_enriched['R'] = 0
        user_enriched['F'] = 0
        user_enriched['M'] = 0
        user_enriched['RFM_Total'] = 0
        user_enriched['RFM_Segment'] = 'Non-Buyer'
        user_enriched['cluster'] = np.nan
        user_enriched['cluster_label'] = 'Non-Buyer'
        return user_enriched

    ref_date = txns['transaction_date'].max() + pd.Timedelta(days=1)
    rfm = txns.groupby('user_id').agg(
        recency=('transaction_date', lambda x: (ref_date - x.max()).days),
        frequency=('transaction_id', 'count'),
        monetary=('amount', 'sum'),
    ).reset_index()

    rfm = rfm.merge(
        user_data[[
            'user_id',
            'engagement_score',
            'total_sessions',
            'avg_session_duration',
            'churn',
            'acquisition_channel',
            'device',
            'age',
            'gender',
            'avg_order_value',
            'days_since_signup',
        ]],
        on='user_id',
        how='left',
    )

    rfm['R'] = quantile_score(rfm['recency'], larger_is_better=False)
    rfm['F'] = quantile_score(rfm['frequency'], larger_is_better=True)
    rfm['M'] = quantile_score(rfm['monetary'], larger_is_better=True)
    rfm['RFM_Total'] = rfm['R'] + rfm['F'] + rfm['M']
    rfm['RFM_Segment'] = rfm.apply(assign_rfm_segment, axis=1)

    cluster_features = ['recency', 'frequency', 'monetary']
    rfm_scaler = StandardScaler()
    x_scaled = rfm_scaler.fit_transform(rfm[cluster_features])
    n_clusters = min(4, len(rfm))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['cluster'] = km.fit_predict(x_scaled)

    cluster_stats = rfm.groupby('cluster').agg(
        size=('user_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        avg_rfm=('RFM_Total', 'mean'),
    ).round(2).sort_values('avg_rfm', ascending=False)

    cluster_names = ['VIP / Champions', 'Engaged Regulars', 'Casual Buyers', 'Dormant / At-Risk']
    label_map = {
        cluster_id: cluster_names[i]
        for i, cluster_id in enumerate(cluster_stats.index)
    }
    rfm['cluster_label'] = rfm['cluster'].map(label_map)
    save_model_artifacts(
        rfm_scaler=rfm_scaler,
        rfm_kmeans=km,
        rfm_cluster_labels={str(k): v for k, v in label_map.items()},
    )

    rfm_export = rfm[[
        'user_id',
        'recency',
        'frequency',
        'monetary',
        'R',
        'F',
        'M',
        'RFM_Total',
        'RFM_Segment',
        'cluster',
        'cluster_label',
    ]].copy()
    rfm_export = rfm_export.rename(columns={
        'recency': 'rfm_recency',
        'frequency': 'rfm_frequency',
        'monetary': 'rfm_monetary',
    })

    user_enriched = user_data.merge(rfm_export, on='user_id', how='left')
    user_enriched['RFM_Segment'] = user_enriched['RFM_Segment'].fillna('Non-Buyer')
    user_enriched['cluster_label'] = user_enriched['cluster_label'].fillna('Non-Buyer')
    user_enriched['R'] = user_enriched['R'].fillna(0).astype(int)
    user_enriched['F'] = user_enriched['F'].fillna(0).astype(int)
    user_enriched['M'] = user_enriched['M'].fillna(0).astype(int)
    user_enriched['RFM_Total'] = user_enriched['RFM_Total'].fillna(0)
    user_enriched['rfm_recency'] = user_enriched['rfm_recency'].fillna(-1)
    user_enriched['rfm_frequency'] = user_enriched['rfm_frequency'].fillna(0)
    user_enriched['rfm_monetary'] = user_enriched['rfm_monetary'].fillna(0)

    return user_enriched


def build_marketing_summary(user_data, campaigns, txns=None):
    channel_summary = user_data.groupby('acquisition_channel').agg(
        users=('user_id', 'count'),
        buyers=('has_purchased', 'sum'),
        total_revenue=('total_revenue', 'sum'),
        conversion_rate=('has_purchased', 'mean'),
        churn_rate=('churn', 'mean'),
        avg_engagement=('engagement_score', 'mean'),
    ).reindex(CHANNEL_ORDER).fillna(0)

    campaign_costs = campaigns.groupby('channel')['cost'].sum().reindex(CHANNEL_ORDER).fillna(0)
    summary = channel_summary.copy()
    summary['total_spend'] = campaign_costs
    summary['is_paid_channel'] = summary['total_spend'] > 0
    summary['CAC_all_users'] = np.where(
        summary['users'] > 0,
        summary['total_spend'] / summary['users'],
        0,
    )
    summary['CAC_per_buyer'] = np.where(
        summary['buyers'] > 0,
        summary['total_spend'] / summary['buyers'],
        0,
    )
    summary['net_profit'] = summary['total_revenue'] - summary['total_spend']
    summary['ROI_pct'] = np.where(
        summary['total_spend'] > 0,
        summary['net_profit'] / summary['total_spend'] * 100,
        0,
    )
    summary['ROAS'] = np.where(
        summary['total_spend'] > 0,
        summary['total_revenue'] / summary['total_spend'],
        0,
    )
    summary['LTV'] = np.where(
        summary['buyers'] > 0,
        summary['total_revenue'] / summary['buyers'],
        0,
    )
    summary['LTV_CAC_ratio'] = np.where(
        summary['CAC_per_buyer'] > 0,
        summary['LTV'] / summary['CAC_per_buyer'],
        0,
    )
    if txns is not None and not txns.empty:
        observed_days = max((txns['transaction_date'].max() - txns['transaction_date'].min()).days, 1)
        observed_months = max(observed_days / 30.4375, 1)
    else:
        observed_months = 6
    summary['monthly_rev_per_buyer'] = summary['LTV'] / observed_months
    summary['payback_months'] = np.where(
        (summary['CAC_per_buyer'] > 0) & (summary['monthly_rev_per_buyer'] > 0),
        summary['CAC_per_buyer'] / summary['monthly_rev_per_buyer'],
        0,
    )
    summary = summary.reset_index().rename(columns={'acquisition_channel': 'channel'})
    return summary


def build_churn_model_summary(user_segmented):
    os.makedirs(MODELS, exist_ok=True)
    model_data = user_segmented.copy()
    for old, new in [('recency', 'rfm_recency'), ('frequency', 'rfm_frequency'), ('monetary', 'rfm_monetary')]:
        if old in model_data.columns and new not in model_data.columns:
            model_data = model_data.rename(columns={old: new})

    encoders = {}
    for col in ['device', 'gender', 'acquisition_channel']:
        encoder = LabelEncoder()
        model_data[col + '_enc'] = encoder.fit_transform(model_data[col].astype(str))
        encoders[col] = encoder

    features = [
        'age',
        'total_sessions',
        'avg_session_duration',
        'total_pages_viewed',
        'days_since_signup',
        'has_purchased',
        'total_purchases',
        'total_revenue',
        'avg_order_value',
        'avg_revenue_per_purchase',
        'revenue_per_session',
        'device_enc',
        'gender_enc',
        'acquisition_channel_enc',
    ]

    if 'churn_eligible' in model_data.columns:
        train_data = model_data[model_data['churn_eligible'].eq(1)].copy()
    else:
        train_data = model_data.copy()

    x = train_data[features]
    y = train_data['churn'].astype(int)
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 5:
        raise ValueError(
            'Churn model needs at least two churn classes with 5+ eligible users each '
            f'for stratified validation. Current class counts: {class_counts.to_dict()}'
        )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    grid_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        GBM_PARAM_GRID,
        cv=grid_cv,
        scoring='roc_auc',
        n_jobs=1,
        verbose=0,
    )
    grid.fit(x_train, y_train)
    best_params = grid.best_params_

    tuned_model = grid.best_estimator_
    proba = tuned_model.predict_proba(x_test)[:, 1]
    pred = tuned_model.predict(x_test)

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_model = GradientBoostingClassifier(random_state=42, **best_params)
    oof_proba = cross_val_predict(
        oof_model,
        x,
        y,
        cv=oof_cv,
        method='predict_proba',
        n_jobs=1,
    )[:, 1]

    risk_scores = train_data[['user_id']].copy()
    risk_scores['churn_proba'] = oof_proba
    risk_tiers = pd.cut(
        risk_scores['churn_proba'],
        bins=[0, 0.3, 0.5, 0.7, 1.001],
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
        right=True,
        include_lowest=True,
    )
    risk_scores['risk_tier'] = pd.Series(risk_tiers, index=risk_scores.index).astype(object)
    risk_scores['risk_tier'] = risk_scores['risk_tier'].where(
        risk_scores['risk_tier'].notna(),
        'Not Observable',
    )

    all_scores = model_data[['user_id']].copy()
    all_scores = all_scores.merge(risk_scores, on='user_id', how='left')
    all_scores['risk_tier'] = all_scores['risk_tier'].fillna('Not Observable')

    final_model = GradientBoostingClassifier(random_state=42, **best_params)
    final_model.fit(x, y)
    churn_metadata = {
        'model_name': 'Tuned Gradient Boosting Classifier',
        'features': features,
        'best_params': best_params,
        'training_rows': int(len(train_data)),
        'scoring_method': 'out_of_fold_probabilities_for_existing_users',
    }
    save_model_artifacts(
        churn_model={
            'model': final_model,
            'features': features,
            'encoders': encoders,
            'best_params': best_params,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'training_rows': int(len(train_data)),
        },
        churn_model_metadata=churn_metadata,
    )

    summary = {
        'model_name': 'Tuned Gradient Boosting Classifier',
        'model_auc': round(float(roc_auc_score(y_test, proba)), 6),
        'model_ap': round(float(average_precision_score(y_test, proba)), 6),
        'model_accuracy': round(float((pred == y_test).mean()), 6),
        'model_features': features,
        'model_best_params': best_params,
        'model_training_rows': int(len(train_data)),
    }
    return summary, all_scores


def build_project_metrics(user_data, user_segmented, marketing_summary, model_summary):
    buyers = user_data[user_data['has_purchased'] == 1].copy()
    total_revenue = float(user_data['total_revenue'].sum())

    top_10_users_share = (
        user_data.nlargest(max(int(np.ceil(len(user_data) * 0.10)), 1), 'total_revenue')['total_revenue'].sum()
        / total_revenue
        if total_revenue > 0
        else 0
    )
    top_10_buyers_share = (
        buyers.nlargest(max(int(np.ceil(len(buyers) * 0.10)), 1), 'total_revenue')['total_revenue'].sum()
        / total_revenue
        if len(buyers) > 0 and total_revenue > 0
        else 0
    )

    paid = marketing_summary[marketing_summary['total_spend'] > 0].copy()
    best_paid_cac = paid.loc[paid['CAC_per_buyer'].idxmin()] if not paid.empty else None
    best_roas = paid.loc[paid['ROAS'].idxmax()] if not paid.empty else None
    best_ltv_cac = paid.loc[paid['LTV_CAC_ratio'].idxmax()] if not paid.empty else None

    metrics = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'total_users': int(len(user_data)),
        'total_revenue': round(total_revenue, 2),
        'churn_rate': round(float(user_data['churn'].mean()), 6),
        'conversion_rate': round(float(user_data['has_purchased'].mean()), 6),
        'avg_ltv_per_buyer': round(float(buyers['total_revenue'].mean()), 2) if len(buyers) else 0,
        'avg_aov': round(float(buyers['avg_order_value'].mean()), 2) if len(buyers) else 0,
        'top_10_users_revenue_share': round(float(top_10_users_share), 6),
        'top_10_buyers_revenue_share': round(float(top_10_buyers_share), 6),
        'segment_count': int(user_segmented['RFM_Segment'].nunique()),
        'cluster_label_count': int(user_segmented['cluster_label'].nunique()),
        'churn_eligible_users': int(user_data['churn_eligible'].sum()) if 'churn_eligible' in user_data.columns else int(len(user_data)),
        'best_paid_cac_channel': None if best_paid_cac is None else str(best_paid_cac['channel']),
        'best_paid_cac_per_buyer': None if best_paid_cac is None else round(float(best_paid_cac['CAC_per_buyer']), 2),
        'best_roas_channel': None if best_roas is None else str(best_roas['channel']),
        'best_roas': None if best_roas is None else round(float(best_roas['ROAS']), 2),
        'best_ltv_cac_channel': None if best_ltv_cac is None else str(best_ltv_cac['channel']),
        'best_ltv_cac_ratio': None if best_ltv_cac is None else round(float(best_ltv_cac['LTV_CAC_ratio']), 2),
    }
    metrics.update(model_summary)
    return metrics


def write_outputs(user_data, user_segmented, marketing_summary, metrics):
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    user_data.to_csv(os.path.join(PROC, 'user_data.csv'), index=False)
    user_segmented.to_csv(os.path.join(PROC, 'user_data_segmented.csv'), index=False)
    marketing_summary.to_csv(os.path.join(PROC, 'marketing_summary.csv'), index=False)
    with open(os.path.join(PROC, 'project_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def run_pipeline():
    log('Starting pipeline...')
    users, sessions, txns, events, campaigns = load_raw_data()
    sessions, txns, events, campaigns = validate_raw_data(users, sessions, txns, events, campaigns)

    ud = build_user_data(users, sessions, txns)
    user_segmented = build_user_segments(ud, txns)
    marketing_summary = build_marketing_summary(ud, campaigns, txns)
    model_summary, risk_scores = build_churn_model_summary(user_segmented)
    user_segmented = user_segmented.merge(risk_scores, on='user_id', how='left')
    metrics = build_project_metrics(ud, user_segmented, marketing_summary, model_summary)

    log(f'  User data shape: {ud.shape}  Churn rate: {ud["churn"].mean():.3f}')
    log(f'  Segmented shape: {user_segmented.shape}')
    log(f'  Revenue: Rs {ud["total_revenue"].sum():,.0f}  Conversion: {ud["has_purchased"].mean():.3f}')
    log(f'  Churn model AUC: {model_summary["model_auc"]:.3f}')

    write_outputs(ud, user_segmented, marketing_summary, metrics)
    log('Pipeline complete.')
    return ud


if __name__ == '__main__':
    run_pipeline()
