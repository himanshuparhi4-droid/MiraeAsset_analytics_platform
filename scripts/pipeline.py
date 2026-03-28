import pandas as pd, numpy as np, os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_pipeline():
    log('Starting pipeline...')
    users    = pd.read_csv(os.path.join(RAW,'users.csv'), parse_dates=['signup_date'])
    sessions = pd.read_csv(os.path.join(RAW,'sessions.csv'), parse_dates=['session_date'])
    txns     = pd.read_csv(os.path.join(RAW,'transactions.csv'), parse_dates=['transaction_date'])
    events   = pd.read_csv(os.path.join(RAW,'events.csv'), parse_dates=['event_date'])

    for name, df in [('users',users),('sessions',sessions),('transactions',txns),('events',events)]:
        if df.isnull().sum().sum() > 0: log(f'  WARNING: {name} has nulls')

    sess_agg = sessions.groupby('user_id').agg(
        total_sessions       = ('session_id','count'),
        avg_session_duration = ('duration_minutes','mean'),
        total_pages_viewed   = ('pages_viewed','sum'),
        last_active_date     = ('session_date','max'),
    ).reset_index()

    txn_agg = txns.groupby('user_id').agg(
        total_revenue       = ('amount','sum'),
        total_purchases     = ('transaction_id','count'),
        avg_order_value     = ('amount','mean'),
        first_purchase_date = ('transaction_date','min'),
        last_purchase_date  = ('transaction_date','max'),
    ).reset_index()

    ud = users.copy()
    ud = ud.merge(sess_agg, on='user_id', how='left')
    ud = ud.merge(txn_agg,  on='user_id', how='left')
    for col in ['total_sessions','total_pages_viewed','total_revenue',
                'total_purchases','avg_order_value','avg_session_duration']:
        ud[col] = ud[col].fillna(0)

    latest = sessions['session_date'].max()
    ud['days_since_signup']       = (latest - ud['signup_date']).dt.days
    ud['days_since_last_active']  = (latest - ud['last_active_date']).dt.days.fillna(ud['days_since_signup'])
    ud['avg_revenue_per_purchase']= np.where(ud['total_purchases']>0, ud['total_revenue']/ud['total_purchases'],0)
    ud['revenue_per_session']     = np.where(ud['total_sessions']>0,  ud['total_revenue']/ud['total_sessions'],0)
    ud['has_purchased']           = (ud['total_purchases'] > 0).astype(int)

    ud['engagement_score_raw'] = (ud['total_sessions']*0.5 + ud['avg_session_duration']*0.3 + ud['total_purchases']*0.2)
    ud['engagement_score']     = MinMaxScaler().fit_transform(ud[['engagement_score_raw']])

    ud['churn'] = np.where(
        ud['last_active_date'].isna() | ((latest - ud['last_active_date']).dt.days > 30), 1, 0)

    assert len(ud) == len(users)
    assert 0.1 < ud['churn'].mean() < 0.9
    log(f'  Shape: {ud.shape}  Churn rate: {ud["churn"].mean():.3f}')

    os.makedirs(PROC, exist_ok=True)
    ud.to_csv(os.path.join(PROC,'user_data.csv'), index=False)
    log('Pipeline complete.')
    return ud

if __name__ == '__main__':
    run_pipeline()
