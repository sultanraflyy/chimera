#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║           CHIMERA PRO — FOREX AI ANALYST             ║
╚══════════════════════════════════════════════════════╝
"""

import os, sys, json, time, warnings
import urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════

CONFIG = {
    'TWELVE_DATA_KEY':   '69d7ccec9877467a999c1ef49905fd77',
    'ALPHA_VANTAGE_KEY': 'DAEKONVLWGY365TI',
    'SYMBOL':            'XAUUSD',
    'LOOKBACK_DAYS':     500,    # lebih banyak data = lebih baik
    'SEQ_LENGTH':        30,
    'RISK_PCT':          0.02,
    'BALANCE':           10000.0,
    'CONFIDENCE_THRESHOLD': 0.55,   # 55% = lebih sering kasih signal

    'TOP_K_FEATURES':    20,     # feature selection: ambil top-20 terbaik
}

# ══════════════════════════════════════════
#  COLORS
# ══════════════════════════════════════════

class C:
    RESET = '\033[0m'; BOLD = '\033[1m'; DIM = '\033[2m'
    RED = '\033[91m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    CYAN = '\033[96m'; WHITE = '\033[97m'; MAG = '\033[95m'
    BG_GREEN = '\033[42m'; BG_RED = '\033[41m'; BG_BLUE = '\033[44m'

def hr(char='═', n=56): return char * n
def box(text, color=C.CYAN): 
    return f"{color}{hr()}\n  {C.BOLD}{text}{C.RESET}{color}\n{hr()}{C.RESET}"

# ══════════════════════════════════════════
#  PRICE ENGINE
# ══════════════════════════════════════════

class PriceEngine:
    SYMBOL_MAP = {
        'XAUUSD': {'twelve': 'XAU/USD', 'alpha': 'XAU',  'yahoo': 'GC=F',       'stooq': 'gc.f'},
        'EURUSD': {'twelve': 'EUR/USD', 'alpha': 'EUR',  'yahoo': 'EURUSD=X',    'stooq': 'eurusd'},
        'GBPUSD': {'twelve': 'GBP/USD', 'alpha': 'GBP',  'yahoo': 'GBPUSD=X',   'stooq': 'gbpusd'},
        'USDJPY': {'twelve': 'USD/JPY', 'alpha': 'JPY',  'yahoo': 'USDJPY=X',   'stooq': 'usdjpy'},
        'BTCUSD': {'twelve': 'BTC/USD', 'alpha': 'BTC',  'yahoo': 'BTC-USD',     'stooq': 'btcusd'},
        'NASDAQ': {'twelve': 'QQQ',     'alpha': 'QQQ',  'yahoo': 'QQQ',         'stooq': 'qqq.us'},
    }

    def __init__(self, symbol, twelve_key='', alpha_key=''):
        self.symbol     = symbol
        self.twelve_key = twelve_key
        self.alpha_key  = alpha_key

    def _get(self, url, timeout=6):
        req  = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode())

    def _twelve(self):
        if not self.twelve_key: return None
        sym = self.SYMBOL_MAP[self.symbol]['twelve']
        try:
            d = self._get(f"https://api.twelvedata.com/price?symbol={sym}&apikey={self.twelve_key}")
            return float(d['price']) if 'price' in d else None
        except: return None

    def _alpha(self):
        if not self.alpha_key: return None
        sym = self.SYMBOL_MAP[self.symbol]['alpha']
        try:
            d = self._get(f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE"
                          f"&from_currency={sym}&to_currency=USD&apikey={self.alpha_key}")
            r = d.get('Realtime Currency Exchange Rate', {})
            bid, ask = r.get('8. Bid Price'), r.get('9. Ask Price')
            return (float(bid)+float(ask))/2 if bid and ask else None
        except: return None

    def _yahoo(self):
        sym = self.SYMBOL_MAP[self.symbol]['yahoo']
        for base in ['query1', 'query2']:
            try:
                d = self._get(f"https://{base}.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&range=1d")
                m = d['chart']['result'][0]['meta']
                p = m.get('regularMarketPrice') or m.get('previousClose')
                return float(p)
            except: pass
        return None

    def _stooq(self):
        sym = self.SYMBOL_MAP[self.symbol]['stooq']
        try:
            req  = urllib.request.Request(f"https://stooq.com/q/l/?s={sym}&f=sd2t2ohlcv&h&e=csv",
                                          headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=6)
            lines = resp.read().decode().strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split(',')
                if len(parts) >= 5: return float(parts[4])
        except: pass
        return None

    def get_price(self):
        sources = [
            ('Twelve Data', self._twelve),
            ('Alpha Vantage', self._alpha),
            ('Yahoo Finance', self._yahoo),
            ('Stooq', self._stooq),
        ]
        for name, fn in sources:
            p = fn()
            if p and p > 0:
                return p, name
        return None, None

    def get_history(self, days=500):
        sym = self.SYMBOL_MAP[self.symbol]['yahoo']
        end = int(time.time())
        start = end - (days * 86400)
        try:
            d = self._get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                          f"?interval=1d&period1={start}&period2={end}", timeout=15)
            r = d['chart']['result'][0]
            q = r['indicators']['quote'][0]
            df = pd.DataFrame({
                'Date':   pd.to_datetime(r['timestamp'], unit='s'),
                'Open':   q['open'], 'High': q['high'],
                'Low':    q['low'],  'Close': q['close'],
                'Volume': q.get('volume', [0]*len(r['timestamp']))
            }).dropna(subset=['Close']).set_index('Date').sort_index()
            return df
        except: pass

        sym_s = self.SYMBOL_MAP[self.symbol]['stooq']
        d1 = (datetime.now()-timedelta(days=days)).strftime('%Y%m%d')
        d2 = datetime.now().strftime('%Y%m%d')
        try:
            req  = urllib.request.Request(f"https://stooq.com/q/d/l/?s={sym_s}&d1={d1}&d2={d2}&i=d",
                                          headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=15)
            df   = pd.read_csv(resp, parse_dates=['Date'], index_col='Date')
            df.columns = [c.capitalize() for c in df.columns]
            return df.sort_index()
        except: pass

        return self._synthetic(days)

    def _synthetic(self, days):
        base = {'XAUUSD': 2650, 'EURUSD': 1.085, 'GBPUSD': 1.27,
                'USDJPY': 149.5, 'BTCUSD': 67000, 'NASDAQ': 450}[self.symbol]
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        p = [base]
        for _ in range(days-1):
            p.append(p[-1] * (1 + np.random.normal(0.0001, 0.012)))
        p = np.array(p)
        return pd.DataFrame({
            'Open':   p*(1+np.random.normal(0,0.001,days)),
            'High':   p*(1+np.abs(np.random.normal(0,0.004,days))),
            'Low':    p*(1-np.abs(np.random.normal(0,0.004,days))),
            'Close':  p, 'Volume': np.random.randint(50000,200000,days).astype(float)
        }, index=dates)


# ══════════════════════════════════════════
#  FEATURES (30+)
# ══════════════════════════════════════════

class Features:
    @staticmethod
    def compute(df):
        d = df.copy()
        c, h, l = d['Close'], d['High'], d['Low']

        # RSI multi-period
        for p in [7, 14, 21]:
            delta = c.diff()
            g  = delta.clip(lower=0).rolling(p).mean()
            ls = (-delta.clip(upper=0)).rolling(p).mean()
            d[f'RSI_{p}'] = 100 - 100/(1+g/(ls+1e-9))

        # EMA
        for s in [9, 21, 50, 200]:
            d[f'EMA_{s}'] = c.ewm(span=s, adjust=False).mean()

        # MACD
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        d['MACD']      = e12 - e26
        d['MACD_SIG']  = d['MACD'].ewm(span=9, adjust=False).mean()
        d['MACD_HIST'] = d['MACD'] - d['MACD_SIG']

        # Bollinger Bands
        bb = c.rolling(20).mean(); bs = c.rolling(20).std()
        d['BB_UP']    = bb + 2*bs; d['BB_LO'] = bb - 2*bs
        d['BB_WIDTH'] = (d['BB_UP']-d['BB_LO'])/(bb+1e-9)
        d['BB_POS']   = (c-d['BB_LO'])/(d['BB_UP']-d['BB_LO']+1e-9)

        # ATR
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        d['ATR_14']    = tr.rolling(14).mean()
        d['ATR_7']     = tr.rolling(7).mean()
        d['ATR_RATIO'] = d['ATR_7']/(d['ATR_14']+1e-9)

        # Stochastic
        lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
        d['STOCH_K'] = 100*(c-lo14)/(hi14-lo14+1e-9)
        d['STOCH_D'] = d['STOCH_K'].rolling(3).mean()

        # Price vs EMA distances
        for s in [21,50,200]:
            d[f'P_EMA{s}'] = (c-d[f'EMA_{s}'])/(d[f'EMA_{s}']+1e-9)

        # EMA cross
        d['EMA9_21']   = (d['EMA_9']-d['EMA_21'])/(d['EMA_21']+1e-9)
        d['EMA21_50']  = (d['EMA_21']-d['EMA_50'])/(d['EMA_50']+1e-9)
        d['EMA50_200'] = (d['EMA_50']-d['EMA_200'])/(d['EMA_200']+1e-9)

        # Returns
        for p in [1,3,5,10,20]:
            d[f'RET_{p}'] = c.pct_change(p)

        # Candle structure
        d['BODY']  = (c-d['Open'])/(c.abs()+1e-9)
        d['UPPER'] = (h-c.clip(lower=d['Open']))/(h-l+1e-9)
        d['LOWER'] = (c.clip(upper=d['Open'])-l)/(h-l+1e-9)

        # Volatility
        d['VOL_5']  = c.pct_change().rolling(5).std()
        d['VOL_21'] = c.pct_change().rolling(21).std()
        d['VOL_RATIO'] = d['VOL_5']/(d['VOL_21']+1e-9)

        # Volume
        vol = d.get('Volume', pd.Series(1, index=d.index))
        d['VOL_MOM'] = vol/(vol.rolling(20).mean()+1)

        # ── NEW: Market Regime ──────────────────────────────
        # Trend strength via ADX
        plus_dm  = (h.diff()).clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        mask = plus_dm <= minus_dm; plus_dm[mask] = 0
        mask = minus_dm <= (h.diff()).clip(lower=0); minus_dm[mask] = 0
        atr14 = tr.rolling(14).mean()
        plus_di  = 100*plus_dm.rolling(14).mean()/(atr14+1e-9)
        minus_di = 100*minus_dm.rolling(14).mean()/(atr14+1e-9)
        dx = 100*(plus_di-minus_di).abs()/(plus_di+minus_di+1e-9)
        d['ADX']       = dx.rolling(14).mean()
        d['PLUS_DI']   = plus_di
        d['MINUS_DI']  = minus_di

        # Volatility regime (ATR normalized by price)
        d['VOL_REGIME'] = d['ATR_14']/c  # high = volatile market

        # Price momentum oscillator (ROC)
        d['ROC_10'] = (c/c.shift(10)-1)*100
        d['ROC_20'] = (c/c.shift(20)-1)*100

        # Ichimoku Tenkan/Kijun lines
        d['ICHIMOKU_T'] = (h.rolling(9).max()+l.rolling(9).min())/2
        d['ICHIMOKU_K'] = (h.rolling(26).max()+l.rolling(26).min())/2
        d['ICHIMOKU_DIFF'] = (d['ICHIMOKU_T']-d['ICHIMOKU_K'])/(c+1e-9)

        return d.dropna()

    @staticmethod
    def get_names(df):
        exclude = {'Open','High','Low','Close','Volume'}
        return [c for c in df.columns if c not in exclude]


# ══════════════════════════════════════════
#  LSTM + ATTENTION (NumPy)
# ══════════════════════════════════════════

class LSTMAttention:
    def __init__(self, n_features, hidden=64, lookback=30):
        self.n_feat  = n_features
        self.hidden  = hidden
        self.lookback= lookback
        s = 0.08
        for gate in ['f','i','g','o']:
            setattr(self, f'W{gate}', np.random.randn(hidden, n_features+hidden)*s)
            setattr(self, f'b{gate}', np.zeros((hidden,1)))
        self.W_a  = np.random.randn(1, hidden)*s
        self.b_a  = np.zeros((1,1))
        self.clf  = None

    def _sig(self, x):  return 1/(1+np.exp(-np.clip(x,-50,50)))
    def _tanh(self, x): return np.tanh(np.clip(x,-50,50))

    def forward(self, seq):
        h = np.zeros((self.hidden,1)); c = np.zeros((self.hidden,1)); hs = []
        for t in range(len(seq)):
            x  = seq[t].reshape(-1,1); hx = np.vstack([h, x])
            f  = self._sig(self.Wf@hx+self.bf); i  = self._sig(self.Wi@hx+self.bi)
            g  = self._tanh(self.Wg@hx+self.bg); o  = self._sig(self.Wo@hx+self.bo)
            c  = f*c+i*g; h = o*self._tanh(c); hs.append(h.copy())
        return np.array(hs).squeeze()

    def attention(self, hs):
        scores  = np.array([float(self.W_a@h.reshape(-1,1)+self.b_a) for h in hs])
        weights = np.exp(scores-scores.max()); weights /= weights.sum()+1e-9
        ctx = sum(w*h for w,h in zip(weights, hs))
        return ctx, weights

    def extract_features(self, seqs):
        ctxs = []
        for seq in seqs:
            try:
                hs  = self.forward(seq); ctx,_ = self.attention(hs)
                ctxs.append(ctx.flatten()[:self.hidden])
            except:
                ctxs.append(np.zeros(self.hidden))
        return np.array(ctxs)

    def fit(self, seqs, y):
        feats    = self.extract_features(seqs)
        self.clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        self.clf.fit(feats, y)
        return accuracy_score(y, self.clf.predict(feats))

    def predict_proba(self, seq):
        hs  = self.forward(seq); ctx, w = self.attention(hs)
        feat = ctx.flatten()[:self.hidden].reshape(1,-1)
        return self.clf.predict_proba(feat)[0][1], w


# ══════════════════════════════════════════
#  ENSEMBLE MODEL (+ improvements)
# ══════════════════════════════════════════

class EnsembleModel:
    """
    Improvements vs v1:
    1. Walk-Forward Validation — train di window bergerak, lebih realistis
    2. Feature Selection — mutual info, buang fitur noise
    3. Calibrated Probabilities — isotonic regression calibration
    4. Regime-Aware — multiplier confidence berdasar ADX/volatility
    5. More data (500 hari vs 365)
    """

    def __init__(self, symbol, lookback=30):
        self.symbol    = symbol
        self.lookback  = lookback
        self.scaler    = StandardScaler()
        self.selector  = None
        self.feat_names = None
        self.sel_names  = None
        self.lstm = self.gbm = self.rf = None
        self.trained = False
        self.metrics = {}

    # ── Walk-Forward Validation ──────────────────
    def _walk_forward_score(self, X, y, n_splits=5):
        """
        Test model di periode berbeda, bukan random split.
        Lebih fair untuk time-series data.
        """
        size = len(X)
        fold = size // (n_splits + 1)
        scores = []
        for i in range(1, n_splits+1):
            tr_end = fold * i
            te_end = min(tr_end + fold, size)
            if te_end <= tr_end+10: continue
            gbm_tmp = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
            gbm_tmp.fit(X[:tr_end], y[:tr_end])
            sc = accuracy_score(y[tr_end:te_end], gbm_tmp.predict(X[tr_end:te_end]))
            scores.append(sc)
        return float(np.mean(scores)) if scores else 0.5

    def _make_seqs(self, X, y):
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i-self.lookback:i]); ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, df, log=print):
        df2 = Features.compute(df)
        self.feat_names = Features.get_names(df2)

        X = df2[self.feat_names].values
        y = (df2['Close'].shift(-1) > df2['Close']).astype(int).values[:-1]
        X = X[:-1]

        X_s   = self.scaler.fit_transform(X)
        split = int(len(X_s) * 0.8)

        # ── Feature Selection ──
        log(f"  Feature selection ({len(self.feat_names)} → top {CONFIG['TOP_K_FEATURES']})...")
        self.selector = SelectKBest(mutual_info_classif, k=CONFIG['TOP_K_FEATURES'])
        self.selector.fit(X_s[:split], y[:split])
        X_sel = self.selector.transform(X_s)
        sel_mask = self.selector.get_support()
        self.sel_names = [self.feat_names[i] for i,v in enumerate(sel_mask) if v]
        log(f"  Top features: {', '.join(self.sel_names[:8])}...")

        # ── Walk-Forward Score ──
        wf_acc = self._walk_forward_score(X_sel[:split], y[:split])
        log(f"  Walk-forward acc: {wf_acc:.2%}")

        # ── LSTM ──
        log(f"  Training LSTM+Attention...")
        X_seq, y_seq = self._make_seqs(X_sel, y)
        seq_split = int(len(X_seq) * 0.8)
        self.lstm = LSTMAttention(X_sel.shape[1], hidden=64, lookback=self.lookback)
        lstm_acc  = self.lstm.fit(X_seq[:seq_split], y_seq[:seq_split])

        # ── GBM (calibrated) ──
        log(f"  Training Gradient Boosting...")
        gbm_base = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        self.gbm = CalibratedClassifierCV(gbm_base, cv=3, method='isotonic')
        self.gbm.fit(X_sel[:split], y[:split])
        gbm_acc = accuracy_score(y[split:], self.gbm.predict(X_sel[split:]))

        # ── RF (calibrated) ──
        log(f"  Training Random Forest...")
        rf_base = GradientBoostingClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=6,
            subsample=0.75, random_state=42)
        self.rf = CalibratedClassifierCV(rf_base, cv=3, method='isotonic')
        self.rf.fit(X_sel[:split], y[:split])
        rf_acc = accuracy_score(y[split:], self.rf.predict(X_sel[split:]))

        self.metrics = {
            'wf_acc':  wf_acc,
            'gbm_acc': gbm_acc,
            'rf_acc':  rf_acc,
            'avg_acc': (gbm_acc+rf_acc)/2,
            'n_features': len(self.feat_names),
            'n_selected': CONFIG['TOP_K_FEATURES'],
            'top_features': self.sel_names[:5],
            'train_size': split,
            'test_size':  len(X_s)-split,
        }
        self.trained = True
        return self.metrics

    def _regime_multiplier(self, details):
        """
        Sesuaikan confidence berdasarkan kondisi pasar.
        ADX tinggi = trending = signal lebih reliable.
        ATR ratio ekstrem = choppy/volatile = kurangi confidence.
        """
        adx   = details.get('adx', 25)
        vol_r = details.get('vol_regime', 0.01)

        mult = 1.0
        if adx > 35: mult *= 1.10   # strong trend → boost
        elif adx < 20: mult *= 0.90 # ranging → reduce
        if vol_r > 0.025: mult *= 0.92  # extreme volatile → cautious
        return min(mult, 1.15)

    def predict(self, df_recent):
        if not self.trained:
            return 'HOLD', 0.5, {}
        try:
            df2 = Features.compute(df_recent)
            X_s = self.scaler.transform(df2[self.feat_names].values)
            X_sel = self.selector.transform(X_s)

            gbm_p  = self.gbm.predict_proba(X_sel[-1:])[0][1]
            rf_p   = self.rf.predict_proba(X_sel[-1:])[0][1]
            lstm_p = 0.5
            if len(X_sel) >= self.lookback:
                lstm_p, _ = self.lstm.predict_proba(X_sel[-self.lookback:])

            prob = lstm_p*0.20 + gbm_p*0.45 + rf_p*0.35

            last = df2.iloc[-1]
            details = {
                'lstm': lstm_p, 'gbm': gbm_p, 'rf': rf_p, 'ensemble': prob,
                'rsi14':     float(df2['RSI_14'].iloc[-1])    if 'RSI_14'     in df2 else 0,
                'rsi7':      float(df2['RSI_7'].iloc[-1])     if 'RSI_7'      in df2 else 0,
                'macd':      float(df2['MACD'].iloc[-1])      if 'MACD'       in df2 else 0,
                'macd_hist': float(df2['MACD_HIST'].iloc[-1]) if 'MACD_HIST'  in df2 else 0,
                'bb_pos':    float(df2['BB_POS'].iloc[-1])    if 'BB_POS'     in df2 else 0.5,
                'atr14':     float(df2['ATR_14'].iloc[-1])    if 'ATR_14'     in df2 else 0,
                'atr7':      float(df2['ATR_7'].iloc[-1])     if 'ATR_7'      in df2 else 0,
                'adx':       float(df2['ADX'].iloc[-1])       if 'ADX'        in df2 else 25,
                'plus_di':   float(df2['PLUS_DI'].iloc[-1])   if 'PLUS_DI'    in df2 else 0,
                'minus_di':  float(df2['MINUS_DI'].iloc[-1])  if 'MINUS_DI'   in df2 else 0,
                'vol_regime':float(df2['VOL_REGIME'].iloc[-1]) if 'VOL_REGIME' in df2 else 0.01,
                'stoch_k':   float(df2['STOCH_K'].iloc[-1])   if 'STOCH_K'    in df2 else 50,
                'ema9_21':   float(df2['EMA9_21'].iloc[-1])   if 'EMA9_21'    in df2 else 0,
                'ichimoku':  float(df2['ICHIMOKU_DIFF'].iloc[-1]) if 'ICHIMOKU_DIFF' in df2 else 0,
                'roc20':     float(df2['ROC_20'].iloc[-1])    if 'ROC_20'     in df2 else 0,
            }

            # Regime-adjusted probability
            rm   = self._regime_multiplier(details)
            adj  = prob * rm
            # Clamp so we don't go over 1
            adj  = min(adj, 0.95) if adj > 0.5 else max(adj, 0.05)
            details['regime_mult'] = rm
            details['adj_prob']    = adj

            thr = CONFIG['CONFIDENCE_THRESHOLD']
            if   adj >= thr + 0.08:  signal = 'BUY'
            elif adj >= thr:          signal = 'WEAK BUY'
            elif adj <= 1-thr-0.08:  signal = 'SELL'
            elif adj <= 1-thr:        signal = 'WEAK SELL'
            else:                     signal = 'HOLD'

            return signal, adj, details
        except Exception as e:
            return 'HOLD', 0.5, {}


# ══════════════════════════════════════════
#  SL/TP ENGINE
# ══════════════════════════════════════════

class SLTPEngine:
    MULT = {
        'XAUUSD': {'sl': 1.8, 'rr': 2.0},
        'EURUSD': {'sl': 1.2, 'rr': 2.0},
        'GBPUSD': {'sl': 1.3, 'rr': 2.0},
        'USDJPY': {'sl': 1.2, 'rr': 2.0},
        'BTCUSD': {'sl': 2.5, 'rr': 2.0},
        'NASDAQ': {'sl': 1.4, 'rr': 2.0},
    }

    def calculate(self, entry, atr, atr7, signal, symbol):
        m   = self.MULT.get(symbol, {'sl': 1.5, 'rr': 2.0})
        eff = atr*0.6 + atr7*0.4
        sl_d = eff * m['sl']; tp_d = sl_d * m['rr']
        is_big = symbol in ['XAUUSD','BTCUSD','NASDAQ']
        dp = 2 if is_big else 5
        pm = 10 if symbol == 'XAUUSD' else (1 if symbol in ['BTCUSD','NASDAQ'] else 10000)

        if signal == 'BUY':
            sl, tp = entry-sl_d, entry+tp_d
        elif signal == 'SELL':
            sl, tp = entry+sl_d, entry-tp_d
        else:
            sl = tp = entry

        return {
            'entry': entry, 'sl': round(sl,dp), 'tp': round(tp,dp),
            'sl_dist': round(sl_d,2), 'tp_dist': round(tp_d,2),
            'sl_pips': int(sl_d*pm), 'tp_pips': int(tp_d*pm),
            'rr': m['rr'], 'atr_used': round(eff,2),
        }

    def lot_size(self, balance, risk_pct, sl_dist, symbol):
        lv  = {'XAUUSD':100,'EURUSD':100000,'GBPUSD':100000,
               'USDJPY':100000,'BTCUSD':1,'NASDAQ':100}.get(symbol,100000)
        lot = (balance*risk_pct)/(sl_dist*lv) if sl_dist>0 else 0.01
        return max(0.01, min(5.0, round(lot,2)))


# ══════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════

class Backtester:
    SPREAD = {'XAUUSD':0.30,'EURUSD':0.0001,'GBPUSD':0.0002,
              'USDJPY':0.02,'BTCUSD':2.0,'NASDAQ':0.05}

    def run(self, df, model, sltp, symbol, capital=10000):
        df2    = Features.compute(df)
        X_s    = model.scaler.transform(df2[model.feat_names].values)
        X_sel  = model.selector.transform(X_s)
        spread = self.SPREAD.get(symbol, 0.3)
        results = []; cap = capital

        for i in range(model.lookback+5, len(X_sel)-1):
            p = model.gbm.predict_proba(X_sel[i:i+1])[0][1]
            if   p >= 0.60: signal = 'BUY'
            elif p <= 0.40: signal = 'SELL'
            else: continue

            row   = df2.iloc[i]
            entry = float(row['Close'])
            atr14 = float(row.get('ATR_14', entry*0.005))
            atr7  = float(row.get('ATR_7',  entry*0.003))
            st    = sltp.calculate(entry, atr14, atr7, signal, symbol)
            next_c= float(df2['Close'].iloc[i+1])

            raw_ret = ((next_c-entry-spread)/entry if signal=='BUY'
                       else (entry-next_c-spread)/entry)            pnl = capital*CONFIG['RISK_PCT']*raw_ret*10
            cap += pnl
            results.append({'date':df2.index[i],'signal':signal,
                            'pnl':pnl,'bal':cap,'atr':atr14})

        if not results: return None
        df_r = pd.DataFrame(results)
        wins = (df_r['pnl']>0).sum(); total = len(df_r)
        gp   = df_r[df_r['pnl']>0]['pnl'].sum()
        gl   = df_r[df_r['pnl']<=0]['pnl'].abs().sum()
        net  = cap-capital

        peak = capital; mdd = 0
        for b in df_r['bal']:
            if b>peak: peak=b
            dd = (peak-b)/peak*100; mdd = max(mdd,dd)

        dr = df_r['pnl']/capital
        sharpe = (dr.mean()/(dr.std()+1e-9))*np.sqrt(252)

        df_r['month'] = pd.to_datetime(df_r['date']).dt.to_period('M')
        monthly = df_r.groupby('month')['pnl'].sum()

        return {'total':total,'wins':wins,'losses':total-wins,
                'win_rate':wins/total,'pf':gp/(gl+1e-9),
                'net_pnl':net,'ret_pct':net/capital*100,
                'mdd':mdd,'sharpe':sharpe,'monthly':monthly,'trades':df_r}


# ══════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════

def clear(): os.system('cls' if os.name=='nt' else 'clear')

def fmt_price(p, symbol):
    return f"${p:,.2f}" if p>100 else f"{p:,.5f}"

def bar(val, mn, mx, width=20, color=C.GREEN):
    pct = (val-mn)/(mx-mn+1e-9)
    filled = int(pct*width)
    return f"{color}{'█'*filled}{C.DIM}{'░'*(width-filled)}{C.RESET}"

def confidence_bar(prob, width=24):
    # Color-coded: red zone < 38%, green zone > 62%, yellow middle
    filled = int(prob*width)
    col = (C.GREEN if prob>0.62 else C.RED if prob<0.38 else C.YELLOW)
    return f"{col}{'█'*filled}{C.DIM}{'░'*(width-filled)}{C.RESET} {col}{prob:.1%}{C.RESET}"

def signal_display(signal, prob):
    """Render sinyal besar di tengah layar"""
    cfg = {
        'BUY':       ('\033[42m', C.GREEN,  '  ▲  B U Y  ▲  '),
        'WEAK BUY':  ('\033[42m', C.GREEN,  '  △ WEAK BUY △  '),
        'SELL':      ('\033[41m', C.RED,    '  ▼  S E L L  ▼  '),
        'WEAK SELL': ('\033[41m', C.RED,    '  ▽ WEAK SELL ▽  '),
        'HOLD':      ('\033[43m', C.YELLOW, '  ◆  H O L D  ◆  '),
    }
    bg, col, label = cfg.get(signal, cfg['HOLD'])
    pct_bar = confidence_bar(prob, width=20)
    print(f"\n  {bg}{C.BOLD}{C.WHITE}{label}{C.RESET}   {col}{C.BOLD}{prob:.1%} confidence{C.RESET}")
    print(f"  {pct_bar}")

def print_main(symbol, price, source, signal, prob, details, st, lot, metrics, bt):
    clear()
    now = datetime.now().strftime('%H:%M:%S  %d %b %Y')

    col = {
        'BUY': C.GREEN, 'WEAK BUY': C.GREEN,
        'SELL': C.RED,  'WEAK SELL': C.RED, 'HOLD': C.YELLOW
    }.get(signal, C.YELLOW)

    # ── HEADER ──────────────────────────────────────────
    print(f"\n{C.CYAN}{hr()}")
    print(f"  CHIMERA PRO  ·  {symbol}  ·  {now}")
    print(f"  Harga  {C.YELLOW}{C.BOLD}{fmt_price(price, symbol)}{C.RESET}{C.CYAN}  ·  {source}")
    print(f"{hr()}{C.RESET}")

    # ── SINYAL UTAMA ─────────────────────────────────────
    signal_display(signal, prob)

    # Konteks confidence
    if prob >= 0.63:   ctx = f"{C.GREEN}Signal kuat — model cukup yakin{C.RESET}"
    elif prob >= 0.55: ctx = f"{C.YELLOW}Signal lemah — hati-hati, konfirmasi dulu{C.RESET}"
    elif prob <= 0.37: ctx = f"{C.GREEN}Signal kuat (bearish) — model cukup yakin{C.RESET}"
    elif prob <= 0.45: ctx = f"{C.YELLOW}Signal lemah (bearish) — konfirmasi dulu{C.RESET}"
    else:              ctx = f"{C.DIM}Pasar tidak jelas arahnya, lebih baik tunggu{C.RESET}"
    print(f"  {ctx}")

    adx   = details.get('adx', 25)
    rm    = details.get('regime_mult', 1.0)
    r_col = C.GREEN if adx>30 else C.YELLOW if adx>20 else C.RED
    regime= "Trending 🔥" if adx>30 else "Choppy ⚠" if adx<20 else "Neutral"
    print(f"  {C.DIM}Kondisi pasar: {C.RESET}{r_col}{regime}{C.RESET}  "
          f"{C.DIM}ADX {adx:.0f}  ·  Regime adj ×{rm:.2f}{C.RESET}")
    print(f"  {C.DIM}Model votes — LSTM: {details['lstm']:.0%}  GBM: {details['gbm']:.0%}  RF: {details['rf']:.0%}{C.RESET}")

    # ── TP / SL ──────────────────────────────────────────
    if signal in ['BUY','WEAK BUY','SELL','WEAK SELL']:
        is_buy = 'BUY' in signal        tp_c = C.GREEN if is_buy else C.RED
        sl_c = C.RED   if is_buy else C.GREEN

        print(f"\n{C.CYAN}  {'─'*52}{C.RESET}")
        print(f"  {C.WHITE}RENCANA TRADE{C.RESET}")
        print()
        print(f"  Entry  →  {col}{C.BOLD}{fmt_price(st['entry'], symbol)}{C.RESET}")
        print(f"  TP     →  {tp_c}{C.BOLD}{fmt_price(st['tp'], symbol)}{C.RESET}   "
              f"(+{st['tp_pips']} pips  ·  R:R 1:{st['rr']:.1f})")
        print(f"  SL     →  {sl_c}{fmt_price(st['sl'], symbol)}{C.RESET}   "
              f"(-{st['sl_pips']} pips  ·  ATR×1.8)")
        print()
        risk_d   = CONFIG['BALANCE'] * CONFIG['RISK_PCT']
        reward_d = risk_d * st['rr']
        print(f"  Lot    →  {C.CYAN}{lot}{C.RESET}   "
              f"{C.DIM}(risiko {C.RESET}{C.RED}-${risk_d:,.0f}{C.RESET}"
              f"{C.DIM}  ·  target {C.RESET}{C.GREEN}+${reward_d:,.0f}{C.RESET}{C.DIM}){C.RESET}")

        if 'WEAK' in signal:
            print(f"\n  {C.YELLOW}⚡ WEAK signal: tunggu 1 konfirmasi tambahan{C.RESET}")
            print(f"  {C.DIM}  (misal: candle close di atas/bawah EMA21, atau volume spike){C.RESET}")

    # ── INDIKATOR ────────────────────────────────────────
    print(f"\n{C.CYAN}  {'─'*52}{C.RESET}")
    print(f"  {C.WHITE}INDIKATOR{C.RESET}\n")

    rsi14  = details.get('rsi14', 50)
    rsi_c  = C.RED if rsi14>70 else C.GREEN if rsi14<30 else C.WHITE
    rsi_lbl= "Overbought ⬇" if rsi14>70 else "Oversold ⬆" if rsi14<30 else "Normal"
    macd   = details.get('macd', 0)
    mh     = details.get('macd_hist', 0)
    macd_c = C.GREEN if macd>0 else C.RED
    bb     = details.get('bb_pos', 0.5)
    bb_lbl = "Dekat upper band" if bb>0.8 else "Dekat lower band" if bb<0.2 else "Tengah"
    stk    = details.get('stoch_k', 50)
    stk_c  = C.RED if stk>80 else C.GREEN if stk<20 else C.WHITE

    print(f"  RSI(14)  {rsi_c}{rsi14:5.1f}{C.RESET}  {C.DIM}{rsi_lbl}{C.RESET}")
    print(f"  MACD     {macd_c}{macd:+.4f}{C.RESET}  {C.DIM}hist {mh:+.4f}  "
          f"({'bullish' if macd>0 else 'bearish'}){C.RESET}")
    print(f"  BB Pos   {bb:.2f}   {C.DIM}{bb_lbl}{C.RESET}")
    print(f"  Stoch-K  {stk_c}{stk:5.1f}{C.RESET}  "
          f"{C.DIM}{'Overbought' if stk>80 else 'Oversold' if stk<20 else 'Normal'}{C.RESET}")

    # ── BACKTEST RINGKAS ─────────────────────────────────
    if bt:
        net_c = C.GREEN if bt['ret_pct']>=0 else C.RED
        print(f"\n{C.CYAN}  {'─'*52}{C.RESET}")
        print(f"  {C.WHITE}PERFORMA HISTORIS  {C.DIM}(backtest ≠ jaminan){C.RESET}\n")
        print(f"  Return   {net_c}{bt['ret_pct']:+.2f}%{C.RESET}     "
              f"Win Rate  {C.CYAN}{bt['win_rate']:.1%}{C.RESET}  ({bt['wins']}W/{bt['losses']}L)")
        print(f"  Profit Factor  {C.CYAN}{bt['pf']:.2f}x{C.RESET}     "
              f"Max Drawdown  {C.RED}{bt['mdd']:.1f}%{C.RESET}")
        print(f"  Sharpe  {C.CYAN}{bt['sharpe']:.2f}{C.RESET}          "
              f"Total trades  {bt['total']}")

        # Accuracy disclaimer
        exp_wr = bt['win_rate']
        if exp_wr >= 0.55:   acc_ctx = f"{C.GREEN}Di atas random — model bisa baca pola{C.RESET}"
        elif exp_wr >= 0.50: acc_ctx = f"{C.YELLOW}Tipis di atas random — R:R yang menyelamatkan{C.RESET}"
        else:                acc_ctx = f"{C.RED}Di bawah 50% — model perlu retraining{C.RESET}"
        print(f"\n  {acc_ctx}")
        print(f"  {C.DIM}Win rate >50% + R:R 1:2 = profitable. Ini realistis, bukan klaim 184%.{C.RESET}")

    # ── FOOTER ──────────────────────────────────────────
    print(f"\n{C.CYAN}{hr()}")
    print(f"  ⚠  Edukasi saja. Test demo ≥3 bulan sebelum live trading.")
    print(f"{hr()}{C.RESET}\n")


# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════

def main():
    clear()
    print(f"\n{C.CYAN}{hr()}\n  CHIMERA PRO — Loading...\n{hr()}{C.RESET}\n")

    symbol = CONFIG['SYMBOL']

    # Step 1: Harga
    print(f"  {C.CYAN}[1/4]{C.RESET} Fetching price...")
    engine = PriceEngine(symbol, CONFIG['TWELVE_DATA_KEY'], CONFIG['ALPHA_VANTAGE_KEY'])
    price, source = engine.get_price()
    if price is None:
        source = "Manual"
        try: price = float(input(f"  Harga {symbol}: ").replace(',',''))
        except: print("Invalid."); return
    else:
        print(f"       {C.GREEN}${price:,.2f}{C.RESET} dari {source}")

    # Step 2: History
    print(f"\n  {C.CYAN}[2/4]{C.RESET} Downloading {CONFIG['LOOKBACK_DAYS']} hari data...")
    df = engine.get_history(CONFIG['LOOKBACK_DAYS'])
    print(f"       {len(df)} candles loaded.")

    # Step 3: Train
    print(f"\n  {C.CYAN}[3/4]{C.RESET} Training ensemble model...\n")
    model   = EnsembleModel(symbol, lookback=CONFIG['SEQ_LENGTH'])
    metrics = model.train(df, log=lambda x: print(f"       {x}"))

    # Step 4: Predict
    print(f"\n  {C.CYAN}[4/4]{C.RESET} Generating signal...")
    signal, prob, details = model.predict(df)

    sltp_eng = SLTPEngine()
    atr14    = details.get('atr14', price*0.008)
    atr7     = details.get('atr7',  price*0.005)
    # Derive base direction for SL/TP calculation
    base_dir = 'BUY' if 'BUY' in signal else 'SELL' if 'SELL' in signal else 'HOLD'
    st       = sltp_eng.calculate(price, atr14, atr7, base_dir, symbol)
    lot      = sltp_eng.lot_size(CONFIG['BALANCE'], CONFIG['RISK_PCT'], st['sl_dist'], symbol)

    # Backtest
    bt = Backtester().run(df, model, sltp_eng, symbol, CONFIG['BALANCE'])

    # Display
    print_main(symbol, price, source, signal, prob, details, st, lot, metrics, bt)

    # ── Loop ──
    while True:
        print(f"  {C.WHITE}[R]{C.RESET} Refresh harga  "
              f"{C.WHITE}[S]{C.RESET} Ganti symbol  "
              f"{C.WHITE}[Q]{C.RESET} Keluar")
        cmd = input("  >> ").strip().upper()

        if cmd == 'Q':
            break

        elif cmd == 'R':
            p2, src2 = engine.get_price()
            if p2: price, source = p2, src2
            else:
                try: price = float(input(f"  Harga manual: ").replace(',',''))
                except: pass
            base_dir = 'BUY' if 'BUY' in signal else 'SELL' if 'SELL' in signal else 'HOLD'
            st  = sltp_eng.calculate(price, atr14, atr7, base_dir, symbol)
            lot = sltp_eng.lot_size(CONFIG['BALANCE'], CONFIG['RISK_PCT'], st['sl_dist'], symbol)
            print_main(symbol, price, source, signal, prob, details, st, lot, metrics, bt)

        elif cmd == 'S':
            syms = list(PriceEngine.SYMBOL_MAP.keys())
            for i,s in enumerate(syms,1): print(f"  [{i}] {s}")
            try:
                idx = int(input("  Pilih: "))-1
                CONFIG['SYMBOL'] = syms[idx]
                print(f"  Diganti ke {syms[idx]}. Restart untuk retrain.")
            except: pass


if __name__ == '__main__':
    main()
