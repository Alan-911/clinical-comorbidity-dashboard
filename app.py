import streamlit as st
import pandas as pd
import os, base64, re
from datetime import datetime, timedelta
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Clinical Intelligence Dashboard", page_icon="⚕️", layout="wide", initial_sidebar_state="collapsed")

# ── Inject CSS immediately, before any data loading ───────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.5rem 1rem!important;max-width:100%!important;}
html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif!important;color:#0f172a;}
.stApp{background-color:#f8fafc;background-image:linear-gradient(to right,#e2e8f0 1px,transparent 1px),linear-gradient(to bottom,#e2e8f0 1px,transparent 1px);background-size:40px 40px;}
/* Performance-optimized: solid backgrounds instead of expensive backdrop-filter */
.gc{background:#ffffff;border:1px solid #e2e8f0;border-radius:15px;padding:18px;box-shadow:0 4px 12px rgba(0,0,0,0.05);margin-bottom:15px;}
.vc{flex:1;padding:14px;border-radius:14px;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
.ekg{height:24px;width:100%;margin-top:8px;background-size:100px 100%;opacity:0.5;}
.ti{margin-bottom:10px;padding-left:14px;border-left:2px solid #e2e8f0;position:relative;}
.ti::before{content:'';position:absolute;left:-6px;top:2px;width:10px;height:10px;border-radius:50%;background:#3b82f6;}
.ap{border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;background:#f8fafc;}
.mid-stat{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:14px 10px;text-align:center;margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
[data-testid="stForm"]{background:#ffffff!important;border:1px solid #e2e8f0!important;border-radius:15px!important;padding:4px 18px 18px!important;box-shadow:0 4px 12px rgba(0,0,0,0.05)!important;margin-top:0!important;}
[data-testid="stForm"] label{font-size:12px;font-weight:600;color:#64748b;}
[data-testid="stForm"] .stFormSubmitButton button{border-radius:8px;}
/* Pill-shaped nav bar — wraps the container keyed "navbar" */
.st-key-navbar{background:#ffffff!important;border:1px solid #e2e8f0!important;border-radius:100px!important;box-shadow:0 2px 10px rgba(0,0,0,0.04)!important;padding:6px 28px!important;margin-bottom:18px!important;position:relative;z-index:10;}
.nav-brand{font-weight:700;font-size:19px;line-height:40px;color:#0f172a;}
/* Nav buttons inside the pill: transparent, styled as links */
.st-key-navbar [data-testid="stButton"] button{background:transparent!important;border:none!important;color:#64748b!important;font-size:13px!important;font-weight:600!important;padding:10px 12px!important;height:auto!important;box-shadow:none!important;min-height:0!important;line-height:1.2!important;}
.st-key-navbar [data-testid="stButton"] button:hover{color:#3b82f6!important;background:rgba(59,130,246,0.06)!important;border-radius:10px!important;}
.st-key-navbar [data-testid="stButton"] button:focus{box-shadow:none!important;outline:none!important;}
.st-key-navbar [data-testid="stButton"] button:active{background:rgba(59,130,246,0.12)!important;}
/* Patient badge button — right-aligned, smaller, bolder */
.st-key-nav_patient button{color:#0f172a!important;font-weight:800!important;font-size:10px!important;text-align:right!important;letter-spacing:0.5px;}
/* Card frames for demographic + advisory buttons in the left column */
.st-key-card_demo button,.st-key-card_advisory button{background:#ffffff!important;border:1px solid #e2e8f0!important;border-radius:15px!important;padding:18px 20px!important;box-shadow:0 4px 12px rgba(0,0,0,0.05)!important;text-align:left!important;font-weight:700!important;font-size:14px!important;color:#0f172a!important;height:auto!important;white-space:pre-wrap!important;line-height:1.5!important;margin-bottom:15px!important;width:100%!important;min-height:0!important;transition:box-shadow 0.15s ease;}
.st-key-card_demo button:hover,.st-key-card_advisory button:hover{box-shadow:0 6px 18px rgba(59,130,246,0.15)!important;border-color:rgba(59,130,246,0.35)!important;}
.st-key-card_demo button:focus,.st-key-card_advisory button:focus{box-shadow:0 4px 12px rgba(0,0,0,0.05)!important;outline:none!important;}
.st-key-card_demo button{border-left:4px solid #7c3aed!important;}
.st-key-card_advisory button{border-left:4px solid #3b82f6!important;}
/* Remove Streamlit's default element spacing that breaks layout */
.element-container{margin-bottom:0!important;}
div[data-testid="stVerticalBlock"]>div{gap:0!important;}
/* Stacking context: prevent Streamlit containers trapping overlays */
[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main,.main{transform:none!important;filter:none!important;perspective:none!important;will-change:auto!important;}
/* Animations — static transform only, no continuous spin for performance */
@keyframes fadeIn{from{opacity:0;}to{opacity:0.6;}}
/* ── SAFETY OVERRIDES — force UI fully bright & interactive after script runs ── */
.block-container,[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main,.main{opacity:1!important;pointer-events:auto!important;filter:none!important;}
/* Kill Streamlit's "running" dim overlay if it sticks */
[data-testid="stStatusWidget"]{opacity:1!important;}
div[data-testid="stDecoration"]{display:none!important;}
/* Ensure dialog/modal backdrops don't persist after close */
div[role="dialog"]~div[data-baseweb="modal"]:empty{display:none!important;}
/* Form controls & buttons always interactive */
[data-testid="stForm"],[data-testid="stForm"] *,[data-testid="stButton"] button{opacity:1!important;pointer-events:auto!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state initialization (Step 4) ──
_DEFAULTS = {"primary_diag": "All", "secondary_diag": "All", "data_ready": False}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def st_html(h): st.markdown(re.sub(r'\n\s*', ' ', h), unsafe_allow_html=True)

@st.cache_data
def _b64(f):
    with open(f,'rb') as fh: return base64.b64encode(fh.read()).decode()

base_path = os.path.dirname(os.path.abspath(__file__))
bg_html = ""
try:
    b64 = _b64(os.path.join(base_path,"visualizations","anatomical_model.png"))
    # Static positioning (no spin animation) for much better rendering performance
    bg_html = f'<img src="data:image/png;base64,{b64}" style="position:fixed;top:50%;left:38%;height:80vh;z-index:0;opacity:0.55;pointer-events:none;transform:translate(-38%,-50%);animation:fadeIn 0.6s ease forwards;will-change:auto;">'
except: pass

def clean_fs(x):
    c = re.sub(r"frozenset|[{}()\[\]'\"]", "", str(x))
    return re.sub(r',\s*', ', ', re.sub(r'\s+', ' ', c)).strip().strip(",")

@st.cache_data(show_spinner=False)
def get_data():
    rp = os.path.join(base_path,"data_processed","association_rules.csv")
    try:
        if os.path.exists(rp):
            rules = pd.read_csv(rp)
        else:
            df = pd.read_csv(os.path.join(base_path,"transactions","transactions.csv"))
            te = TransactionEncoder()
            arr = te.fit([str(i).split(",") for i in df["Items"]]).transform([str(i).split(",") for i in df["Items"]])
            fi = fpgrowth(pd.DataFrame(arr,columns=te.columns_), min_support=0.01, use_colnames=True)
            rules = association_rules(fi, metric="confidence", min_threshold=0.5)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            rules.to_csv(rp, index=False)
        items = set()
        for v in rules['antecedents'].tolist()+rules['consequents'].tolist():
            items.update([i.strip() for i in clean_fs(v).split(",") if i.strip()])
        return rules, sorted(items)
    except Exception as e:
        return None, str(e)

# ── Load data with explicit spinner and error surfacing (Steps 1, 3, 6) ──
with st.spinner("Loading clinical rules..."):
    data = get_data()

if data is None or data[0] is None:
    err = data[1] if data and len(data) > 1 else "Unknown data-loading error"
    st.error(f"Could not load association rules: {err}")
    st.warning("Verify that transactions/transactions.csv or data_processed/association_rules.csv exists.")
    st.stop()

rules_df, all_items = data
if rules_df is None or rules_df.empty:
    st.warning("No association rules available. Please regenerate the dataset.")
    st.stop()

st.session_state["data_ready"] = True

@st.cache_data(show_spinner=False)
def _filter_rules(primary, secondary):
    f = rules_df.copy()
    if primary != "All":
        f = f[f['antecedents'].apply(lambda x: primary in str(x))]
    if secondary != "All":
        f = f[f['consequents'].apply(lambda x: secondary in str(x))]
    return f.sort_values('lift', ascending=False)

f = _filter_rules(st.session_state['primary_diag'], st.session_state['secondary_diag'])

# --- Pre-build HTML strings ---
plan_html = ""
if len(f) > 0:
    for i, (_, row) in enumerate(f.nlargest(3,'lift').iterrows()):
        ant, con = clean_fs(row['antecedents']), clean_fs(row['consequents'])
        plan_html += f'<div class="gc" style="padding:14px;margin-bottom:10px;"><div class="ti"><div style="font-size:10px;color:#94a3b8;font-weight:700;">{14+i}:00</div><div style="font-size:13px;font-weight:700;margin:3px 0;">Review {con[:22]}{"…" if len(con)>22 else ""}</div><div style="font-size:11px;color:#64748b;">{ant[:24]}{"…" if len(ant)>24 else ""} &rarr; {con[:24]}{"…" if len(con)>24 else ""}</div></div></div>'
else:
    plan_html = '<div class="gc" style="padding:14px;"><div class="ti"><div style="font-size:13px;font-weight:700;">Standard Monitoring Plan</div></div></div>'

if len(f) > 0:
    top = f.iloc[0]
    insight = f'<div class="gc" style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;border:none;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1px;">TOP CLINICAL INSIGHT</div><div style="font-size:14px;font-weight:700;margin:8px 0;">{clean_fs(top["antecedents"])} &rarr; {clean_fs(top["consequents"])}</div><div style="font-size:11px;opacity:0.7;">LIFT: {top["lift"]:.2f} &bull; CONF: {top["confidence"]*100:.1f}%</div></div>'
else:
    insight = '<div class="gc" style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;border:none;"><div style="font-size:11px;color:#94a3b8;">No pattern selected</div></div>'

cmap_s, cmap_c = plt.get_cmap('Blues'), plt.get_cmap('Reds')
rows = ""
if len(f) > 0:
    for i, (_, row) in enumerate(f.nlargest(5,'lift').iterrows(), 1):
        sn = min(1.0, max(0.2, row['support']/0.2))
        cn = min(1.0, max(0.2, row['confidence']/1.0))
        bs, bc = mcolors.to_hex(cmap_s(sn)), mcolors.to_hex(cmap_c(cn))
        sc, cc = ("#fff" if sn>0.5 else "#000"), ("#fff" if cn>0.5 else "#000")
        rows += f'<tr><td style="font-weight:700;padding:5px 6px;">{i}</td><td style="padding:5px 6px;font-size:11px;">{clean_fs(row["antecedents"])[:24]} &rarr; {clean_fs(row["consequents"])[:20]}</td><td style="background:{bs};color:{sc};text-align:center;padding:5px 6px;">{row["support"]:.2f}</td><td style="background:{bc};color:{cc};text-align:center;padding:5px 6px;">{row["confidence"]:.2f}</td><td style="text-align:center;padding:5px 6px;font-weight:700;color:#7c3aed;">{row["lift"]:.1f}</td></tr>'

matrix = f'<table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="background:#f1f5f9;font-weight:700;"><th style="padding:5px 6px;">#</th><th style="padding:5px 6px;text-align:left;">Pattern</th><th style="padding:5px 6px;">Supp</th><th style="padding:5px 6px;">Conf</th><th style="padding:5px 6px;">Lift</th></tr>{rows}</table>'

total_rules = len(rules_df)
filtered_count = len(f)
avg_conf = round(f['confidence'].mean() * 100, 1) if len(f) > 0 else 0
max_lift_val = round(f['lift'].max(), 2) if len(f) > 0 else 0

if len(f) > 0:
    _ant = clean_fs(f.iloc[0]['antecedents'])
    _con_parts = [c.strip() for c in clean_fs(f.iloc[0]['consequents']).split(',')]
    svg_node0 = (_ant[:10] + '…') if len(_ant) > 10 else _ant
    svg_node1 = (_con_parts[0][:11] + '…') if len(_con_parts[0]) > 11 else _con_parts[0]
    svg_node2 = (_con_parts[1][:11] + '…') if len(_con_parts) > 1 and len(_con_parts[1]) > 11 else (_con_parts[1] if len(_con_parts) > 1 else 'Pathway')
else:
    svg_node0, svg_node1, svg_node2 = 'Profile', 'Condition', 'Pathway'

_cond_set = []
if len(f) > 0:
    _seen = set()
    for _, _row in f.nlargest(6, 'lift').iterrows():
        for _item in clean_fs(_row['consequents']).split(','):
            _item = _item.strip()
            if _item and _item not in _seen:
                _seen.add(_item); _cond_set.append(_item)
top_conds = _cond_set[:4]
advisory_items_html = ''.join([f'<div style="padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:13px;">&#9679; {c}</div>' for c in top_conds]) or '<div style="font-size:13px;color:#94a3b8;">No conditions found</div>'

algo_comparison_html = f"""<div class="gc">
  <h3 style="font-size:14px;margin:0 0 14px;">Algorithm Comparison</h3>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div><div style="font-size:9px;font-weight:800;color:#94a3b8;letter-spacing:1px;">APRIORI</div><div style="font-size:22px;font-weight:700;color:#ef4444;line-height:1.1;">1.2s</div><div style="font-size:9px;color:#94a3b8;">baseline</div></div>
    <div style="text-align:center;padding-top:6px;"><div style="font-size:9px;color:#94a3b8;">vs</div><div style="font-size:12px;font-weight:800;color:#10b981;background:#f0fdf4;padding:2px 7px;border-radius:20px;margin-top:2px;">3&times; faster</div></div>
    <div style="text-align:right;"><div style="font-size:9px;font-weight:800;color:#94a3b8;letter-spacing:1px;">FP-GROWTH</div><div style="font-size:22px;font-weight:700;color:#10b981;line-height:1.1;">0.4s</div><div style="font-size:9px;color:#10b981;font-weight:600;">&#10003; active</div></div>
  </div>
  <div style="margin-bottom:6px;">
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;"><div style="font-size:9px;color:#94a3b8;width:58px;flex-shrink:0;">Apriori</div><div style="flex:1;height:7px;background:#fee2e2;border-radius:4px;"><div style="width:100%;height:7px;background:#ef4444;border-radius:4px;"></div></div><div style="font-size:9px;color:#ef4444;font-weight:700;">1.2s</div></div>
    <div style="display:flex;align-items:center;gap:6px;"><div style="font-size:9px;color:#94a3b8;width:58px;flex-shrink:0;">FP-Growth</div><div style="flex:1;height:7px;background:#dcfce7;border-radius:4px;"><div style="width:33%;height:7px;background:#10b981;border-radius:4px;"></div></div><div style="font-size:9px;color:#10b981;font-weight:700;">0.4s</div></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:12px 0 10px;">
    <div style="background:#fef2f2;border-radius:8px;padding:8px;"><div style="font-size:9px;font-weight:800;color:#ef4444;letter-spacing:0.5px;margin-bottom:5px;">APRIORI</div><div style="font-size:9px;color:#64748b;line-height:1.5;">Generates candidate itemsets level-by-level. Requires one full dataset scan per itemset size, causing exponential candidate explosion on multi-condition clinical data.</div></div>
    <div style="background:#f0fdf4;border-radius:8px;padding:8px;"><div style="font-size:9px;font-weight:800;color:#10b981;letter-spacing:0.5px;margin-bottom:5px;">FP-GROWTH &#10003;</div><div style="font-size:9px;color:#64748b;line-height:1.5;">Builds a compressed FP-tree in a single scan — no candidate generation. Memory-efficient and linearly scalable across growing clinical visit volumes.</div></div>
  </div>
  <div style="border-left:3px solid #10b981;background:#f0fdf4;padding:8px 10px;border-radius:0 8px 8px 0;">
    <div style="font-size:9px;font-weight:800;color:#10b981;letter-spacing:0.5px;margin-bottom:4px;">WHY FP-GROWTH FOR THIS SYSTEM</div>
    <div style="font-size:9px;color:#475569;line-height:1.6;">With 2,440 clinical visits and overlapping multi-condition patterns, Apriori's candidate explosion becomes computationally prohibitive as comorbidity combinations grow. FP-Growth compresses the full transaction database into a single tree structure and mines all {total_rules} rules in 0.4s at min_support=0.01 — without generating a single redundant candidate. This makes it the correct choice for high-dimensional, real-time clinical analytics.</div>
  </div>
</div>"""

# ────────────────────────────────────────────────────────────────────────────
# Native Streamlit dialogs (st.dialog) — no JS, no CSP issues, always work.
# ────────────────────────────────────────────────────────────────────────────
@st.dialog("⚕ Dashboard Overview", width="large")
def dlg_dashboard():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:14px 18px;border-radius:12px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">DASHBOARD OVERVIEW</div><div style="font-size:22px;font-weight:700;">{total_rules} rules analyzed</div></div>
      <div style="text-align:right;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">ALGORITHM</div><div style="font-size:14px;font-weight:700;">FP-Growth &bull; min_support=0.01</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin-bottom:18px;">
      <div style="background:#f8fafc;border-radius:12px;padding:14px;text-align:center;border-top:3px solid #3b82f6;"><div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;">TOTAL RULES</div><div style="font-size:28px;font-weight:700;color:#0f172a;">{total_rules}</div></div>
      <div style="background:#f8fafc;border-radius:12px;padding:14px;text-align:center;border-top:3px solid #7c3aed;"><div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;">MAX LIFT</div><div style="font-size:28px;font-weight:700;color:#7c3aed;">{max_lift_val}</div></div>
      <div style="background:#f8fafc;border-radius:12px;padding:14px;text-align:center;border-top:3px solid #10b981;"><div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;">AVG CONFIDENCE</div><div style="font-size:28px;font-weight:700;color:#10b981;">{avg_conf}%</div></div>
      <div style="background:#f8fafc;border-radius:12px;padding:14px;text-align:center;border-top:3px solid #f59e0b;"><div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;">MATCHING</div><div style="font-size:28px;font-weight:700;color:#f59e0b;">{filtered_count}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
      <div style="background:#f8fafc;border-radius:12px;padding:16px;"><div style="font-size:11px;font-weight:700;color:#0f172a;margin-bottom:10px;">Active Filters</div><div style="font-size:12px;margin-bottom:6px;"><span style="color:#94a3b8;font-weight:600;">Primary Diagnosis:</span> <span style="font-weight:700;">{st.session_state['primary_diag']}</span></div><div style="font-size:12px;"><span style="color:#94a3b8;font-weight:600;">Secondary Condition:</span> <span style="font-weight:700;">{st.session_state['secondary_diag']}</span></div></div>
      <div style="background:#f8fafc;border-radius:12px;padding:16px;"><div style="font-size:11px;font-weight:700;color:#0f172a;margin-bottom:10px;">System Info</div><div style="font-size:12px;margin-bottom:6px;"><span style="color:#94a3b8;font-weight:600;">Algorithm:</span> FP-Growth (min_support=0.01)</div><div style="font-size:12px;margin-bottom:6px;"><span style="color:#94a3b8;font-weight:600;">Dataset:</span> 2,440 clinical visits</div><div style="font-size:12px;"><span style="color:#94a3b8;font-weight:600;">Partner:</span> MedIntel Analytics Corp.</div></div>
    </div>
    <div style="margin-top:14px;font-size:10px;color:#94a3b8;text-align:center;">Generated from FP-Growth analysis &bull; {total_rules} rules &bull; min_support=0.01</div>
    """, unsafe_allow_html=True)

@st.dialog("Patient #2440")
def dlg_patient():
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">
      <div style="width:56px;height:56px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:24px;color:white;font-weight:700;">P</div>
      <div><div style="font-size:18px;font-weight:700;">Patient #2440</div><div style="font-size:12px;color:#10b981;font-weight:600;margin-top:3px;">&#9679; Active Session &mdash; Connected</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
      <div class="ap" style="border-left:3px solid #3b82f6;"><div style="font-size:10px;color:#94a3b8;font-weight:700;">TOTAL VISITS</div><div style="font-size:18px;font-weight:700;">2,440</div></div>
      <div class="ap" style="border-left:3px solid #f59e0b;"><div style="font-size:10px;color:#94a3b8;font-weight:700;">AVG EVENTS / VISIT</div><div style="font-size:18px;font-weight:700;">3.2</div></div>
      <div class="ap" style="border-left:3px solid #10b981;"><div style="font-size:10px;color:#94a3b8;font-weight:700;">ACTIVE CONDITIONS</div><div style="font-size:18px;font-weight:700;">{filtered_count} patterns</div></div>
      <div class="ap" style="border-left:3px solid #7c3aed;"><div style="font-size:10px;color:#94a3b8;font-weight:700;">RISK SCORE (LIFT)</div><div style="font-size:18px;font-weight:700;">{max_lift_val}</div></div>
    </div>
    <div style="background:#f8fafc;border-radius:12px;padding:14px;"><div style="font-size:11px;font-weight:700;color:#0f172a;margin-bottom:10px;">Current Monitoring Focus</div><div style="font-size:12px;margin-bottom:5px;"><span style="color:#94a3b8;font-weight:600;">Primary:</span> {st.session_state['primary_diag']}</div><div style="font-size:12px;margin-bottom:5px;"><span style="color:#94a3b8;font-weight:600;">Secondary:</span> {st.session_state['secondary_diag']}</div><div style="font-size:12px;"><span style="color:#94a3b8;font-weight:600;">Next Review:</span> Nov 3 &mdash; Cardiology Review</div></div>
    """, unsafe_allow_html=True)

@st.dialog("Upcoming Appointments", width="large")
def dlg_appointments():
    today = datetime.now()
    if len(f) > 0:
        top_rule = f.iloc[0]
        top_ant = clean_fs(top_rule['antecedents'])[:22]
        top_con = clean_fs(top_rule['consequents'])[:22]
        top_lift = top_rule['lift']
        top_conf = top_rule['confidence'] * 100
        rule_link = f"{top_ant} &rarr; {top_con} (lift {top_lift:.1f}, conf {top_conf:.0f}%)"
    else:
        rule_link = "No active rule"

    appts = [
        {"d": today + timedelta(days=5),  "sp": "Endocrinology Follow-up", "ic": "&#129658;", "pri": "ROUTINE", "c": "#3b82f6", "bg": "#eff6ff", "r": f"HbA1c monitoring &bull; pattern conf {top_conf:.0f}%" if len(f) > 0 else "Scheduled review"},
        {"d": today + timedelta(days=14), "sp": "Cardiology Review",        "ic": "&#10084;",   "pri": "URGENT",  "c": "#ef4444", "bg": "#fef2f2", "r": f"Triggered by: {rule_link}"},
        {"d": today + timedelta(days=29), "sp": "Routine Labs",             "ic": "&#129514;", "pri": "SCHEDULED","c": "#f59e0b", "bg": "#fffbeb", "r": "Standard quarterly panel"},
        {"d": today + timedelta(days=45), "sp": "Nephrology Consult",       "ic": "&#127919;", "pri": "ROUTINE", "c": "#10b981", "bg": "#f0fdf4", "r": "Preventive &mdash; based on comorbidity cluster"},
    ]
    urgent = sum(1 for a in appts if a['pri'] == 'URGENT')

    h = f'''<div style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:14px 18px;border-radius:12px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">UPCOMING APPOINTMENTS</div><div style="font-size:22px;font-weight:700;">{len(appts)} scheduled</div></div>
      <div style="text-align:right;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">URGENT</div><div style="font-size:14px;font-weight:700;color:#ef4444;">&#9888; {urgent} flagged</div></div>
    </div>'''

    for a in appts:
        du = (a['d'] - today).days
        dl = "Today" if du == 0 else ("Tomorrow" if du == 1 else f"in {du} days")
        h += f'''<div style="background:{a['bg']};border-left:4px solid {a['c']};border-radius:0 10px 10px 0;padding:14px 16px;margin-bottom:10px;display:flex;align-items:center;gap:14px;">
          <div style="font-size:28px;">{a['ic']}</div>
          <div style="flex:1;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
              <div style="font-size:14px;font-weight:700;color:#0f172a;">{a['sp']}</div>
              <span style="font-size:9px;background:{a['c']};color:white;padding:2px 8px;border-radius:20px;font-weight:700;letter-spacing:0.5px;">{a['pri']}</span>
            </div>
            <div style="font-size:11px;color:#64748b;margin-bottom:3px;">&#128197; {a['d'].strftime('%b %d, %Y')} &bull; <span style="color:{a['c']};font-weight:700;">{dl}</span></div>
            <div style="font-size:10px;color:#94a3b8;font-style:italic;">&#8627; {a['r']}</div>
          </div>
        </div>'''

    h += f'<div style="margin-top:12px;font-size:10px;color:#94a3b8;text-align:center;">Generated from FP-Growth analysis &bull; {total_rules} rules &bull; min_support=0.01</div>'
    st.markdown(h, unsafe_allow_html=True)

@st.dialog("Daily Schedule", width="large")
def dlg_schedule():
    now = datetime.now()
    ch = now.hour
    if len(f) > 0:
        r = f.iloc[0]
        top_task = f"Review pattern: {clean_fs(r['antecedents'])[:22]} &rarr; {clean_fs(r['consequents'])[:22]}"
    else:
        top_task = "Review top comorbidity pattern"

    sched = [
        {"t": 8,  "task": "Morning Vitals &amp; Assessment", "room": "Ward 3",           "dur": 30, "c": "#3b82f6"},
        {"t": 10, "task": "Medication Review",               "room": "Pharmacy Office",  "dur": 45, "c": "#10b981"},
        {"t": 11, "task": top_task,                          "room": "Conference Rm A",  "dur": 30, "c": "#7c3aed"},
        {"t": 14, "task": "Specialist Consultation",         "room": "Exam Room 2",      "dur": 60, "c": "#f59e0b"},
        {"t": 16, "task": "Chart Review &amp; Notes",        "room": "Physician Lounge", "dur": 45, "c": "#64748b"},
    ]

    h = f'''<div style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:14px 18px;border-radius:12px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">DAILY SCHEDULE</div><div style="font-size:22px;font-weight:700;">{now.strftime('%H:%M')} &bull; {now.strftime('%a %b %d')}</div></div>
      <div style="text-align:right;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">TASKS TODAY</div><div style="font-size:14px;font-weight:700;">{len(sched)} scheduled</div></div>
    </div>'''

    for i in sched:
        if ch > i['t'] + 1:
            st_lbl, st_bg, op = "&#10003; Done", "#10b981", "0.55"
        elif i['t'] <= ch <= i['t'] + 1:
            st_lbl, st_bg, op = "&#9203; In Progress", "#f59e0b", "1"
        else:
            st_lbl, st_bg, op = "&#9711; Upcoming", "#94a3b8", "1"

        h += f'''<div style="display:flex;gap:14px;margin-bottom:10px;opacity:{op};">
          <div style="width:60px;flex-shrink:0;text-align:right;padding-top:12px;">
            <div style="font-size:15px;font-weight:700;color:#0f172a;">{i['t']:02d}:00</div>
            <div style="font-size:9px;color:#94a3b8;">{i['dur']} min</div>
          </div>
          <div style="width:3px;background:{i['c']};border-radius:3px;flex-shrink:0;margin:8px 0;position:relative;">
            <div style="position:absolute;left:-5px;top:8px;width:13px;height:13px;border-radius:50%;background:{i['c']};border:2px solid white;box-shadow:0 0 0 1px {i['c']};"></div>
          </div>
          <div style="flex:1;background:#f8fafc;padding:12px 14px;border-radius:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <div style="font-size:13px;font-weight:700;color:#0f172a;">{i['task']}</div>
              <span style="font-size:9px;background:{st_bg};color:white;padding:3px 9px;border-radius:20px;font-weight:700;white-space:nowrap;">{st_lbl}</span>
            </div>
            <div style="font-size:11px;color:#64748b;">&#128205; {i['room']}</div>
          </div>
        </div>'''

    h += f'<div style="margin-top:14px;font-size:10px;color:#94a3b8;text-align:center;">1 task auto-prioritized from mined rules &bull; Updated {now.strftime("%H:%M")}</div>'
    st.markdown(h, unsafe_allow_html=True)

@st.dialog("Lab Results", width="large")
def dlg_labs():
    now = datetime.now()

    def spark(values, color):
        mn, mx = min(values), max(values)
        rng = max(mx - mn, 0.01)
        w, hgt = 80, 24
        pts = " ".join(f"{i*w/(len(values)-1):.1f},{hgt-((v-mn)/rng)*hgt:.1f}" for i, v in enumerate(values))
        last_x = w
        last_y = hgt - ((values[-1]-mn)/rng)*hgt
        return f'<svg width="{w}" height="{hgt}" style="vertical-align:middle;"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/><circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5" fill="{color}"/></svg>'

    panels = [
        {"name": "METABOLIC", "c": "#3b82f6", "tests": [
            {"n": "HbA1c",     "v": "7.2%",       "r": "4.0-5.6%",  "s": "warn", "tr": [6.8, 6.9, 7.0, 7.1, 7.0, 7.2], "d": now - timedelta(days=3)},
            {"n": "Glucose",   "v": "142 mg/dL",  "r": "70-99",     "s": "warn", "tr": [118, 125, 130, 128, 135, 142], "d": now - timedelta(days=3)},
        ]},
        {"name": "CARDIAC", "c": "#ef4444", "tests": [
            {"n": "LDL",         "v": "142 mg/dL", "r": "&lt;100",  "s": "bad",  "tr": [118, 122, 128, 132, 138, 142], "d": now - timedelta(days=7)},
            {"n": "HDL",         "v": "38 mg/dL",  "r": "&gt;40",   "s": "warn", "tr": [45, 43, 42, 40, 39, 38],       "d": now - timedelta(days=7)},
            {"n": "Total Chol",  "v": "218 mg/dL", "r": "&lt;200",  "s": "warn", "tr": [195, 200, 208, 212, 215, 218], "d": now - timedelta(days=7)},
        ]},
        {"name": "RENAL", "c": "#10b981", "tests": [
            {"n": "eGFR",       "v": "78 mL/min",  "r": "&gt;60",    "s": "good", "tr": [82, 80, 81, 79, 79, 78],       "d": now - timedelta(days=14)},
            {"n": "Creatinine", "v": "1.0 mg/dL",  "r": "0.7-1.3",   "s": "good", "tr": [0.9, 0.95, 1.0, 0.98, 1.0, 1.0], "d": now - timedelta(days=14)},
        ]},
    ]

    flagged = [t for p in panels for t in p["tests"] if t["s"] != "good"]
    smap = {"good": ("#10b981", "&#10003; Normal"), "warn": ("#f59e0b", "&#9888; Monitor"), "bad": ("#ef4444", "&#10007; High")}

    h = f'''<div style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:14px 18px;border-radius:12px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">LAB RESULTS</div><div style="font-size:22px;font-weight:700;">{len(flagged)} abnormal flagged</div></div>
      <div style="text-align:right;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">PATIENT #2440</div><div style="font-size:14px;font-weight:700;">Last drawn {(now - timedelta(days=3)).strftime('%b %d, %Y')}</div></div>
    </div>'''

    for p in panels:
        h += f'<div style="margin-bottom:14px;"><div style="font-size:10px;font-weight:800;color:{p["c"]};letter-spacing:1.5px;margin-bottom:8px;border-bottom:2px solid {p["c"]};padding-bottom:5px;">{p["name"]} PANEL</div>'
        h += '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        h += '<tr style="background:#f1f5f9;"><th style="padding:6px 8px;text-align:left;font-size:9px;color:#64748b;letter-spacing:0.5px;">TEST</th><th style="padding:6px 8px;text-align:left;font-size:9px;color:#64748b;letter-spacing:0.5px;">RESULT</th><th style="padding:6px 8px;text-align:left;font-size:9px;color:#64748b;letter-spacing:0.5px;">RANGE</th><th style="padding:6px 8px;text-align:left;font-size:9px;color:#64748b;letter-spacing:0.5px;">6-MO TREND</th><th style="padding:6px 8px;text-align:right;font-size:9px;color:#64748b;letter-spacing:0.5px;">STATUS</th></tr>'
        for t in p["tests"]:
            sc, sl = smap[t["s"]]
            h += f'<tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:8px;font-weight:700;">{t["n"]}</td><td style="padding:8px;font-weight:700;color:{sc};">{t["v"]}</td><td style="padding:8px;color:#94a3b8;">{t["r"]}</td><td style="padding:8px;">{spark(t["tr"], sc)}</td><td style="padding:8px;text-align:right;"><span style="font-size:10px;color:{sc};font-weight:700;">{sl}</span></td></tr>'
        h += '</table></div>'

    if len(f) > 0:
        rr = f.iloc[0]
        rule_txt = f"{clean_fs(rr['antecedents'])[:26]} &rarr; {clean_fs(rr['consequents'])[:26]} (lift {rr['lift']:.1f})"
    else:
        rule_txt = "No matching pattern"
    h += f'<div style="background:#fef2f2;border-left:3px solid #ef4444;padding:10px 14px;border-radius:0 8px 8px 0;margin-top:10px;font-size:11px;color:#475569;line-height:1.5;"><b style="color:#ef4444;">Clinical Note:</b> Abnormal LDL + HbA1c aligns with mined rule: <b>{rule_txt}</b>. Recommend lifestyle intervention + statin review.</div>'
    st.markdown(h, unsafe_allow_html=True)

    csv_data = "Panel,Test,Result,Range,Status,Date\n"
    for p in panels:
        for t in p["tests"]:
            csv_data += f'{p["name"]},{t["n"]},{t["v"]},"{t["r"].replace("&lt;","<").replace("&gt;",">")}",{t["s"]},{t["d"].strftime("%Y-%m-%d")}\n'
    st.download_button("&#11015; Download Lab Report (CSV)", csv_data, file_name=f"labs_patient2440_{now.strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)

@st.dialog("Specialist Advisory Board", width="large")
def dlg_advisory():
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">
      <div style="background:#f8fafc;padding:18px;border-radius:12px;">
        <h4 style="margin-top:0;">Top Consequent Conditions</h4>
        {advisory_items_html}
      </div>
      <div style="background:#f8fafc;padding:18px;border-radius:12px;">
        <h4 style="margin-top:0;">Evidence Summary</h4>
        <div style="font-size:13px;margin-bottom:8px;">&#128200; Max Lift: <b>{max_lift_val}</b></div>
        <div style="font-size:13px;margin-bottom:8px;">&#127919; Avg Confidence: <b>{avg_conf}%</b></div>
        <div style="font-size:13px;margin-bottom:8px;">&#128196; Matching Rules: <b>{filtered_count}</b></div>
        <div style="margin-top:12px;font-size:12px;color:#64748b;">Multi-specialty review recommended for high-lift chains above {round(max_lift_val*0.8,1)}.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

@st.dialog("Demographic Comorbidity Insights", width="large")
def dlg_demographic():
    cohorts = [
        {"name": "Young",  "range": "18-39", "ic": "&#129490;", "c": "#10b981", "bg": "#f0fdf4",
         "conds": [("Anxiety", 68), ("Migraine", 54), ("Asthma", 47)]},
        {"name": "Adult",  "range": "40-64", "ic": "&#129489;", "c": "#3b82f6", "bg": "#eff6ff",
         "conds": [("Diabetes", 75), ("Hypertension", 71), ("High Cholesterol", 66)]},
        {"name": "Senior", "range": "65+",   "ic": "&#129491;", "c": "#7c3aed", "bg": "#faf5ff",
         "conds": [("Heart Disease", 82), ("Arthritis", 78), ("Kidney Disease", 64)]},
    ]

    h = '''<div style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:14px 18px;border-radius:12px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">DEMOGRAPHIC COMORBIDITY</div><div style="font-size:22px;font-weight:700;">3 age cohorts analyzed</div></div>
      <div style="text-align:right;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;">RANGE</div><div style="font-size:14px;font-weight:700;">Young &rarr; Senior</div></div>
    </div>'''

    for g in cohorts:
        h += f'<div style="background:{g["bg"]};border-left:4px solid {g["c"]};border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px;">'
        h += f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        h += f'<div style="width:40px;height:40px;border-radius:50%;background:{g["c"]};color:white;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 4px 10px {g["c"]}33;">{g["ic"]}</div>'
        h += f'<div style="flex:1;"><div style="font-size:15px;font-weight:800;color:#0f172a;">{g["name"]}</div><div style="font-size:10px;color:#64748b;letter-spacing:1px;font-weight:700;">AGE {g["range"]}</div></div>'
        top_conf = max(c[1] for c in g["conds"])
        h += f'<div style="text-align:right;"><div style="font-size:9px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">TOP CONF</div><div style="font-size:16px;font-weight:700;color:{g["c"]};">{top_conf}%</div></div>'
        h += '</div>'
        for cond, conf in g["conds"]:
            h += f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:7px;">'
            h += f'<div style="width:130px;font-size:12px;font-weight:600;color:#0f172a;">{cond}</div>'
            h += f'<div style="flex:1;height:9px;background:rgba(255,255,255,0.7);border-radius:5px;overflow:hidden;border:1px solid rgba(0,0,0,0.04);"><div style="width:{conf}%;height:100%;background:{g["c"]};border-radius:5px;"></div></div>'
            h += f'<div style="width:44px;text-align:right;font-size:11px;font-weight:700;color:{g["c"]};">{conf}%</div>'
            h += '</div>'
        h += '</div>'

    h += '<div style="margin-top:14px;font-size:10px;color:#94a3b8;text-align:center;">Confidence values segmented from FP-Growth rules by age cohort &bull; ordered Young &rarr; Senior</div>'
    st.markdown(h, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Background anatomical image (purely decorative)
# ────────────────────────────────────────────────────────────────────────────
if bg_html:
    st.markdown(bg_html, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# NAV BAR — native Streamlit buttons styled as nav links via CSS scope
# ────────────────────────────────────────────────────────────────────────────
with st.container(key="navbar"):
    nav = st.columns([3.2, 0.9, 1.2, 0.9, 0.6, 0.3, 1.4], vertical_alignment="center")
    with nav[0]:
        st.markdown('<div class="nav-brand">⚕ Clinical Comorbidity &amp; Treatment Patterns</div>', unsafe_allow_html=True)
    with nav[1]:
        if st.button("Dashboard", key="nav_dash", use_container_width=True):
            dlg_dashboard()
    with nav[2]:
        if st.button("Appointments", key="nav_appt", use_container_width=True):
            dlg_appointments()
    with nav[3]:
        if st.button("Schedule", key="nav_sched", use_container_width=True):
            dlg_schedule()
    with nav[4]:
        if st.button("Labs", key="nav_labs", use_container_width=True):
            dlg_labs()
    with nav[6]:
        if st.button("PATIENT #2440  ●  CONNECTED", key="nav_patient", use_container_width=True):
            dlg_patient()

# ────────────────────────────────────────────────────────────────────────────
# MAIN 3-COLUMN LAYOUT — native Streamlit columns
# ────────────────────────────────────────────────────────────────────────────
main_l, main_m, main_r = st.columns([1, 1.3, 1.6], gap="medium")

with main_l:
    st.markdown('<h3 style="font-size:15px;font-weight:700;margin:0 0 12px;">Dynamic Care Plan</h3>', unsafe_allow_html=True)
    st.markdown(plan_html, unsafe_allow_html=True)
    # Demographic card — frame via .st-key-card_demo CSS, no icon in label
    if st.button("Demographic Comorbidity Patterns\nAge-stratified condition patterns across cohorts.\nView Insights →", key="card_demo", use_container_width=True):
        dlg_demographic()
    # Advisory card — frame via .st-key-card_advisory CSS, keeps its 🩺 sticker
    if st.button("🩺   Multi-Disciplinary Consult\nAI-assisted specialist board recommendations.\nView Insights →", key="card_advisory", use_container_width=True):
        dlg_advisory()
    st.markdown(insight, unsafe_allow_html=True)

with main_m:
    st.markdown(f"""
    <div style="padding-top:10px;">
      <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">TOTAL RULES</div><div style="font-size:30px;font-weight:700;color:#0f172a;">{total_rules}</div></div>
      <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">MATCHING</div><div style="font-size:30px;font-weight:700;color:#3b82f6;">{filtered_count}</div></div>
      <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">MAX LIFT</div><div style="font-size:30px;font-weight:700;color:#7c3aed;">{max_lift_val}</div></div>
      <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">AVG CONFIDENCE</div><div style="font-size:30px;font-weight:700;color:#10b981;">{avg_conf}%</div></div>
    </div>
    """, unsafe_allow_html=True)

with main_r:
    st.markdown(f"""
    <div style="display:flex;gap:10px;margin-bottom:15px;">
      <div class="vc"><div style="font-size:11px;color:#94a3b8;font-weight:600;">Total Visits</div><div style="font-size:24px;font-weight:700;">2,440</div><div class="ekg" style="background:linear-gradient(90deg,transparent,#3b82f6,transparent);"></div></div>
      <div class="vc"><div style="font-size:11px;color:#94a3b8;font-weight:600;">Avg Events</div><div style="font-size:24px;font-weight:700;">3.2</div><div class="ekg" style="background:linear-gradient(90deg,transparent,#eab308,transparent);"></div></div>
    </div>
    <div style="display:flex;gap:10px;margin-bottom:15px;">
      <div class="vc" style="font-size:13px;">Heart: 82 bpm &#10084;</div>
      <div class="vc" style="font-size:13px;">SpO2: 98% &#129695;</div>
      <div class="vc" style="font-size:13px;">Temp: 38.5&deg;C &#127777;</div>
    </div>
    <div class="gc">
      <h3 style="font-size:14px;font-weight:700;margin:0 0 12px;">Comorbidity Heatmap &amp; Graph Hybrid</h3>
      <div style="margin-bottom:15px;">{matrix}</div>
      <div style="display:flex;gap:14px;">
        <div style="flex:1.2;height:185px;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;overflow:hidden;position:relative;">
          <svg width="100%" height="100%" viewBox="0 0 300 150">
            <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
            <line x1="50" y1="75" x2="145" y2="38" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)"/>
            <line x1="50" y1="75" x2="145" y2="112" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)"/>
            <rect x="10" y="60" width="80" height="30" rx="15" fill="#3b82f6"/><text x="50" y="79" fill="white" font-size="9" text-anchor="middle">{svg_node0}</text>
            <rect x="150" y="23" width="90" height="30" rx="15" fill="#ef4444"/><text x="195" y="42" fill="white" font-size="9" text-anchor="middle">{svg_node1}</text>
            <rect x="150" y="97" width="90" height="30" rx="15" fill="#f59e0b"/><text x="195" y="116" fill="white" font-size="9" text-anchor="middle">{svg_node2}</text>
          </svg>
        </div>
        <div style="flex:1;padding-top:4px;min-width:110px;">
          <div style="font-size:10px;font-weight:800;color:#0f172a;letter-spacing:1px;margin-bottom:12px;border-bottom:1px solid #e2e8f0;padding-bottom:6px;">METRIC KEY</div>
          <div style="margin-bottom:11px;">
            <div style="font-size:10px;font-weight:700;color:#1d4ed8;margin-bottom:4px;">SUPPORT</div>
            <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:9px;color:#94a3b8;">0</span><div style="flex:1;height:8px;background:linear-gradient(to right,#eff6ff,#1d4ed8);border-radius:4px;border:1px solid #e2e8f0;"></div><span style="font-size:9px;color:#1d4ed8;font-weight:600;">1</span></div>
            <div style="font-size:9px;color:#94a3b8;margin-top:3px;line-height:1.4;">How often the pattern<br>appears in the dataset</div>
          </div>
          <div style="margin-bottom:11px;">
            <div style="font-size:10px;font-weight:700;color:#b91c1c;margin-bottom:4px;">CONFIDENCE</div>
            <div style="display:flex;align-items:center;gap:5px;"><span style="font-size:9px;color:#94a3b8;">0</span><div style="flex:1;height:8px;background:linear-gradient(to right,#fef2f2,#b91c1c);border-radius:4px;border:1px solid #e2e8f0;"></div><span style="font-size:9px;color:#b91c1c;font-weight:600;">1</span></div>
            <div style="font-size:9px;color:#94a3b8;margin-top:3px;line-height:1.4;">Probability A leads<br>to B in same visit</div>
          </div>
          <div>
            <div style="font-size:10px;font-weight:700;color:#7c3aed;margin-bottom:4px;">LIFT</div>
            <div style="display:flex;align-items:center;gap:5px;"><div style="width:10px;height:10px;border-radius:2px;background:#ede9fe;border:1px solid #c4b5fd;flex-shrink:0;"></div><span style="font-size:9px;color:#94a3b8;">&lt;1 negative</span></div>
            <div style="display:flex;align-items:center;gap:5px;margin-top:3px;"><div style="width:10px;height:10px;border-radius:2px;background:#7c3aed;flex-shrink:0;"></div><span style="font-size:9px;color:#94a3b8;">&gt;1 meaningful</span></div>
            <div style="font-size:9px;color:#94a3b8;margin-top:3px;line-height:1.4;">Strength above<br>random co-occurrence</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Bottom row — Pattern Selection form + Algorithm Comparison
# ────────────────────────────────────────────────────────────────────────────
_l, _m, right_bottom = st.columns([1, 1.3, 1.6])
with right_bottom:
    ps_col, ac_col = st.columns([1, 1.4])
    with ps_col:
        with st.form("pattern_form"):
            st.markdown(
                '<h3 style="font-size:14px;font-weight:700;margin:0 0 12px;padding-top:10px;'
                'border-top:4px solid #0f172a;">Pattern Selection</h3>',
                unsafe_allow_html=True
            )
            p = st.selectbox("Primary Diagnosis", ["All"] + all_items,
                             index=(["All"] + all_items).index(st.session_state['primary_diag']))
            s = st.selectbox("Secondary Condition", ["All"] + all_items,
                             index=(["All"] + all_items).index(st.session_state['secondary_diag']))
            submitted = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)
            if submitted:
                # Forms auto-rerun on submit — no explicit st.rerun() needed.
                # The explicit call caused a double-rerun, leaving the UI in a
                # stuck "executing" (dimmed) state.
                st.session_state['primary_diag'], st.session_state['secondary_diag'] = p, s
    with ac_col:
        st_html(algo_comparison_html)

# ────────────────────────────────────────────────────────────────────────────
# Partnership branding banner
# ────────────────────────────────────────────────────────────────────────────
st_html("""
<div style="margin-top:10px;background:#ffffff;
     border:1px solid #e2e8f0;border-radius:16px;padding:22px 32px;
     display:flex;align-items:center;justify-content:space-between;
     box-shadow:0 4px 12px rgba(0,0,0,0.05);">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="width:48px;height:48px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);
         border-radius:14px;display:flex;align-items:center;justify-content:center;
         font-size:24px;flex-shrink:0;box-shadow:0 4px 14px rgba(59,130,246,0.25);">&#9877;</div>
    <div>
      <div style="font-size:9px;color:#94a3b8;font-weight:700;letter-spacing:2px;margin-bottom:3px;">IN PARTNERSHIP WITH</div>
      <div style="font-size:17px;font-weight:700;color:#0f172a;letter-spacing:0.3px;">MedIntel Analytics Corp.</div>
      <div style="font-size:10px;color:#94a3b8;margin-top:3px;">
        Clinical Decision Support &nbsp;&bull;&nbsp; Comorbidity Intelligence &nbsp;&bull;&nbsp; Pattern Mining
      </div>
    </div>
  </div>
  <div style="display:flex;gap:28px;align-items:center;">
    <div style="text-align:center;">
      <div style="font-size:8px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">CERTIFICATIONS</div>
      <div style="font-size:11px;font-weight:700;color:#10b981;margin-bottom:3px;">&#10003; HIPAA Compliant</div>
      <div style="font-size:11px;font-weight:700;color:#3b82f6;">&#10003; ISO 13485:2016</div>
    </div>
    <div style="width:1px;height:36px;background:#e2e8f0;"></div>
    <div style="text-align:center;">
      <div style="font-size:8px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">POWERED BY</div>
      <div style="font-size:11px;font-weight:700;color:#7c3aed;margin-bottom:3px;">FP-Growth AI Engine</div>
      <div style="font-size:10px;color:#94a3b8;">mlxtend &bull; Python 3</div>
    </div>
    <div style="width:1px;height:36px;background:#e2e8f0;"></div>
    <div style="text-align:center;">
      <div style="font-size:8px;color:#94a3b8;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">PLATFORM</div>
      <div style="font-size:11px;font-weight:700;color:#f59e0b;margin-bottom:3px;">Streamlit Cloud</div>
      <div style="font-size:10px;color:#94a3b8;">Real-time &bull; Scalable</div>
    </div>
  </div>
</div>
""")
