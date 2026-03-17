import numpy as np
import re
import json
import joblib
import os
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'housing_model.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'features.pkl'))

with open(os.path.join(BASE_DIR, 'mappings.json'), 'r', encoding='utf-8') as f:
    mappings = json.load(f)

QTE_MEAN = mappings['quartier_te_mean']
QTE_MEDIAN = mappings['quartier_te_median']
TB_TE_MEAN = mappings['type_bien_te_mean']
Q_LABEL = mappings['quartier_label_mapping']
T_LABEL = mappings['type_label_mapping']
G = mappings['global_stats']


def extract_type_bien(titre):
    titre = str(titre)
    if re.search(r'فيلا|villa', titre, re.I): return 'villa'
    elif re.search(r'دوبلكس|ديبلكس|دبلكس|duplex|دوبليكس', titre, re.I): return 'duplex'
    elif re.search(r'آبرتمه|شقة|appartement|آبارتمه|ابارتمه', titre, re.I): return 'appartement'
    elif re.search(r'نيمرو|أرض|terrain|ارض', titre, re.I): return 'terrain'
    elif re.search(r'شانتيه|chantier', titre, re.I): return 'chantier'
    elif re.search(r'منزل', titre, re.I): return 'maison'
    elif re.search(r'دار', titre, re.I): return 'dar'
    else: return 'autre'


def _count_kw(text, kws):
    text = str(text)
    return sum(1 for k in kws if k in text)


def _extract_price(text):
    text = str(text)
    m = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:مليون|ملايين|مليو)', text)
    if m:
        try:
            v = float(m[0].replace(',', '.'))
            if 0.1 <= v <= 200:
                return v * 1_000_000
        except:
            pass
    return 0


def predict_price(quartier, surface_m2, nb_chambres, nb_salons,
                  nb_sdb=None, titre='', description='', caracteristiques='',
                  date_publication=None):
    """
    Prédit le prix d'un bien immobilier à Nouakchott.
    
    Returns:
        dict: prix_estime, prix_min, prix_max, prix_m2, quartier, type_bien
    """
    qc = str(quartier).lower().strip()
    
    # Imputation
    sdb = nb_sdb if nb_sdb is not None else 0
    sdb_miss = 1 if nb_sdb is None else 0
    ch = nb_chambres if nb_chambres is not None else G['nb_chambres_median']
    ch_miss = 1 if nb_chambres is None else 0
    sal = nb_salons if nb_salons is not None else G['nb_salons_median']
    sal = min(sal, 10); ch = min(ch, 15)
    
    # Date — si pas de date fournie, on simule une annonce récente (max_date)
    max_dt = datetime.strptime(G['max_date'], '%Y-%m-%d')
    if date_publication:
        dt = datetime.strptime(str(date_publication), '%Y-%m-%d')
    else:
        dt = max_dt
    days = max((max_dt - dt).days, 0)
    
    # Caractéristiques
    car = str(caracteristiques)
    tf = 1 if 'Titre foncier' in car else 0
    gar = 1 if 'Garage' in car else 0
    cam = 1 if ('Caméra' in car or 'Camera' in car) else 0
    bm = re.search(r'(\d+)\s*balcon', car)
    bal = float(bm.group(1)) if bm else 0
    rm = re.search(r'Taille rue:\s*([\d.]+)', car)
    rue = float(rm.group(1)) if rm else 0
    has_car = 1 if car and car != '' and car != 'nan' else 0
    pipes = car.count('|')
    
    # NLP
    tb = extract_type_bien(titre)
    tb_enc = T_LABEL.get(tb, T_LABEL.get('autre', 0))
    
    lux = ['فاخر','لوكس','luxe','فخم','ممتاز','راقي']
    opp = ['فرصة','فرصه','سمعه','سمعة']
    new = ['جديد','جديدة','neuf','مجدد','اجديده']
    com = ['تجاري','تجارية','محل','بوتيك']
    eta = ['طابق','طابقين','étage']
    
    mf = 1 if re.search('طابقين|طابق.*فوق', str(description) or '') else 0
    pm = _extract_price(description)
    pt = _extract_price(titre)
    pma = max(pm, pt)
    
    # Interactions
    ls = np.log1p(surface_m2)
    npt = ch + sal + sdb
    spp = surface_m2 / (npt + 1)
    cxs = ch * surface_m2
    ss = surface_m2 ** 2
    sqr = np.sqrt(surface_m2)
    cr = ch / (surface_m2 + 1)
    
    # Target encoding
    qm = QTE_MEAN.get(qc, G['prix_mean'])
    qmed = QTE_MEDIAN.get(qc, G['prix_median'])
    tm = TB_TE_MEAN.get(tb, G['prix_mean'])
    sxq = surface_m2 * qm / 1e6
    lq = np.log1p(qm)
    qe = Q_LABEL.get(qc, 0)
    
    features = np.array([
        surface_m2, ch, sal, sdb, sdb_miss, ch_miss,
        days, dt.month, dt.weekday(),
        tf, gar, cam, bal, rue, has_car, pipes, tb_enc,
        len(str(description)), len(str(titre)),
        _count_kw(titre,lux), _count_kw(titre,opp), _count_kw(titre,new),
        _count_kw(titre,com), _count_kw(titre,eta),
        _count_kw(description,lux), _count_kw(description,opp), _count_kw(description,new),
        _count_kw(description,com), _count_kw(description,eta),
        pm, pma, mf,
        ls, npt, spp, cxs, ss, sqr, cr,
        qe, qm, qmed, tm, sxq, lq,
    ]).reshape(1, -1)
    
    logging.info("[predict_price] features shape: %s", features.shape)
    logging.info("[predict_price] days=%s month=%s qm=%s qe=%s tb=%s ls=%s",
                 days, dt.month, qm, qe, tb, ls)
    log_pred = model.predict(features)[0]
    logging.info("[predict_price] log_pred=%.4f  raw_prix=%.0f", log_pred, np.expm1(log_pred))
    prix = max(float(np.expm1(log_pred)), 100_000)

    return {
        'prix_estime': round(prix),
        'prix_min': round(prix * 0.80),
        'prix_max': round(prix * 1.20),
        'prix_m2': round(prix / surface_m2) if surface_m2 > 0 else 0,
        'quartier': quartier,
        'type_bien': tb,
    }


if __name__ == '__main__':
    tests = [
        ('Tevragh Zeina', 300, 4, 2, 2, 'فيلا للبيع', 'فيلا فاخرة', 'Titre foncier | Garage'),
        ('Arafat', 150, 3, 1, None, 'دار للبيع', 'دار مظبوطة', ''),
        ('Toujounine', 180, 3, 2, None, 'منزل', '', 'Titre foncier'),
        ('Teyarett', 200, 5, 2, 2, 'ديبلكس', '25 مليون', 'Garage'),
    ]
    for q, s, ch, sa, sdb, t, d, car in tests:
        r = predict_price(q, s, ch, sa, sdb, t, d, car)
        print(f"{q:18s} {s}m² → {r['prix_estime']:>12,} MRU ({r['type_bien']})")
