import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_price

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)
CORS(app, origins="*")

QUARTIERS = [
    "Tevragh Zeina", "Ksar", "Arafat", "Dar Naim",
    "Toujounine", "Sebkha", "Riyadh", "Teyarett"
]
QUARTIERS_LOWER = {q.lower(): q for q in QUARTIERS}

STATS = {
    "total_annonces": 1153,
    "prix_moyen": 4339931,
    "prix_median": 2600000,
    "nb_quartiers": 8,
    "quartiers": {
        "Tevragh Zeina": {"prix_moyen": 8209772, "prix_median": 6500000, "count": 373},
        "Teyarett":      {"prix_moyen": 3330407, "prix_median": 2900000, "count": 270},
        "Arafat":        {"prix_moyen": 1676270, "prix_median": 1300000, "count": 244},
        "Toujounine":    {"prix_moyen": 1571087, "prix_median": 1100000, "count": 115},
        "Dar Naim":      {"prix_moyen": 2489878, "prix_median": 1700000, "count": 82},
        "Ksar":          {"prix_moyen": 3610870, "prix_median": 2900000, "count": 46},
        "Riyadh":        {"prix_moyen": 3440769, "prix_median": 850000,  "count": 13},
        "Sebkha":        {"prix_moyen": 3780000, "prix_median": 3850000, "count": 10},
    }
}


@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "API Prédiction Immobilière Nouakchott",
        "version": "1.0"
    })


@app.route("/api/quartiers")
def quartiers():
    return jsonify({"quartiers": QUARTIERS})


@app.route("/api/stats")
def stats():
    return jsonify(STATS)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}

    # --- Required fields ---
    quartier = data.get("quartier")
    if not quartier:
        return jsonify({"success": False, "error": "Le champ 'quartier' est obligatoire"}), 400

    quartier_key = str(quartier).lower().strip()
    if quartier_key not in QUARTIERS_LOWER:
        return jsonify({"success": False, "error": f"Quartier invalide. Valeurs acceptées : {QUARTIERS}"}), 400
    quartier_canonical = QUARTIERS_LOWER[quartier_key]

    surface_m2 = data.get("surface_m2")
    if surface_m2 is None:
        return jsonify({"success": False, "error": "Le champ 'surface_m2' est obligatoire"}), 400
    try:
        surface_m2 = float(surface_m2)
        if surface_m2 <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Le champ 'surface_m2' doit être un nombre > 0"}), 400

    nb_chambres = data.get("nb_chambres")
    if nb_chambres is None:
        return jsonify({"success": False, "error": "Le champ 'nb_chambres' est obligatoire"}), 400
    try:
        nb_chambres = float(nb_chambres)
        if nb_chambres < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Le champ 'nb_chambres' doit être un nombre >= 0"}), 400

    nb_salons = data.get("nb_salons")
    if nb_salons is None:
        return jsonify({"success": False, "error": "Le champ 'nb_salons' est obligatoire"}), 400
    try:
        nb_salons = float(nb_salons)
        if nb_salons < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Le champ 'nb_salons' doit être un nombre >= 0"}), 400

    # --- Optional fields ---
    nb_sdb = data.get("nb_sdb")
    if nb_sdb is not None:
        try:
            nb_sdb = float(nb_sdb)
            if nb_sdb < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Le champ 'nb_sdb' doit être un nombre >= 0"}), 400

    titre = str(data.get("titre", ""))
    description = str(data.get("description", ""))
    caracteristiques = str(data.get("caracteristiques", ""))

    inputs = {
        "quartier": quartier_canonical,
        "surface_m2": surface_m2,
        "nb_chambres": nb_chambres,
        "nb_salons": nb_salons,
        "nb_sdb": nb_sdb,
        "titre": titre,
        "description": description,
        "caracteristiques": caracteristiques,
    }
    logging.info("[predict] inputs: %s", inputs)

    try:
        result = predict_price(
            quartier=quartier_canonical,
            surface_m2=surface_m2,
            nb_chambres=nb_chambres,
            nb_salons=nb_salons,
            nb_sdb=nb_sdb,
            titre=titre,
            description=description,
            caracteristiques=caracteristiques,
        )
        logging.info("[predict] result: %s", result)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        logging.exception("[predict] ERREUR predict_price: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
