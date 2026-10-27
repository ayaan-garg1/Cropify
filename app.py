import json
import os
from datetime import date

import joblib
import pandas as pd
import requests
import streamlit as st


# ---------- CONFIG ----------
st.set_page_config(
    page_title="Cropify | Smart Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "crop_recommendation_model.joblib"
DATASET_PATH = "Crop recommendation dataset.csv"
FERTILIZER_MODEL_PATH = "fertilizer_recommendation_model.joblib"
FERTILIZER_DATASET_PATH = "Fertilizer_Prediction.csv"
YIELD_MODEL_PATH = "crop_yield_model.joblib"
FERTILIZER_CROP_OPTIONS = [
    "Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley",
    "Millets", "Oil seeds", "Pulses", "Ground Nuts", "Wheat",
]
FERTILIZER_SOIL_OPTIONS = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
FERTILIZER_SOIL_MAP = {
    "Black Soil": "Black",
    "Clay soil": "Clayey",
    "Loamy soil": "Loamy",
    "Red soil": "Red",
    "Sandy soil": "Sandy",
}


# ---------- STYLE ----------
st.markdown(
    """
    <style>
    .stApp { background: #F5F7F2; color: #1D2B24; }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;
    }
    h1, h2, h3, h4 { color: #173D2D !important; font-weight: 700 !important; }
    p, li, label, [data-testid="stMarkdownContainer"] { color: #263A30; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123D2E 0%, #0B2D23 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #F3F7F1 !important;
    }
    [data-testid="stSidebar"] input { background: #FFFFFF !important; color: #182820 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #FFFFFF !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #182820 !important; }
    .hero {
        background: linear-gradient(120deg, #E3EFE5 0%, #F8F5E9 100%);
        border: 1px solid #D0E1D2; border-left: 6px solid #28734D;
        border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;
    }
    .hero h1 { color: #142B20 !important; font-size: 2.8rem !important; margin: 8px 0; }
    .hero p { color: #344B3D !important; font-size: 1.05rem; margin-bottom: 0; }
    .crop-card {
        background: linear-gradient(120deg, #123D2E 0%, #286747 100%);
        border: 1px solid #2E7450; border-radius: 18px; padding: 25px;
    }
    .crop-card h2, .crop-card p { color: #FFFFFF !important; }
    .crop-card .eyebrow { color: #D9EBDD !important; font-weight: 700; letter-spacing: 1.2px; }
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #287B51, #17563A);
        color: #FFFFFF !important; border-radius: 10px; font-weight: 700;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #DCE6DD;
        border-radius: 13px; padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_fertilizer_model():
    return joblib.load(FERTILIZER_MODEL_PATH)


@st.cache_resource
def load_yield_bundle():
    return joblib.load(YIELD_MODEL_PATH)


@st.cache_data
def load_water_reference():
    """Return median WATERREQUIRED per crop from the original training dataset."""
    if not os.path.exists(DATASET_PATH):
        return None
    source = pd.read_csv(DATASET_PATH)
    needed = {"CROPS", "WATERREQUIRED"}
    if not needed.issubset(source.columns):
        return None
    source["WATERREQUIRED"] = pd.to_numeric(source["WATERREQUIRED"], errors="coerce")
    source["crop_key"] = source["CROPS"].astype(str).str.strip().str.casefold()
    summary = (
        source.dropna(subset=["WATERREQUIRED"])
        .groupby("crop_key", as_index=False)
        .agg(crop_name=("CROPS", "first"), median_water_requirement=("WATERREQUIRED", "median"))
    )
    if summary.empty:
        return None
    return summary


def grouped_model_importance(model):
    """Aggregate one-hot/random-forest importances back to the original input fields."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        labels = {
            "SOIL": "Soil type",
            "SEASON": "Growing season",
            "WATER_SOURCE": "Water source",
            "SOIL_PH": "Soil pH",
            "TEMP": "Temperature",
            "RELATIVE_HUMIDITY": "Relative humidity",
            "N": "Nitrogen (N)",
            "P": "Phosphorus (P)",
            "K": "Potassium (K)",
        }
        totals = {key: 0.0 for key in labels}
        for transformed_name, importance in zip(transformed_names, importances):
            raw_name = str(transformed_name).split("__", 1)[-1]
            for key in labels:
                if raw_name == key or raw_name.startswith(key + "_"):
                    totals[key] += float(importance)
                    break
        rows = [
            {"Input factor": labels[key], "Relative model importance (%)": round(value * 100, 2)}
            for key, value in totals.items()
        ]
        return pd.DataFrame(rows).sort_values("Relative model importance (%)", ascending=False)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


# ---------- HEADER ----------
st.markdown(
    """
    <div class="hero">
        <div style="color:#315B43;font-size:13px;font-weight:700;letter-spacing:2px;">
            SMART AGRICULTURE PLATFORM
        </div>
        <h1>🌱 Cropify</h1>
        <p>See your top predicted crop and its matching yield estimate, with fertilizer and sustainability guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("**Your farm, your data, smarter decisions.** Enter the field details in the panel on the left.")

SOIL_OPTIONS = [
    "Alluvial soil", "Black Soil", "Clay soil", "Laterite soil",
    "Loamy soil", "Red soil", "Sandy soil",
]
SEASON_OPTIONS = ["kharif", "rabi", "Zaid"]
WATER_OPTIONS = ["irrigated", "rainfed"]

# ---------- INPUTS ----------
with st.sidebar:
    st.markdown("## 🌿 Farm Profile")
    st.caption("Use actual soil-test results where possible.")

    with st.form("farm_details"):
        city = st.text_input("City or town", placeholder="e.g. Vellore")
        season = st.selectbox("Growing season", SEASON_OPTIONS, index=0)
        soil_type = st.selectbox("Soil type", SOIL_OPTIONS)
        water_source = st.selectbox("Water source", WATER_OPTIONS, index=1)

        # Reuse the soil selected above for the fertilizer model.
        # Translate matching labels to the names used by its training dataset.
        # If a soil category is not represented in that dataset, pass the original
        # label through: this model's OneHotEncoder(handle_unknown="ignore") safely
        # ignores that unknown category while still using the other input features.
        fertilizer_soil_type = FERTILIZER_SOIL_MAP.get(soil_type, soil_type)

        st.markdown("### Soil test values")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=50.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=40.0, step=1.0)
        potassium = st.number_input("Potassium (K)", min_value=0.0, value=40.0, step=1.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        soil_moisture = st.number_input(
            "Soil moisture (%) for fertilizer model",
            min_value=0.0, max_value=100.0, value=40.0, step=1.0,
            help="The fertilizer dataset records moisture values mainly from 25 to 65. Use a measured value when available.",
        )
        submitted = st.form_submit_button("Analyze My Farm", use_container_width=True)

if not submitted:
    st.markdown("## Your Cropify dashboard")
    a, b, c = st.columns(3)
    a.metric("ML engine", "Random Forest")
    b.metric("Water comparison", "Dataset-based")
    c.metric("Crop classes", "57")
    st.info("Fill in the Farm Profile panel and select **Analyze My Farm** to get crop and sustainability results.")
    st.markdown(
        """
        ### What Cropify provides
        - Ranks candidate crops by the trained ML model's suitability scores.
        - Compares recorded water requirements for those candidates when the dataset supports it.
        - Keeps suitability and water demand separate, so one does not misleadingly inflate the other.
        - Offers location-aware advice, weather screening, soil-health checks, an optional historical yield estimate, and a downloadable report.
        """
    )
    st.caption("Keep the trained model and dataset in the same folder as app.py.")
    st.stop()

if not city.strip():
    st.error("Please enter your farm's city or town.")
    st.stop()

try:
    model = load_model()

    # ---------- LOCATION ----------
    with st.spinner("Finding your location..."):
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city.strip(), "count": 5, "language": "en", "format": "json"},
            timeout=20,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

    if not geo_data.get("results"):
        st.error("Location not found. Try a nearby city or town.")
        st.stop()

    place = geo_data["results"][0]
    latitude, longitude = place["latitude"], place["longitude"]

    st.markdown("## 📍 Farm overview")
    st.write(
        f"**{place['name']}, {place.get('country', '')}** · {season} season · "
        f"{soil_type} · {water_source}"
    )

    # ---------- HISTORICAL WEATHER ----------
    current_year = date.today().year
    start_date = f"{current_year - 6}-01-01"
    end_date = f"{current_year - 1}-12-31"

    with st.spinner(f"Getting historical weather for the {season} season..."):
        weather_response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
                "timezone": "auto",
            },
            timeout=60,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

    daily = weather_data.get("daily", {})
    if not daily.get("time"):
        st.error("No historical weather data was returned for this location.")
        st.stop()

    weather = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature": daily.get("temperature_2m_mean"),
        "humidity": daily.get("relative_humidity_2m_mean"),
        "rainfall_mm": daily.get("precipitation_sum"),
    })
    annual_rainfall = None
    if weather["rainfall_mm"].notna().any():
        rainfall_by_year = weather.groupby(weather["date"].dt.year)["rainfall_mm"].sum(min_count=1)
        if not rainfall_by_year.empty:
            annual_rainfall = float(rainfall_by_year.mean())

    season_months = {
        "kharif": [6, 7, 8, 9],
        "rabi": [10, 11, 12, 1, 2, 3],
        "Zaid": [3, 4, 5, 6],
    }
    months = season_months[season]
    selected = weather[weather["date"].dt.month.isin(months)].copy()
    if season == "rabi":
        selected["season_year"] = selected["date"].dt.year + (
            selected["date"].dt.month >= 10
        ).astype(int)
    else:
        selected["season_year"] = selected["date"].dt.year

    complete_years = [
        year for year, group in selected.groupby("season_year")
        if set(group["date"].dt.month.unique()) == set(months)
    ]
    complete = selected[selected["season_year"].isin(complete_years)]
    if complete.empty:
        st.error("Not enough complete historical seasons were found for this location.")
        st.stop()

    temperature = float(complete["temperature"].mean())
    humidity = float(complete["humidity"].mean())
    seasonal_rainfall = None
    if "rainfall_mm" in complete.columns and complete["rainfall_mm"].notna().any():
        rainfall_by_season = complete.groupby("season_year")["rainfall_mm"].sum(min_count=1)
        if not rainfall_by_season.empty:
            seasonal_rainfall = float(rainfall_by_season.mean())
    if pd.isna(temperature) or pd.isna(humidity):
        st.error("Weather data contains missing temperature or humidity values.")
        st.stop()

    # ---------- PRIMARY RESULT: TOP CROP + YIELD FOR THAT SAME CROP ----------
    farm_data = pd.DataFrame([{
        "SOIL": soil_type,
        "SEASON": season,
        "WATER_SOURCE": water_source,
        "SOIL_PH": ph,
        "TEMP": temperature,
        "RELATIVE_HUMIDITY": humidity,
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
    }])

    if not hasattr(model, "predict_proba"):
        st.error("The loaded model does not support ranked predictions. Please use the supplied Random Forest pipeline.")
        st.stop()

    probabilities = model.predict_proba(farm_data)[0]
    classes = model.classes_
    candidates = sorted(
        [(str(crop), float(score)) for crop, score in zip(classes, probabilities)],
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    top_crop, top_crop_score = candidates[0]

    st.markdown("## 🌾 Your predicted crop")
    st.markdown(
        f"""<div class="crop-card">
            <p class="eyebrow">BEST MATCH FROM THE CROP ML MODEL</p>
            <h2 style="font-size:2.2rem;margin:8px 0;">{top_crop.title()}</h2>
            <p>Model suitability score: {top_crop_score:.1%}</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("The model score is not a probability of harvest or a guarantee that the crop will grow successfully.")

    # Estimate yield only for the crop predicted above. Never silently substitute a different crop.
    yield_estimate = None
    yield_crop_selected = None
    yield_model_features = None
    yield_bundle = None
    yield_season = {"kharif": "kharif", "rabi": "rabi", "Zaid": "whole year"}.get(season, season.strip().casefold())

    if os.path.exists(YIELD_MODEL_PATH):
        try:
            yield_bundle = load_yield_bundle()
        except Exception as model_error:
            st.warning(f"Could not load the crop-yield model: {model_error}")

    if not isinstance(yield_bundle, dict) or "model" not in yield_bundle:
        st.info(
            f"Yield prediction is not set up yet. Place the trained `{YIELD_MODEL_PATH}` file in this folder. "
            "The top crop prediction is still available."
        )
    else:
        yield_crop_options = list(yield_bundle.get("crop_options", []))
        yield_crop_selected = next(
            (crop for crop in yield_crop_options if str(crop).strip().casefold() == top_crop.strip().casefold()),
            None,
        )
        if yield_crop_selected is None:
            st.info(
                f"The yield model does not support **{top_crop.title()}**, so it cannot estimate yield for this exact predicted crop. "
                "No different crop has been substituted."
            )
        else:
            yield_state = str(place.get("admin1", "")).strip().casefold()
            yield_district = str(place.get("admin2") or place.get("name", "")).strip().casefold()
            yield_model_features = pd.DataFrame([{
                "Crop": yield_crop_selected.casefold(),
                "Season": yield_season,
                "State": yield_state,
                "District": yield_district,
                "Temperature": temperature,
                "Rainfall": annual_rainfall if annual_rainfall is not None else float("nan"),
                "Humidity": humidity,
                "pH": ph,
                "Nitrogen": nitrogen,
                "Phosphorus": phosphorus,
                "Potassium": potassium,
            }])
            try:
                yield_estimate = float(yield_bundle["model"].predict(yield_model_features)[0])
                yield_col, yield_info = st.columns([1, 2])
                yield_col.metric("Estimated yield for this crop", f"{max(0.0, yield_estimate):,.0f} kg/ha")
                yield_info.write(
                    f"**Crop:** {yield_crop_selected.title()}  \n"
                    f"**Location:** {place.get('name', city)}, {place.get('admin1', place.get('country', ''))}  \n"
                    f"**Season used by yield model:** {yield_season.title()}"
                )
                if season == "Zaid":
                    st.warning("The historical dataset has no separate Zaid category; this estimate uses the dataset's 'Whole Year' category as a rough proxy.")
                known_states = set(yield_bundle.get("state_options", []))
                known_districts = set(yield_bundle.get("district_options", []))
                if yield_state not in known_states:
                    st.warning("This state was not present in the yield training data, so the estimate may be less reliable.")
                elif yield_district not in known_districts:
                    st.warning("This district was not present in the yield training data, so the estimate may be less reliable.")
                st.caption("This is a historical-data prototype estimate, not a forecast or a guarantee of harvest. Check local agricultural statistics.")
            except (ValueError, TypeError, KeyError) as yield_error:
                st.warning(f"Yield estimate could not be generated for {top_crop.title()}: {yield_error}")
                yield_estimate = None

    st.markdown("## 🌦️ Historical seasonal climate")
    st.caption(
        f"Historical period: {start_date} to {end_date}. "
        "These are historical averages, not a forecast."
    )
    w1, w2, w3 = st.columns(3)
    w1.metric("Mean temperature", f"{temperature:.1f} °C")
    w2.metric("Mean relative humidity", f"{humidity:.1f}%")
    w3.metric("Average seasonal rainfall", f"{seasonal_rainfall:.0f} mm" if seasonal_rainfall is not None else "Unavailable")

    st.markdown("## 🧪 Soil profile")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Nitrogen (N)", f"{nitrogen:g}")
    s2.metric("Phosphorus (P)", f"{phosphorus:g}")
    s3.metric("Potassium (K)", f"{potassium:g}")
    s4.metric("Soil pH", f"{ph:g}")

    # ---------- LOCATION-AWARE SUSTAINABLE FARMING PRACTICES ----------
    st.markdown("## 🌍 Sustainable farming practices for this location")
    st.caption(
        f"Practical suggestions for {place['name']} based on historical seasonal weather, "
        "selected soil type, water source, and entered soil pH."
    )
    practice_notes = []
    if seasonal_rainfall is not None:
        if seasonal_rainfall < 150:
            practice_notes.append(
                "**Prioritize water conservation:** consider mulching, timely weeding, "
                "rainwater harvesting, and drip irrigation where feasible. The rainfall cutoff "
                "is a screening heuristic, not a local agronomic standard."
            )
        elif seasonal_rainfall < 400:
            practice_notes.append(
                "**Manage rainfall efficiently:** consider mulching and farm-level rainwater "
                "storage; schedule irrigation using crop needs and soil-moisture checks rather "
                "than a fixed calendar."
            )
        else:
            practice_notes.append(
                "**Plan for heavy rainfall:** check field drainage where waterlogging is a risk, "
                "consider suitable contour measures, and store excess rainwater where feasible."
            )
    else:
        practice_notes.append(
            "**Conserve water:** assess mulching and soil-moisture-based irrigation scheduling "
            "with local agricultural guidance. Seasonal rainfall data was unavailable."
        )

    if temperature >= 30:
        practice_notes.append(
            "**Reduce heat and moisture stress:** retain suitable crop residues as mulch and "
            "ask local agricultural guidance about heat-tolerant varieties or windbreaks."
        )

    soil_lower = soil_type.lower()
    if "sandy" in soil_lower:
        practice_notes.append(
            "**For sandy soil:** suitable organic matter and cover crops can help improve water "
            "retention; base nutrient additions on a soil test."
        )
    elif "clay" in soil_lower or "black" in soil_lower:
        practice_notes.append(
            "**For clay/black soil:** avoid working soil when very wet and check drainage and "
            "compaction risks; use residue cover where suitable."
        )
    else:
        practice_notes.append(
            "**Protect soil structure:** keep soil covered with suitable crop residues or cover "
            "crops and use soil-test-based nutrient management."
        )

    if water_source == "rainfed":
        practice_notes.append(
            "**For rainfed farming:** align sowing with the locally advised rainfall window and "
            "consider bunds or rainwater harvesting where the site is suitable."
        )
    else:
        practice_notes.append(
            "**For irrigated farming:** schedule irrigation around crop stage and soil moisture; "
            "check whether drip or another efficient method suits the crop and farm."
        )

    if ph < 5.5:
        practice_notes.append(
            "**Soil pH is acidic:** confirm with a reliable soil test and consult local guidance "
            "before applying lime or other amendments."
        )
    elif ph > 7.5:
        practice_notes.append(
            "**Soil pH is alkaline:** confirm with a soil test and local guidance before choosing "
            "an amendment; avoid treating pH alone as sufficient evidence."
        )

    for note in practice_notes:
        st.markdown(f"- {note}")
    st.caption(
        "These are rule-based suggestions, not ML predictions or a validated farm plan. "
        "Historical averages are not a forecast; check recommendations with local agricultural experts."
    )

    # ---------- WEATHER RISK SCREENING ----------
    st.markdown("## 🌦️ Weather risk screening")
    st.caption("Uses historical seasonal averages to flag conditions worth checking. It is not a forecast or a crop-disease prediction.")
    weather_alerts = []
    if seasonal_rainfall is not None:
        if seasonal_rainfall < 150:
            weather_alerts.append("**Low seasonal rainfall signal:** review water storage, soil-moisture conservation, and locally suitable drought-management practices.")
        elif seasonal_rainfall >= 800:
            weather_alerts.append("**High seasonal rainfall signal:** check field drainage, erosion risk, and whether water harvesting/storage can be managed safely.")
    if temperature >= 30:
        weather_alerts.append("**Warm seasonal average:** check crop-specific heat tolerance and plan irrigation around crop stage and soil moisture.")
    elif temperature < 10:
        weather_alerts.append("**Low seasonal average temperature:** confirm the selected crop and sowing window are suitable for local temperature conditions.")
    if humidity >= 85:
        weather_alerts.append("**High seasonal humidity:** monitor crops regularly for disease symptoms and use local integrated-pest-management guidance; humidity alone does not confirm disease risk.")
    if not weather_alerts:
        weather_alerts.append("No screening flag was triggered by these broad thresholds. This does not mean the farm has no weather risks.")
    for alert in weather_alerts:
        st.markdown(f"- {alert}")
    st.caption("Thresholds are preliminary screening heuristics, not region-specific agronomic standards. Use a current local forecast for operational decisions.")

    # ---------- SOIL HEALTH ACTION PLAN ----------
    st.markdown("## 🧪 Soil health action plan")
    soil_actions = []
    if ph < 5.5:
        soil_actions.append("**pH check:** the entered pH is acidic. Confirm with a reliable soil test and ask a local agricultural officer whether an amendment is appropriate before applying anything.")
    elif ph > 7.5:
        soil_actions.append("**pH check:** the entered pH is alkaline. Confirm the result and get local advice before choosing an amendment; pH alone is not enough to prescribe treatment.")
    else:
        soil_actions.append("**pH check:** the entered pH is within a broadly common range for many crops, but crop-specific requirements can differ. Confirm with a soil test.")

    dataset_nutrient_stats = None
    if os.path.exists(DATASET_PATH):
        try:
            nutrient_source = pd.read_csv(DATASET_PATH, usecols=["N", "P", "K"])
            dataset_nutrient_stats = nutrient_source.quantile([0.25, 0.75], numeric_only=True)
        except (ValueError, OSError, KeyError):
            dataset_nutrient_stats = None
    if dataset_nutrient_stats is not None:
        for nutrient, value, label in [("N", nitrogen, "Nitrogen"), ("P", phosphorus, "Phosphorus"), ("K", potassium, "Potassium")]:
            q1 = float(dataset_nutrient_stats.loc[0.25, nutrient])
            q3 = float(dataset_nutrient_stats.loc[0.75, nutrient])
            if value < q1:
                soil_actions.append(f"**{label}:** the entered value is below the 25th percentile of the app dataset. Treat this only as a dataset comparison—not proof of deficiency—and verify units and a laboratory soil-test report.")
            elif value > q3:
                soil_actions.append(f"**{label}:** the entered value is above the 75th percentile of the app dataset. Do not add more fertilizer based on this comparison; verify units and follow soil-test/local recommendations.")
            else:
                soil_actions.append(f"**{label}:** the entered value is within the middle 50% of the app dataset. This does not establish that the nutrient level is sufficient for your crop.")
    else:
        soil_actions.append("**NPK review:** keep the lab report and measurement units available; nutrient advice cannot be reliably inferred from the numbers alone.")
    soil_actions.append("**Good practice:** use soil-test-based nutrient planning, avoid blanket fertilizer applications, and keep suitable crop residues or cover crops on the soil where practical.")
    for action in soil_actions:
        st.markdown(f"- {action}")
    st.caption("NPK comparisons are relative to the supplied training dataset, whose measurement units may not be independently verified. They are not fertilizer prescriptions.")

    # ---------- OTHER TOP CROP OPTIONS ----------
    st.markdown("## Other crop suitability options")
    st.caption("These are the next-best options from the same crop model; scores are not yield probabilities.")
    for rank, (crop, score) in enumerate(candidates[1:], start=2):
        st.write(f"{rank}. **{crop.title()}** — model score: {score:.1%}")

    # ---------- SECOND ML MODEL: FERTILIZER CATEGORY ----------
    st.markdown("## 🧴 Experimental fertilizer-category recommendation")
    st.caption(
        "This separate classifier was trained on a small public dataset. Its output is a prototype suggestion, "
        "not a fertilizer dose or a validated prescription."
    )
    fertilizer_model = None
    fertilizer_prediction = None
    fertilizer_top_options = []
    fertilizer_crop_type = None
    fertilizer_features = None
    if os.path.exists(FERTILIZER_MODEL_PATH):
        try:
            fertilizer_model = load_fertilizer_model()
        except Exception as model_error:
            st.warning(f"Could not load the fertilizer model: {model_error}")

    if fertilizer_model is None:
        st.warning(
            f"Fertilizer model not found or could not be loaded. Run `python train_fertilizer_model.py` "
            f"with `{FERTILIZER_DATASET_PATH}` in this folder. Crop predictions still work."
        )
    else:
        crop_aliases = {
            "rice": "Paddy", "paddy": "Paddy", "groundnut": "Ground Nuts",
            "ground nuts": "Ground Nuts", "millet": "Millets", "millets": "Millets",
            "oilseed": "Oil seeds", "oilseeds": "Oil seeds", "oil seeds": "Oil seeds",
        }
        default_fertilizer_crop = None
        for candidate_crop, _ in candidates:
            normalized = candidate_crop.strip().casefold()
            normalized = crop_aliases.get(normalized, candidate_crop.strip().title())
            match = next((option for option in FERTILIZER_CROP_OPTIONS if option.casefold() == normalized.casefold()), None)
            if match:
                default_fertilizer_crop = match
                break
        default_index = FERTILIZER_CROP_OPTIONS.index(default_fertilizer_crop) if default_fertilizer_crop else 0
        fertilizer_crop_type = st.selectbox(
            "Crop to assess with the fertilizer model",
            FERTILIZER_CROP_OPTIONS,
            index=default_index,
            help="The fertilizer training dataset covers only these 11 crop categories. Select the crop you intend to grow.",
        )
        fertilizer_features = pd.DataFrame([{
            "Temperature": temperature,
            "Humidity": humidity,
            "Moisture": soil_moisture,
            "Soil_Type": fertilizer_soil_type,
            "Crop_Type": fertilizer_crop_type,
            "Nitrogen": nitrogen,
            "Potassium": potassium,
            "Phosphorus": phosphorus,
        }])
        try:
            fertilizer_probabilities = fertilizer_model.predict_proba(fertilizer_features)[0]
            fertilizer_classes = fertilizer_model.classes_
            fertilizer_top_options = sorted(
                [(str(label), float(score)) for label, score in zip(fertilizer_classes, fertilizer_probabilities)],
                key=lambda item: item[1], reverse=True,
            )[:3]
            fertilizer_prediction = fertilizer_top_options[0][0]
            st.success(f"Top model output: **{fertilizer_prediction}**")
            st.write("Other classifier outputs:")
            for label, score in fertilizer_top_options:
                st.write(f"- **{label}** — model score: {score:.1%}")
            if fertilizer_soil_type not in FERTILIZER_SOIL_OPTIONS:
                st.warning(
                    f"The selected soil ({soil_type}) is not represented in the fertilizer model's training data. "
                    "The model has ignored this unknown soil category and used the other inputs; treat this fertilizer result as less reliable."
                )
            if not 25 <= soil_moisture <= 65:
                st.warning("The entered soil-moisture value is outside the main range represented in the training dataset, so the prediction may be unreliable.")
            st.info(
                "Important: the training dataset has only 99 rows and 7 fertilizer labels. These model scores are not calibrated probabilities, "
                "and the result does not determine fertilizer quantity or timing. Verify nutrient needs with a soil test and local agricultural guidance."
            )
        except (ValueError, TypeError, KeyError) as fertilizer_error:
            fertilizer_prediction = None
            fertilizer_top_options = []
            st.warning(f"The fertilizer prediction could not be generated for these inputs; the rest of Cropify will continue. Details: {fertilizer_error}")

    # ---------- MODEL EXPLANATION ----------
    with st.expander("🔎 Understand what influences the model"):
        importance_df = grouped_model_importance(model)
        if importance_df is not None and not importance_df.empty:
            st.markdown("The chart shows **overall feature importance across the trained model**, not a guaranteed explanation for one specific crop or farm.")
            st.bar_chart(importance_df.set_index("Input factor")["Relative model importance (%)"], use_container_width=True)
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance could not be extracted from this model file. Crop predictions are still available.")
        st.caption("Importance indicates how the Random Forest used input fields across its trees; it does not prove causation or show whether a factor increases crop suitability.")

    # ---------- SEPARATE SUITABILITY + WATER COMPARISON ----------
    st.markdown("## 💧 Water-demand comparison")
    st.caption(
        "The crop order below remains based on the ML model. Water information is shown as a separate "
        "comparison and does not change the model's suitability ranking."
    )
    water_reference = load_water_reference()

    if water_reference is None:
        st.warning(
            f"Water comparison needs `{DATASET_PATH}` in the same folder as app.py, "
            "with CROPS and WATERREQUIRED columns. Crop suitability predictions still work."
        )
        comparison = pd.DataFrame([
            {"Suitability rank": rank, "Crop": crop.title(), "Model score (%)": round(score * 100, 1),
             "Recorded water requirement": "Not available"}
            for rank, (crop, score) in enumerate(candidates, start=1)
        ])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        lookup = {
            row["crop_key"]: float(row["median_water_requirement"])
            for _, row in water_reference.iterrows()
        }
        all_water = water_reference["median_water_requirement"].astype(float)
        global_min, global_max = float(all_water.min()), float(all_water.max())
        comparison_rows = []

        for rank, (crop, model_score) in enumerate(candidates, start=1):
            demand = lookup.get(crop.strip().casefold())
            if demand is None:
                water_score = None
                water_rank = None
            else:
                # Relative indicator only: higher means lower recorded water requirement
                # compared with crop medians in this dataset, not real-world sustainability.
                water_score = (
                    100 * (global_max - demand) / (global_max - global_min)
                    if global_max > global_min else 50.0
                )
                water_rank = 1 + int((all_water < demand).sum())

            comparison_rows.append({
                "Suitability rank": rank,
                "Crop": crop.title(),
                "Model score (%)": round(model_score * 100, 1),
                "Recorded water requirement": round(demand, 2) if demand is not None else "Not available",
                "Relative lower-water indicator (0–100)": round(water_score, 1) if water_score is not None else None,
                "Water-demand rank in dataset": water_rank,
            })

        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption(
            "The lower-water indicator is calculated from each crop's median WATERREQUIRED value relative "
            "to the range of crop medians in the dataset. The dataset's measurement unit has not been verified, "
            "so these are relative dataset values—not litres, millimetres, irrigation forecasts, or a complete "
            "sustainability score. Suitability ranks are not altered by this comparison."
        )

        known = [
            (crop, score, lookup.get(crop.strip().casefold()))
            for crop, score in candidates
            if lookup.get(crop.strip().casefold()) is not None
        ]
        if known:
            lowest_demand_crop, _, lowest_demand = min(known, key=lambda item: item[2])
            top_crop, top_score = candidates[0]
            left, right = st.columns(2)
            left.metric("Top ML suitability match", top_crop.title(), f"Model score: {top_score:.1%}")
            right.metric("Lowest recorded water demand among top predictions", lowest_demand_crop.title(), f"Dataset value: {lowest_demand:g}")
            st.info(
                "Use these as two separate decision points: the first is the model's strongest match; "
                "the second has the lowest recorded water requirement among the displayed candidates. "
                "Check local crop guidance and water availability before deciding."
            )

    # ---------- DOWNLOADABLE FARM REPORT ----------
    st.markdown("## 📄 Download your farm report")
    report_lines = [
        "CROPIFY FARM ANALYSIS REPORT",
        "=" * 32,
        f"Location: {place['name']}, {place.get('country', '')}",
        f"Season: {season}",
        f"Soil type: {soil_type}",
        f"Water source: {water_source}",
        "",
        "SOIL INPUTS",
        f"Nitrogen (N): {nitrogen:g}",
        f"Phosphorus (P): {phosphorus:g}",
        f"Potassium (K): {potassium:g}",
        f"Soil pH: {ph:g}",
        "",
        "HISTORICAL SEASONAL WEATHER (not a forecast)",
        f"Period: {start_date} to {end_date}",
        f"Mean temperature: {temperature:.1f} C",
        f"Mean relative humidity: {humidity:.1f}%",
        f"Average seasonal rainfall: {seasonal_rainfall:.0f} mm" if seasonal_rainfall is not None else "Average seasonal rainfall: unavailable",
        "",
        "TOP CROP MODEL OUTPUTS (scores are not yield probabilities)",
    ]
    report_lines.extend([f"{rank}. {crop.title()} — model score {score:.1%}" for rank, (crop, score) in enumerate(candidates, start=1)])
    report_lines.extend(["", "FERTILIZER MODEL OUTPUT (experimental)"])
    if fertilizer_prediction is not None:
        report_lines.append(f"Crop assessed: {fertilizer_crop_type}")
        report_lines.append(f"Fertilizer category output: {fertilizer_prediction}")
        report_lines.append(f"Farm soil selected: {soil_type}")
        report_lines.append(f"Fertilizer model soil category supplied: {fertilizer_soil_type}")
        if fertilizer_soil_type not in FERTILIZER_SOIL_OPTIONS:
            report_lines.append("The soil category was not represented in the fertilizer training data and was ignored by the model encoder.")
        report_lines.append(f"Soil moisture input: {soil_moisture:g}%")
        report_lines.extend([f"{label}: model score {score:.1%}" for label, score in fertilizer_top_options])
    else:
        report_lines.append("Fertilizer model unavailable; no fertilizer category output was generated.")
    report_lines.extend(["", "CROP YIELD ESTIMATE (historical prototype)"])
    if yield_estimate is not None:
        report_lines.append(f"Crop assessed: {yield_crop_selected.title() if yield_crop_selected else 'Not selected'}")
        report_lines.append(f"Estimated yield: {max(0.0, yield_estimate):.0f} kg/ha")
        report_lines.append(f"Historical mean annual rainfall used for yield model: {annual_rainfall:.0f} mm" if annual_rainfall is not None else "Historical mean annual rainfall used for yield model: unavailable")
        report_lines.append("This estimate is based on historical district-level records and is not a guarantee of harvest.")
    else:
        report_lines.append("Yield model unavailable or no estimate generated.")
    report_lines.extend(["", "SUSTAINABLE FARMING PRACTICE SUGGESTIONS"])
    report_lines.extend(["- " + str(note).replace("**", "") for note in practice_notes])
    report_lines.extend(["", "WEATHER SCREENING FLAGS"])
    report_lines.extend(["- " + str(note).replace("**", "") for note in weather_alerts])
    report_lines.extend(["", "SOIL HEALTH ACTION PLAN"])
    report_lines.extend(["- " + str(note).replace("**", "") for note in soil_actions])
    report_lines.extend(["", "WATER-DEMAND COMPARISON", comparison.to_string(index=False), "", "IMPORTANT LIMITATIONS",
        "This is a prototype decision-support tool, not a guarantee of yield or sustainability.",
        "Sustainability practices and weather alerts use preliminary rules, not a separate ML model or a validated farm plan.",
        "NPK dataset comparisons are not nutrient deficiency diagnoses or fertilizer prescriptions.",
        "The fertilizer classifier was trained on a 99-row public dataset and is experimental; its output is not a dose recommendation.",
        "The yield estimate uses historical district-level data and may be unreliable for unseen locations, different practices, or future climate conditions.",
        "Check current forecasts and consult local agricultural experts before making farm decisions."])
    report_text = "\n".join(report_lines)
    st.download_button(
        "⬇️ Download farm report (.txt)",
        data=report_text,
        file_name=f"cropify_farm_report_{place['name'].strip().replace(' ', '_').lower()}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("View the exact data sent to the ML model"):
        st.dataframe(farm_data, use_container_width=True, hide_index=True)

    st.warning(
        "This is a prototype decision-support tool, not a guarantee of yield or sustainability. "
        "The water comparison uses recorded dataset values and does not yet account for rainfall, "
        "irrigation volume, soil water-holding capacity, crop rotation, or local crop calendars. "
        "Validate results with local agricultural experts."
    )

    # ---------- SAVE RECORD ----------
    record = {
        "location": place["name"],
        "country": place.get("country", ""),
        "latitude": latitude,
        "longitude": longitude,
        "season": season,
        "soil_type": soil_type,
        "water_source": water_source,
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "pH": ph,
        "seasonal_temperature": temperature,
        "seasonal_humidity": humidity,
        "average_seasonal_rainfall_mm": seasonal_rainfall,
        "average_annual_rainfall_mm": annual_rainfall,
        "sustainable_practice_suggestions": practice_notes,
        "ml_top_three": [{"crop": crop, "model_score": score} for crop, score in candidates],
        "fertilizer_model_crop": fertilizer_crop_type,
        "farm_soil_type_used_for_fertilizer": soil_type,
        "fertilizer_model_soil_category_supplied": fertilizer_soil_type,
        "fertilizer_soil_category_supported_by_model": fertilizer_soil_type in FERTILIZER_SOIL_OPTIONS,
        "soil_moisture_percent": soil_moisture,
        "fertilizer_model_output": fertilizer_prediction,
        "fertilizer_model_top_options": [{"fertilizer": label, "model_score": score} for label, score in fertilizer_top_options],
        "yield_model_crop": yield_crop_selected,
        "yield_estimate_kg_per_ha": max(0.0, yield_estimate) if yield_estimate is not None else None,
        "recommendation_method": "Crop ML ranking, experimental fertilizer-category ML output, historical crop-yield regression estimate, separate dataset-based water-demand comparison, and location-aware rule-based farming advice",
    }
    filename = "agriwise_records.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                records = []
        except (json.JSONDecodeError, OSError):
            records = []
    else:
        records = []
    records.append(record)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4)

    st.success("Farm analysis saved locally to agriwise_records.json.")

except FileNotFoundError:
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info(
        "Place crop_recommendation_model.joblib in the same folder as app.py, "
        "or run train_crop_model.py with the dataset to create it."
    )
except requests.exceptions.RequestException as error:
    st.error("Could not retrieve location or weather data from the online service.")
    st.caption(str(error))
except (KeyError, ValueError, TypeError) as error:
    st.error("Some input or returned data could not be processed.")
    st.caption(str(error))
except Exception as error:
    st.error("Something went wrong during the analysis.")
    st.caption(str(error))
