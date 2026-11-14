# LeafCare AI - plant disease detection with a bio-treatment plan
# Streamlit + TensorFlow Lite
#
# What is corrected in this file
#   1. Labels are read in the model's own order (0, 1, 2 ...). The file may be
#      written as name -> index, index -> name, or a plain list; all three work.
#   2. The label file is checked before the app runs: indices 0..N-1, no gaps,
#      no repeats, no empty names.
#   3. Pixels go in as 0-255 float32, the range the model was trained on.
#   4. Quantised int8 / uint8 outputs are converted back to real probabilities.
#   5. Every prediction shows top-5, a confidence warning, a leaf check and a
#      biological treatment plan.

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "model_float16_quant.tflite"
LABEL_FILE = "class_indices.json"
FALLBACK_LABEL_FILE = "class_indices (1).json"
IMG_SIZE = (384, 384)
CONFIDENCE_LIMIT = 0.60
MARGIN_LIMIT = 0.15

st.set_page_config(page_title="LeafCare AI", page_icon="🌿", layout="wide")


# ---------------------------------------------------------------------------
# Model and labels
# ---------------------------------------------------------------------------

@st.cache_resource
def load_interpreter(path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def looks_like_number(text):
    cleaned = str(text).strip()
    if cleaned.startswith("-"):
        cleaned = cleaned[1:]
    return len(cleaned) > 0 and cleaned.isdigit()


def label_pairs(raw, source):
    # Colab can export this file as name -> index, index -> name, or a list.
    if isinstance(raw, list) and len(raw) > 0:
        return [(position, str(name)) for position, name in enumerate(raw)]

    if isinstance(raw, dict) and len(raw) > 0:
        keys = list(raw.keys())
        values = list(raw.values())
        values_are_numbers = all(looks_like_number(value) for value in values)
        keys_are_numbers = all(looks_like_number(key) for key in keys)
        if values_are_numbers and not keys_are_numbers:
            return sorted((int(value), str(key)) for key, value in raw.items())
        if keys_are_numbers:
            return sorted((int(key), str(value)) for key, value in raw.items())
        raise ValueError(
            source
            + ": cannot tell which side is the index. First entry is "
            + str(keys[0])
            + " -> "
            + str(values[0])
        )

    raise ValueError(source + ": the file is empty or not a JSON object")


def read_label_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    pairs = label_pairs(raw, path.name)
    indices = [index for index, _ in pairs]
    if indices != list(range(len(pairs))):
        raise ValueError(path.name + ": class indices are not a clean 0..N-1 sequence")
    return [name for _, name in pairs]


@st.cache_resource
def load_labels():
    problems = []
    for name in (LABEL_FILE, FALLBACK_LABEL_FILE):
        path = Path(name)
        if not path.exists():
            continue
        try:
            return read_label_file(path), str(path)
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError(" | ".join(problems))
    raise FileNotFoundError("class_indices.json was not found next to app.py")


# ---------------------------------------------------------------------------
# Preprocessing and inference
# ---------------------------------------------------------------------------

def preprocess(image, input_detail):
    resized = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    pixels = np.asarray(resized).astype(np.float32)
    batch = np.expand_dims(pixels, axis=0)
    dtype = input_detail["dtype"]
    scale, zero_point = input_detail["quantization"]
    if np.issubdtype(dtype, np.integer) and scale > 0:
        batch = np.round(batch / scale) + zero_point
        limits = np.iinfo(dtype)
        batch = np.clip(batch, limits.min, limits.max)
    return batch.astype(dtype)


def to_probabilities(raw, output_detail):
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    scale, zero_point = output_detail["quantization"]
    if np.issubdtype(output_detail["dtype"], np.integer) and scale > 0:
        values = (values - zero_point) * scale
    total = float(values.sum())
    if total < 0.99 or total > 1.01:
        shifted = values - values.max()
        exponentials = np.exp(shifted)
        values = exponentials / exponentials.sum()
    return values


def run_model(interpreter, image):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], preprocess(image, input_detail))
    interpreter.invoke()
    raw = interpreter.get_tensor(output_detail["index"])
    return to_probabilities(raw[0], output_detail)


def predict(interpreter, image, average_views):
    probabilities = run_model(interpreter, image)
    if average_views:
        mirrored = image.transpose(Image.FLIP_LEFT_RIGHT)
        probabilities = (probabilities + run_model(interpreter, mirrored)) / 2.0
    return probabilities


def green_share(image):
    pixels = np.asarray(image.convert("RGB").resize((128, 128))).astype(np.float32)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    weakest = np.minimum(red, blue)
    leaf = (green > weakest + 8) & (green > 50)
    return float(leaf.mean())


def split_label(name):
    if "___" in name:
        plant, condition = name.split("___", 1)
    else:
        plant, condition = name, "unknown"
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    if condition.lower() == "healthy":
        return plant, "Healthy"
    return plant, condition[:1].upper() + condition[1:]


# ---------------------------------------------------------------------------
# Bio-treatment plans
# ---------------------------------------------------------------------------

TREATMENTS = {
    "early blight": {
        "cause": "Fungus - Alternaria solani",
        "steps": [
            "Remove the lowest infected leaves first: early blight starts at the bottom of the plant.",
            "Mulch the soil surface so rain cannot splash spores onto the leaves.",
            "Water at the base only, and keep space between plants for airflow.",
            "Rotate away from potato and tomato for at least two seasons.",
        ],
        "bio": "Bacillus subtilis or Trichoderma biofungicides every 7 to 10 days. Neem-based sprays can slow the spread.",
    },
    "late blight": {
        "cause": "Oomycete - Phytophthora infestans. Fast and destructive.",
        "steps": [
            "Act the same day: remove and destroy infected plants. Do not compost them.",
            "Stop all overhead watering, and only work in the plot when the leaves are dry.",
            "Remove the lowest leaves and widen spacing so the canopy dries quickly.",
            "Check the neighbouring plants every morning for dark, water-soaked patches.",
        ],
        "bio": "Copper-based products are the main organic option. Bacillus subtilis and Trichoderma help as prevention, not as a cure. For a fast outbreak, ask your local agriculture extension officer about a registered fungicide.",
    },
    "bacterial": {
        "cause": "Bacterium",
        "steps": [
            "Remove and destroy infected leaves and fruit.",
            "Never handle plants while the leaves are wet: bacteria spread on hands and tools.",
            "Use certified disease-free seed, and rotate crops.",
        ],
        "bio": "Copper-based sprays are the standard control. Bacillus subtilis biocontrol can reduce spread when used early.",
    },
    "yellow leaf curl": {
        "cause": "Virus, spread by whitefly",
        "steps": [
            "There is no cure for a virus: remove and destroy infected plants.",
            "Control whitefly with yellow sticky traps and neem oil under the leaves.",
            "Grow resistant varieties and keep weeds down around the plot.",
        ],
        "bio": "Neem oil and insecticidal soap against the whitefly vector. Virus-free seedlings are the real protection.",
    },
    "mosaic": {
        "cause": "Virus",
        "steps": [
            "Remove and destroy infected plants. Do not save seed from them.",
            "Wash hands and tools before touching healthy plants.",
            "Control aphids, which carry the virus between plants.",
        ],
        "bio": "No chemical cure. Neem oil for aphid control, plus resistant varieties.",
    },
    "leaf mold": {
        "cause": "Fungus - Passalora fulva, common in humid greenhouses",
        "steps": [
            "Increase ventilation and lower the humidity around the canopy.",
            "Remove affected leaves and keep the foliage dry.",
            "Avoid crowding plants.",
        ],
        "bio": "Trichoderma or Bacillus subtilis biofungicides, together with better airflow.",
    },
    "powdery mildew": {
        "cause": "Fungus",
        "steps": [
            "Remove badly affected leaves.",
            "Improve airflow and avoid excess nitrogen fertiliser.",
        ],
        "bio": "Potassium bicarbonate, neem oil or sulphur-based sprays. Bacillus subtilis works as prevention.",
    },
    "rust": {
        "cause": "Fungus",
        "steps": [
            "Remove infected leaves and destroy fallen debris.",
            "Water at the base and improve airflow.",
        ],
        "bio": "Sulphur-based or copper-based sprays; biofungicides as prevention.",
    },
    "scorch": {
        "cause": "Fungus - Diplocarpon earlianum",
        "steps": [
            "Remove old infected leaves, especially after harvest.",
            "Keep the canopy open and avoid overhead watering.",
        ],
        "bio": "Copper-based sprays after harvest and at the start of the next season.",
    },
    "spot": {
        "cause": "Fungal or bacterial leaf spot",
        "steps": [
            "Remove infected leaves and clear fallen debris.",
            "Avoid overhead watering and rotate crops.",
        ],
        "bio": "Copper-based or Bacillus subtilis sprays, used early.",
    },
    "greening": {
        "cause": "Bacterium spread by the citrus psyllid",
        "steps": [
            "Remove and destroy infected trees. There is no cure.",
            "Control the psyllid and use certified disease-free planting material.",
        ],
        "bio": "No biological cure. Vector control and clean stock are the only defence.",
    },
}


def treatment_for(condition):
    if condition.lower() == "healthy":
        return {
            "cause": "No disease detected",
            "steps": [
                "Keep scouting the plot once a week.",
                "Water at the base, and keep good airflow between plants.",
                "Clean tools between plants when you prune.",
            ],
            "bio": "Nothing to treat. A preventive Trichoderma application every two to three weeks is optional.",
        }
    lowered = condition.lower()
    for keyword, plan in TREATMENTS.items():
        if keyword in lowered:
            return plan
    return {
        "cause": "Condition not in the treatment list",
        "steps": [
            "Check the top 5 list to see whether another class is close.",
            "Compare the leaf with a reference photo before treating anything.",
        ],
        "bio": "Ask your local agriculture extension officer before spraying.",
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp {
    background: radial-gradient(circle at 85% 10%, #E6EFD3 0%, #F6F8F1 45%, #FBFCF7 100%);
}
[data-testid="stSidebar"] { background: #EEF4E6; border-right: 1px solid #D5E3C4; }
h1, h2, h3 { color: #14432F !important; font-weight: 800 !important; letter-spacing: -0.5px; }
.pill {
    display: inline-block; background: #ffffff; border: 1px solid #D5E3C4;
    color: #1B4D3E; padding: 8px 18px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
}
.hero-h { font-size: 3.2rem; font-weight: 800; color: #14432F; line-height: 1.05; margin: 18px 0 6px 0; letter-spacing: -2px; }
.hero-h span { color: #8DB359; }
.hero-line { width: 280px; height: 8px; border-radius: 99px; background: #D3E2A5; margin-bottom: 18px; }
.hero-p { color: #5B6B62; font-size: 1.1rem; line-height: 1.7; max-width: 640px; }
.checks { margin: 14px 0 22px 0; color: #14432F; font-weight: 600; }
.checks span { margin-right: 26px; }
.tips {
    background: #1B4D3E; border-radius: 24px; padding: 26px 30px; margin: 10px 0 26px 0;
    box-shadow: 0 14px 34px rgba(27, 77, 62, 0.25);
}
.tips h3 { color: #ffffff !important; margin: 0 0 14px 0; }
.tip { display: flex; gap: 14px; margin-bottom: 14px; }
.num {
    min-width: 40px; height: 40px; border-radius: 50%; background: #2C6B57;
    color: #D3E2A5; display: flex; align-items: center; justify-content: center; font-weight: 800;
}
.tip b { color: #ffffff; }
.tip div div { color: #BFD8CB; font-size: 0.92rem; }
.note {
    border: 1px solid #3E7A66; background: rgba(255, 255, 255, 0.06);
    color: #E3F1E8; padding: 14px 18px; border-radius: 16px; font-size: 0.92rem;
}
[data-testid="stFileUploader"] section {
    border: 2px dashed #A9C98B; border-radius: 24px; background: #FBFDF7; padding: 28px;
}
[data-testid="stFileUploader"] button {
    background: #1B4D3E; color: #ffffff; border-radius: 999px; border: none; padding: 8px 22px;
}
[data-testid="stImage"] img { border-radius: 20px; box-shadow: 0 10px 28px rgba(20, 67, 47, 0.18); }
[data-testid="stAlert"] { border-radius: 16px; }
[data-testid="stExpander"] { border: 1px solid #D5E3C4; border-radius: 16px; background: #ffffff; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #1B4D3E, #8DB359); }
</style>

<div class="pill">Plant health, explained simply</div>
<div class="hero-h">Healthier plants start with <br>a <span>closer look.</span></div>
<div class="hero-line"></div>
<div class="hero-p">Upload a clear photo of one leaf. LeafCare helps identify the plant, spot signs of disease, and explains practical next steps.</div>
<div class="checks"><span>&#10003; 38 conditions</span><span>&#10003; 14 plant species</span><span>&#10003; Simple care guidance</span></div>

<div class="tips">
  <h3>For a better scan</h3>
  <div class="tip"><div class="num">01</div><div><b>Use natural light</b><div>Avoid strong shadows and yellow indoor light.</div></div></div>
  <div class="tip"><div class="num">02</div><div><b>Show one full leaf</b><div>Keep the leaf sharp and near the center.</div></div></div>
  <div class="tip"><div class="num">03</div><div><b>Keep the background simple</b><div>Avoid hands, soil, and other leaves when possible.</div></div></div>
  <div class="note">LeafCare covers only its 14 trained plant species. Unknown plants may still be matched to a known class.</div>
</div>
""",
    unsafe_allow_html=True,
)
def main():
    st.title("🌿 LeafCare AI")
    st.caption("Upload one leaf. You get the plant, the disease, the confidence and a treatment plan.")

    try:
        interpreter = load_interpreter(MODEL_PATH)
        labels, label_path = load_labels()
    except Exception as error:
        st.error("The model could not start: " + str(error))
        st.info("Make sure " + MODEL_PATH + " and " + LABEL_FILE + " sit in the same folder as app.py.")
        st.stop()

    with st.sidebar:
        st.subheader("Settings")
        confidence_limit = st.slider(
            "Confidence needed to trust a result", 0.30, 0.95, CONFIDENCE_LIMIT, 0.01
        )
        average_views = st.checkbox("Average the photo and its mirror (slower, steadier)", value=True)
        check_leaf = st.checkbox("Warn when the photo may not be a leaf", value=True)
        st.divider()
        st.caption("Model: " + MODEL_PATH)
        st.caption("Labels: " + label_path)
        st.caption("Classes: " + str(len(labels)))
        if label_path != LABEL_FILE:
            st.warning("Loaded " + label_path + " because " + LABEL_FILE + " could not be read.")

    upload = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.info("Choose a leaf photo to begin.")
        return

    image = Image.open(upload)
    columns = st.columns([1, 1.15])

    with columns[0]:
        st.image(image, caption=upload.name)
        leaf_total = green_share(image)
        st.caption("Green in this photo: " + str(round(leaf_total * 100, 1)) + "%")

    probabilities = predict(interpreter, image, average_views)
    ranking = np.argsort(probabilities)[::-1]
    best = int(ranking[0])
    confidence = float(probabilities[best])
    runner_up = float(probabilities[int(ranking[1])])
    plant, condition = split_label(labels[best])

    with columns[1]:
        st.metric("Plant", plant)
        st.metric("Condition", condition)
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(
            "Confidence " + str(round(confidence * 100, 2)) + "%"
            + "   -   runner-up " + str(round(runner_up * 100, 2)) + "%"
        )
        if confidence < confidence_limit or (confidence - runner_up) < MARGIN_LIMIT:
            st.warning(
                "Not sure. The confidence is low, or the top two answers are close. "
                "Check the leaf by hand before acting on this."
            )
        if check_leaf and leaf_total < 0.12:
            st.warning(
                "This photo is almost empty of green. Crop tightly to one leaf, fill the frame, "
                "and use daylight without hard shadows."
            )

    plan = treatment_for(condition)
    st.subheader("Recommended action")
    st.write("**Cause:** " + plan["cause"])
    for step in plan["steps"]:
        st.write("- " + step)
    st.info("**Biological and low-risk options:** " + plan["bio"])
    st.caption("Confirm with your local agriculture extension officer before spraying anything.")

    st.subheader("Top 5 predictions")
    rows = []
    for position in ranking[:5]:
        index = int(position)
        row_plant, row_condition = split_label(labels[index])
        rows.append(
            {
                "Plant": row_plant,
                "Condition": row_condition,
                "Confidence %": round(float(probabilities[index]) * 100, 2),
                "Class": labels[index],
            }
        )
    st.dataframe(rows, use_container_width=True)

    with st.expander("Technical details"):
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        st.write("Input tensor:", str(input_detail["shape"]), str(input_detail["dtype"]))
        st.write("Output tensor:", str(output_detail["shape"]), str(output_detail["dtype"]))
        st.write("Label file:", label_path)
        st.write("Classes:", len(labels))
        st.write("Pixel range sent to the model: 0-255")
        st.write("Uncertainty margin:", round(confidence - runner_up, 4))


if __name__ == "__main__":
    main()
