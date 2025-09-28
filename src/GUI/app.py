import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import cv2

from src.segmentation.segment import segment_image, build_overlay_image
from src.preprocessing.preprocessing import prepare_for_fnn, prepare_for_seq, prepare_for_cnn
from src.models.fnn_wrapper import FNNModel
from src.models.seq_wrapper import SeqModel
from src.models.cnn_wrapper import CNNModel

# Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Save uploaded images and overlays in the static folder so browser can fetch
UPLOAD_DIR = os.path.join("static", "uploads")
OVERLAY_DIR = os.path.join("static", "overlays")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# Saved Model Paths (kept as you had them)
FNN_MODEL_PATH = os.path.join("..", "models", "saved_models", "fnn_net.pt")
SEQ_MODEL_PATH = os.path.join("..", "models", "saved_models", "Sequential.keras")
CNN_MODEL_PATH = os.path.join("..", "models", "saved_models", "cnn_model_best.pth")

# Model Loader
_models = {"fnn": None, "seq": None, "cnn": None}

def load_models():
    if _models["fnn"] is None:
        if not os.path.exists(FNN_MODEL_PATH):
            app.logger.error(f"FNN model not found: {FNN_MODEL_PATH}")
        else:
            _models["fnn"] = FNNModel(FNN_MODEL_PATH)
    if _models["seq"] is None:
        if not os.path.exists(SEQ_MODEL_PATH):
            app.logger.error(f"SEQ model not found: {SEQ_MODEL_PATH}")
        else:
            _models["seq"] = SeqModel(SEQ_MODEL_PATH)
    if _models["cnn"] is None:
        if not os.path.exists(CNN_MODEL_PATH):
            app.logger.error(f"CNN model not found: {CNN_MODEL_PATH}")
        else:
            _models["cnn"] = CNNModel(CNN_MODEL_PATH)

# Minimal in-memory settings storage (per-model hyperparameters)
SETTINGS = {
    "model_choice": "ensemble",  # ensemble, cnn, seq, fnn
    "cnn": {"epochs": 12, "learning_rate": 0.001, "batch_size": 128},
    "seq": {"epochs": 5, "learning_rate": 0.001, "batch_size": 128},
    "fnn": {"epochs": 10, "learning_rate": 0.05, "batch_size": 200},
}

@app.route("/", methods=["GET", "POST"])

def index():
    overlay_url = None
    predictions = []  # per-crop dicts

    # Handle parameter-saving POST separately (same endpoint)
    if request.method == "POST":
        # If the user clicked Save Parameters (form includes 'action' == 'save_params')
        if request.form.get("action") == "save_params":
            # read hyperparams for each model (validate types)
            try:
                SETTINGS["model_choice"] = request.form.get("model_choice", "ensemble")
                # CNN params
                SETTINGS["cnn"]["epochs"] = int(request.form.get("cnn_epochs", SETTINGS["cnn"]["epochs"]))
                SETTINGS["cnn"]["learning_rate"] = float(request.form.get("cnn_lr", SETTINGS["cnn"]["learning_rate"]))
                SETTINGS["cnn"]["batch_size"] = int(request.form.get("cnn_bs", SETTINGS["cnn"]["batch_size"]))
                # SEQ params
                SETTINGS["seq"]["epochs"] = int(request.form.get("seq_epochs", SETTINGS["seq"]["epochs"]))
                SETTINGS["seq"]["learning_rate"] = float(request.form.get("seq_lr", SETTINGS["seq"]["learning_rate"]))
                SETTINGS["seq"]["batch_size"] = int(request.form.get("seq_bs", SETTINGS["seq"]["batch_size"]))
                # FNN params
                SETTINGS["fnn"]["epochs"] = int(request.form.get("fnn_epochs", SETTINGS["fnn"]["epochs"]))
                SETTINGS["fnn"]["learning_rate"] = float(request.form.get("fnn_lr", SETTINGS["fnn"]["learning_rate"]))
                SETTINGS["fnn"]["batch_size"] = int(request.form.get("fnn_bs", SETTINGS["fnn"]["batch_size"]))
                flash("Saved hyper-parameters (in-memory).", "success")
            except Exception as e:
                flash(f"Invalid parameter(s): {e}", "error")
            return render_template("index.html", overlay_url=overlay_url, predictions=predictions, settings=SETTINGS)

        # file upload + inference updating
        if "image" not in request.files:
            flash("No file part in request", "error")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        # Save uploaded file
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png"]:
            flash("Unsupported file type. Use .PNG", "error")
            return redirect(request.url)

        uid = uuid.uuid4().hex[:8]
        upload_name = f"upload_{uid}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, upload_name)
        file.save(upload_path)

        # Load Models
        load_models()
        if _models["fnn"] is None and _models["seq"] is None and _models["cnn"] is None:
            flash("No models found. Ensure models exist under src/models/saved_models/", "error")
            return redirect(request.url)

        model_choice = request.form.get("model_choice", SETTINGS.get("model_choice", "ensemble"))
        SETTINGS["model_choice"] = model_choice  # store last chosen selection
        # Run segmentation
        seg_res = segment_image(upload_path, out_dir="outputs", bin_dir="binarized", pad=4, min_area=30, save_crops_flag=False, return_overlay=True)
        centered_crops = seg_res["centered_crops"]   # list of 28x28 uint8 images
        boxes = seg_res["boxes"]
        # For each crop, prepare inputs and call model.predict_from_preprocessed
        pred_labels_for_overlay = []
        predictions = []
        # choose which models to run
        run_cnn = model_choice in ("ensemble", "cnn")
        run_seq = model_choice in ("ensemble", "seq")
        run_fnn = model_choice in ("ensemble", "fnn")
        for i, centered in enumerate(centered_crops):
            # prepare inputs
            fnn_in = prepare_for_fnn(centered)   # (1,784) numpy float32
            seq_in = prepare_for_seq(centered)   # (1,28,28) numpy float32
            cnn_in = prepare_for_cnn(centered)   # torch tensor (1,1,28,28)

            # call models
            crop_pred = {}
            if run_fnn and _models.get("fnn"):
                try:
                    lab, conf = _models["fnn"].predict_from_preprocessed(fnn_in)
                except Exception:
                    lab, conf = None, None
                crop_pred["fnn"] = {"label": lab, "conf": conf}
            else:
                crop_pred["fnn"] = None

            if run_seq and _models.get("seq"):
                try:
                    lab, conf = _models["seq"].predict_from_preprocessed(seq_in)
                except Exception:
                    lab, conf = None, None
                crop_pred["seq"] = {"label": lab, "conf": conf}
            else:
                crop_pred["seq"] = None

            if run_cnn and _models.get("cnn"):
                try:
                    lab, conf = _models["cnn"].predict_from_preprocessed(cnn_in)
                except Exception:
                    lab, conf = None, None
                crop_pred["cnn"] = {"label": lab, "conf": conf}
            else:
                crop_pred["cnn"] = None

            # pick overlay label priority / best performing model: cnn -> seq -> fnn
            chosen = crop_pred.get("cnn") or crop_pred.get("seq") or crop_pred.get("fnn")
            overlay_label = "" if chosen is None or chosen["label"] is None else str(chosen["label"])
            pred_labels_for_overlay.append(overlay_label)
            predictions.append(crop_pred)

        # Display image with predictions as labels
        color_img = cv2.imread(upload_path, cv2.IMREAD_COLOR)
        if color_img is None:
            # try to load grayscale and convert to color
            g = cv2.imread(upload_path, cv2.IMREAD_GRAYSCALE)
            color_img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        overlay_img = build_overlay_image(color_img, boxes, labels=pred_labels_for_overlay, pad=4)
        # Save overlay to static/overlays with unique name
        overlay_name = f"overlay_{uid}.png"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_name)
        cv2.imwrite(overlay_path, overlay_img)
        overlay_url = url_for("static", filename=f"overlays/{overlay_name}")

        return render_template("index.html", overlay_url=overlay_url, predictions=predictions, settings=SETTINGS)
    return render_template("index.html", overlay_url=overlay_url, predictions=predictions, settings=SETTINGS)


if __name__ == "__main__":
    # Run on host 0.0.0.0 port 8080 like before
    app.run(debug=True, host="0.0.0.0", port=8080)