const form = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const previewBox = document.getElementById("previewBox");
const previewImg = document.getElementById("previewImg");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const submitBtn = document.getElementById("submitBtn");

function fmt(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "Not available";
    }
    return Number(value).toFixed(digits);
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewBox.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files[0];
    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("known_label", document.getElementById("knownLabel").value);

    loading.classList.remove("hidden");
    result.classList.add("hidden");
    submitBtn.disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Prediction failed.");
        }

        const data = await response.json();
        renderResult(data);

    } catch (err) {
        alert(err.message);
    } finally {
        loading.classList.add("hidden");
        submitBtn.disabled = false;
    }
});

function renderResult(data) {
    result.classList.remove("hidden");

    const predLabel = document.getElementById("predLabel");
    predLabel.textContent = `Prediction: ${data.prediction}`;
    predLabel.className = data.prediction.toLowerCase() === "fake" ? "fake" : "real";

    document.getElementById("confidence").textContent = `Confidence: ${fmt(data.confidence, 3)}`;

    document.getElementById("realProb").textContent = fmt(data.real_probability, 4);
    document.getElementById("fakeProb").textContent = fmt(data.fake_probability, 4);

    document.getElementById("realBar").style.width = `${100 * data.real_probability}%`;
    document.getElementById("fakeBar").style.width = `${100 * data.fake_probability}%`;

    document.getElementById("temperature").textContent = fmt(data.temperature, 4);
    document.getElementById("rawLogits").textContent = `[${data.raw_logits.map(x => fmt(x, 4)).join(", ")}]`;
    document.getElementById("calLogits").textContent = `[${data.calibrated_logits.map(x => fmt(x, 4)).join(", ")}]`;

    document.getElementById("modelEce").textContent = fmt(data.model_ece, 4);
    document.getElementById("modelBrier").textContent = fmt(data.model_brier, 4);
    document.getElementById("singleBrier").textContent = fmt(data.single_image_brier, 4);
    document.getElementById("singleCalError").textContent = fmt(data.single_image_calibration_error, 4);

    document.getElementById("metricNote").textContent = `${data.ece_note} ${data.brier_note}`;

    document.getElementById("explanation").textContent = data.explanation;
    document.getElementById("regionText").textContent = data.region_text;
    document.getElementById("cueText").textContent = data.cue_phrases.join("; ");

    const cacheBust = `?t=${Date.now()}`;
    document.getElementById("inputImage").src = data.input_image_url + cacheBust;
    document.getElementById("gradcamImage").src = data.gradcam_url + cacheBust;
    document.getElementById("fftImage").src = data.fft_url + cacheBust;
    document.getElementById("combinedImage").src = data.combined_url + cacheBust;

    result.scrollIntoView({ behavior: "smooth" });
}
