"""
Flask App — Conditional GAN Image Generator
============================================
Loads generator_final.pth + class_map.json from MODEL_DIR.
User selects an animal class from a dropdown → model generates an image.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import io
import json
import torch
import torchvision.utils as vutils
from flask import Flask, render_template_string, send_file, request, jsonify

app = Flask(__name__)

# ── Configuration ────────────────────────────
MODEL_DIR  = "."                      # Folder containing .pth and .json files
MODEL_PATH = os.path.join(MODEL_DIR, "generator_final.pth")
MAP_PATH   = os.path.join(MODEL_DIR, "class_map.json")
Z_DIM      = 128
DEVICE     = "cpu"

# ── Lazy-loaded globals ───────────────────────
_generator     = None
_class_to_idx  = None
_idx_to_class  = None


def load_class_map():
    global _class_to_idx, _idx_to_class
    if _class_to_idx is None:
        with open(MAP_PATH, "r") as f:
            _class_to_idx = json.load(f)
        _idx_to_class = {v: k for k, v in _class_to_idx.items()}
    return _class_to_idx, _idx_to_class


def load_generator():
    global _generator
    if _generator is None:
        from model import Generator
        class_to_idx, _ = load_class_map()
        num_classes = len(class_to_idx)

        G = Generator(z_dim=Z_DIM, num_classes=num_classes).to(DEVICE)
        G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        G.eval()
        _generator = G
        print(f"✅ Generator loaded ({num_classes} classes)")
    return _generator


# ── HTML Template ────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Animal GAN Generator</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #f0f4f8;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 48px 16px;
            min-height: 100vh;
        }
        h1 { font-size: 2rem; color: #1a202c; margin-bottom: 8px; }
        p.sub { color: #718096; margin-bottom: 32px; }

        .card {
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            width: 100%;
            max-width: 420px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }

        select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
            background: #f7fafc;
            cursor: pointer;
        }

        button {
            width: 100%;
            padding: 12px;
            background: #4f46e5;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #4338ca; }
        button:disabled { background: #a5b4fc; cursor: not-allowed; }

        #image-wrap {
            width: 256px;
            height: 256px;
            border-radius: 12px;
            overflow: hidden;
            background: #edf2f7;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #gan-img {
            width: 256px;
            height: 256px;
            object-fit: cover;
            display: block;
        }

        #status {
            font-size: 0.875rem;
            color: #718096;
            min-height: 20px;
        }
    </style>
</head>
<body>
    <h1>🐾 Animal GAN</h1>
    <p class="sub">Select an animal and generate a unique AI image</p>

    <div class="card">
        <div id="image-wrap">
            <img id="gan-img" src="" alt="Generated image will appear here">
        </div>

        <select id="animal-select">
            {% for cls in classes %}
            <option value="{{ cls }}">{{ cls.replace('_', ' ').title() }}</option>
            {% endfor %}
        </select>

        <button id="gen-btn" onclick="generate()">✨ Generate Image</button>
        <span id="status">Choose an animal and click Generate</span>
    </div>

    <script>
        function generate() {
            const cls   = document.getElementById('animal-select').value;
            const btn   = document.getElementById('gen-btn');
            const img   = document.getElementById('gan-img');
            const status = document.getElementById('status');

            btn.disabled = true;
            status.textContent = 'Generating...';

            // Cache-bust with timestamp so browser always fetches fresh
            const url = `/generate_image?class=${encodeURIComponent(cls)}&t=${Date.now()}`;

            const newImg = new Image();
            newImg.onload = () => {
                img.src = newImg.src;
                btn.disabled = false;
                status.textContent = `Generated: ${cls.replace('_', ' ')}`;
            };
            newImg.onerror = () => {
                btn.disabled = false;
                status.textContent = '❌ Error generating image. Try again.';
            };
            newImg.src = url;
        }
    </script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────

@app.route('/')
def index():
    class_to_idx, _ = load_class_map()
    classes = sorted(class_to_idx.keys())
    return render_template_string(HTML, classes=classes)


@app.route('/generate_image')
def generate_image():
    class_name = request.args.get('class', None)

    class_to_idx, idx_to_class = load_class_map()

    # Fall back to a random class if none specified or invalid
    if class_name not in class_to_idx:
        import random
        class_name = random.choice(list(class_to_idx.keys()))

    label_idx = class_to_idx[class_name]

    G = load_generator()

    with torch.no_grad():
        z      = torch.randn(1, Z_DIM).to(DEVICE)
        labels = torch.tensor([label_idx]).to(DEVICE)
        fake   = G(z, labels)

    img_io = io.BytesIO()
    vutils.save_image(fake, img_io, format='PNG', normalize=True)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


@app.route('/classes')
def list_classes():
    """Utility endpoint — returns all available animal classes as JSON."""
    class_to_idx, _ = load_class_map()
    return jsonify(sorted(class_to_idx.keys()))


# ── Entry Point ──────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)