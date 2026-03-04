from flask import Flask, render_template_string, send_file
import io
from PIL import Image

import torch
import torchvision.utils as vutils

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "generator_final.pth"
LATENT_DIM = 512
DEVICE = "cpu"

# --- LAZY MODEL LOADING ---
generator = None

def load_gan():
    global generator
    if generator is None:
        from model import Generator

        model = Generator().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        generator = model

    return generator

# --- HTML TEMPLATE ---
HTML_PAGE = '''
<!DOCTYPE html>
<html>
    <head><title>GAN Image Generator</title></head>
    <body style="text-align: center; font-family: sans-serif; padding-top: 50px;">
        <h1>AI Image Generator</h1>
        <div>
            <img src="/generate_image?{{ v }}" id="gan-img" style="width: 256px; border: 5px solid #eee; border-radius: 10px;">
        </div>
        <br>
        <button onclick="location.reload()" style="padding: 10px 20px; cursor: pointer;">Generate New Image</button>
    </body>
</html>
'''

@app.route('/')
def index():
    import time
    return render_template_string(HTML_PAGE, v=time.time())

@app.route('/generate_image')
def generate_image():

    net = load_gan()

    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        fake = net(z)

    img_io = io.BytesIO()
    vutils.save_image(fake, img_io, format='PNG', normalize=True)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    # IMPORTANT: disable reloader to prevent double import of torch
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
