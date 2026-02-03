from flask import Flask, render_template_string, send_file
import torch
import io
from PIL import Image
import torchvision.utils as vutils
from model import Generator # This imports your architecture

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "generator_final.pth" # The file you downloaded from Drive
LATENT_DIM = 512                   # Must match what you used in training
DEVICE = torch.device("cpu")       # Running on local CPU

# --- LOAD MODEL ---
def load_gan():
    model = Generator().to(DEVICE)
    # map_location='cpu' is vital for running without a GPU!
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

generator = load_gan()

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
    with torch.no_grad():
        # Create random noise (z)
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        # Generate the image
        fake = generator(z)
        
    # Convert the tensor to a viewable image (PNG)
    img_io = io.BytesIO()
    # Normalize=True is important to bring pixels from [-1, 1] to [0, 1]
    vutils.save_image(fake, img_io, format='PNG', normalize=True)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)