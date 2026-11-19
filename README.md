# Fabrisketch

> **Zero-Credit, AI-Powered PNG-to-PES Machine Embroidery Digitizer & Interactive Studio**

Fabrisketch is a local web application that converts raster images (`.png`) into machine-ready Brother embroidery files (`.pes`) with full control over stitch quality, hoop sizing, and thread colors.

## Features
- **Deterministic Computational Geometry:** Uses [Shapely](https://github.com/shapely/shapely) and [pyembroidery](https://github.com/EmbroidePy/pyembroidery) to convert closed shapes into valid underlay, Tatami fill, and satin border needle stitches.
- **Brother Palette Matching:** Perceptual CIE LAB $K$-Means clustering mapped directly to the [Brother Embroidery Thread Color List](https://support.brother.com/g/s/hf/mobileapp_info/artspira/thread/name_pdf/en.pdf).
- **Procedural Zero-PES Training:** Train region segmentation and thread-angle prediction models without needing proprietary `.pes` training datasets.
- **Interactive Studio:** Live browser canvas for tweaking thread density, moving nodes, and drawing custom stitches by hand.

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Fabrisketch.git
cd Fabrisketch

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Step 1 Verification Test
Verify that your local geometry engine and `.pes` compiler work cleanly:
```bash
python backend/step1_first_pes_test.py
```
This generates:
- `1_input_logo.png`: Sample 2-color test badge.
- `2_detected_corners.png`: Visual contour and corner detection map.
- `3_visual_stitch_preview.png`: 2D simulation showing every thread and needle drop.
- `4_output_embroidery.pes`: Physical Brother embroidery machine file.

### 3. Generate Procedural Training Data
Generate synthetic training pairs for the U-Net with zero `.pes` files needed:
```bash
python ml/zero_pes_dataset.py --output_dir ./procedural_embroidery_dataset --samples 500
```
