# Fabrisketch

> **Zero-Credit, AI-Powered PNG-to-PES Machine Embroidery Digitizer & Interactive Studio**

Fabrisketch is a local web application that converts raster images (`.png`) into machine-ready Brother embroidery files (`.pes`) with full control over stitch quality, hoop sizing, and thread colors.

## Features
- **Deterministic Computational Geometry:** Uses Shapely and pyembroidery to convert closed shapes into valid underlay, Tatami fill, and satin border needle stitches.
- **Brother Palette Matching:** Perceptual CIE LAB K-Means clustering mapped directly to Brother embroidery threads.
- **Procedural Zero-PES Training:** Train region segmentation and thread-angle prediction models without needing proprietary `.pes` training datasets.
- **Interactive Studio:** Live browser canvas for tweaking thread density, moving nodes, and drawing custom stitches by hand.