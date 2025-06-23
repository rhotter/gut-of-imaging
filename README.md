# Gut of Imaging: Interactive Signal Processing Visualizations

A project to visualize and explain fundamental concepts in signal processing, linear algebra, and information theory through interactive web-based visualizations.

## Overview

This repository contains visualizations designed to help understand the mathematical foundations of image processing and signal processing. The visualizations focus on the following concepts:

1. Linear models as matrices of Gaussians multiplying vectors
2. Fourier transforms and their relationship to linear operations
3. Eigenvalues and their connection to Fourier analysis
4. Orthogonality properties of Fourier transforms and noise
5. Signal-to-Noise Ratio and channel coding in the frequency domain

## Project Structure

- `web/` - Contains the interactive web visualizations built with React

## Quick Start

To run the visualizations:

```bash
cd web
npm install
npm run dev
```

Then open your browser to http://localhost:5173

## Visualizations

### 1. Linear Model Visualization

Shows how a linear model can be visualized as a matrix of Gaussians multiplying a vector, and how it smooths data. This helps in understanding convolution operations and kernel-based methods in signal processing.

### 2. Fourier Transform Visualization

Demonstrates how linear operations in the spatial or time domain correspond to multiplications in the Fourier domain. This is fundamental to understanding frequency-domain processing.

### 3. Eigenvalues Visualization

Illustrates how eigenvalues and eigenvectors relate to linear transformations and how they connect to Fourier analysis. This visualization helps explain why the Fourier basis is special for certain operations.

### 4. Orthogonality and Noise

Shows how the orthogonality property of Fourier transforms affects noise distribution. This visualization demonstrates how noise gets converted to similar noise in Fourier space.

### 5. SNR and Channel Coding

Applies Signal-to-Noise Ratio concepts and channel coding arguments to independent Fourier components. This helps in understanding optimal filtering and information transmission concepts.

## Technical Details

The web visualizations are built using:

- React.js with Vite for the frontend
- Plotly.js for interactive charts
- MathJS for mathematical operations
- KaTeX for mathematical equation rendering

See the README in the `web/` directory for more detailed information on running and developing the visualization app.

## License

This project is licensed under the MIT License - see the LICENSE file for details.