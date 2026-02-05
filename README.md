# PEPI: Positron Emission Projection Imaging

PEPI is a Python module designed for the Positron Emission Projection Imaging (PEPI) technique where concentration field measurements are obtained from gamma ray data. It provides tools for voxelization, normalization, filtering, and visualization of 3D and 2D data.

<div align="center">
    <img src ="https://raw.githubusercontent.com/robh96/PEPI/main/images/cover_img.jpg" alt="graphical abstract of the PEPI method." width="800">
    <br>
    <em>PEPI works by injecting a positron emitting radioisotope into a system surrounded by gamma-ray detectors, which correlate the detection frequency to the spatio-temporal distribution of radiolabelled material.</em>
</div>


This implementation of PEPI is based on work by Hart-Villamil et al (2024).
> Hart-Villamil, R., Sykes, J., Ingram, A., Windows-Yule, C. R., & Gupta, S. K. (2024). Positron Emission Projection Imaging: A technique for concentration field measurements in opaque industrial systems. Particuology, 94, 1-15. https://doi.org/10.1016/j.partic.2024.07.009

## Features

- **Voxelization**: Discretizes LoRs into 3D voxel grids.
- **Normalization**: Normalizes voxel and pixel data based on the 99th percentile.
- **Temporal Filtering**: Reduces noise using a dynamic moving median filter.
- **Visualization**: Generates heatmaps, histograms, and other visualizations.
- **Serialization**: Saves and loads processed data using `dill`.

## Installation

Clone the repository:

```bash
git clone https://github.com/Robh96/PEPI.git
cd PEPI
```

Ensure you have the required dependencies installed. You can install them using `pip`:

```bash
pip install numpy scipy matplotlib opencv-python dill
```

## Usage

### Example Pipeline

```python
from pepi import Helpers, Pepi, Norm, TemporalFilter, PostProcess
import pept

file_path = "path/to/your/data.da01_01"
save_path = "path/to/save/output"
time_skip = 20  # seconds
time_length = 15  # seconds
time_slice = 0.5  # seconds
overlap = 0.5  # fraction
cell_size = 4  # mm

line_data = Helpers.read_lines(file_path, time_skip=time_skip, time_length=time_length)
line_data.sample_size = pept.TimeWindow(time_slice * 1000)
line_data.overlap = pept.TimeWindow(overlap * time_slice * 1000)

pipeline = pept.Pipeline([
    Pepi(line_data, cell_size, xlim=(200, 440), ylim=(200, 380)),
    Norm(),
    TemporalFilter(),
    PostProcess(save_path=save_path, heatmap=True, histogram=True)
])
pipeline.fit(line_data, max_workers=1)
```

`Pepi()` is the core algorithm that transforms gamma ray coincidences into activity fields.
`Norm()` normalises the activity fields to the 99th percentile maximum in the time series.
`TemporalFilter()` is a median average filter that ensures temporal coherence between frames.
`PostProcess` produces colourmap plots and histograms and saves them to file.

## License

This project is licensed under the GNU GPL V3 License. See the `LICENSE` file for details.

## Citing
If you use this algorithm for research purposes we kindly ask that you cite the following:
> Hart-Villamil, R., Sykes, J., Ingram, A., Windows-Yule, C. R., & Gupta, S. K. (2024). Positron Emission Projection Imaging: A technique for concentration field measurements in opaque industrial systems. Particuology, 94, 1-15. https://doi.org/10.1016/j.partic.2024.07.009

>Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329. https://doi.org/10.1063/1.5129251
