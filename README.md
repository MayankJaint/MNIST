# MNIST Binary Dataset Loader
### Streamlit :https://mayank-mnist.streamlit.app/

A high-performance, vectorized Python parser for reading binary IDX-formatted datasets (such as MNIST) into NumPy arrays without relying on heavy third-party framework dependencies.

## Features

- **Fast Binary I/O:** Utilizes C-optimized `numpy.frombuffer` for near-instant memory loading of raw binary data.
- **Dynamic Dimension Extraction:** Automatically parses image dimensions (`rows`, `cols`, `size`) directly from binary header magic bytes rather than hardcoding.
- **Zero Python Overhead:** Eliminates standard Python `for` loops and intermediate lists for data slicing.

## File Format Specs

The parser processes standard IDX byte stream files:
- **Image Bytes:** Magic Number `2051` — Header contains `[magic (4B), size (4B), rows (4B), cols (4B)]`
- **Label Bytes:** Magic Number `2049` — Header contains `[magic (4B), size (4B)]`

## Prerequisites

- Python  3.12 not higher
- requirements.txt
- set paths for dataset

```bash
uv pip install -r requirements.txt
- download pretraned model here : https://drive.google.com/file/d/1mq9FCmyXfwy2BS99v0kZTnZI23ogmNRR/view?usp=sharing
