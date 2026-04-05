# malaria-detection-cv
This repository hosts a deep learning pipeline to detect malaria parasites in thin blood smear images.

## Setup

Create and activate a Python 3.12 environment, then install dependencies:

```bash
python -m venv .venv312
.\\.venv312\\Scripts\\activate
pip install -r requirements.txt
```

## Launch the Inference App

Run the Streamlit web interface for malaria detection inference:

```bash
python -m streamlit run src/app.py
```

The app opens at `http://localhost:8501` (or the next free port, for example `8502`).
The UI loads the first trained model it finds in `models/`, typically `models/best_model.h5`.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Make sure these files are committed:
	- `src/app.py`
	- `src/image_utils.py`
	- `requirements.txt`
	- `runtime.txt`
	- `models/best_model.h5`
3. Go to `https://share.streamlit.io/` and click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to:

```text
src/app.py
```

6. Click **Deploy**.

If `models/best_model.h5` is large and not suitable for GitHub, host it externally and download it at startup.

## Train + Evaluate

This repository does not currently include a checked-in `src/run_pipeline.py` entrypoint.
Use the individual scripts instead:

```bash
python src/train.py
python src/evaluate.py
```

Optional examples with custom settings:

```bash
python src/train.py --epochs 20 --batch-size 32 --image-size 128 --model-out models/best_model.h5
python src/evaluate.py --model-path models/best_model.h5 --batch-size 32 --image-size 128
```

## Run Individual Steps

Train only:

```bash
python src/train.py
```

Evaluate only:

```bash
python src/evaluate.py
```
