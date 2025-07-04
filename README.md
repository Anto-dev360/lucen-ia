# 🚀 LucenAI – Sentiment Analysis on Crypto Tweets

## 🔍 Objective

This project aims to analyze **Bitcoin (BTC)** sentiment using public Twitter data. It leverages pre-trained NLP models—especially **DistilBERT**—through a full pipeline from preprocessing to deployment via FastAPI.

---

## 🧩 Key Pipeline Stages

- 📥 **Data ingestion and cleaning**
- 🔧 **Fine-tuning** of `DistilBERT` for binary sentiment classification (positive/negative)
- 📈 **Imbalance handling**, overfitting prevention, and ambiguous cases filtering
- 🧪 **Evaluation** on a dedicated test set
- 🌐 **Deployment** of a REST API with FastAPI

---

## 📁 Project Structure

```bash
lucen-ai/
├── models/                            # 🧠 Fine-tuned models (saved checkpoints)
│   └── distilbert_sentiment/
│       ├── config.json
│       ├── tokenizer/
│       └── tf_model.h5
├── data/                              # 📊 Dataset (raw)
│   └── BTC_Tweets_Sentiments.csv
├── notebooks/                         # 📓 Jupyter notebooks (EDA & experimentation)
│   ├── data/                          # Intermediate files used during experimentation
│   ├── logs/                          # Training logs (TensorBoard, etc.)
│   ├── models/                        # Temporary model outputs
│   ├── preparation_dataset.ipynb      # Data cleaning and preparation
│   └── BERTweet_vs_distilBERT.ipynb   # Model comparison notebook
├── lucenai/                           # 🧠 Core Python package
│   ├── train/                         # 🔧 Data pipeline and model logic
│   │   ├── preprocess.py              # Text preprocessing
│   │   ├── tokenizer.py               # Tokenizer loading/wrapping
│   │   ├── model.py                   # Model architecture & compilation
│   │   ├── test.py                    # Evaluation functions
│   │   └── utils.py                   # Utility functions
│   ├── api/                           # 🚀 FastAPI application
│   │   ├── predict.py                 # Inference endpoint
│   │   └── schemas.py                 # Pydantic schemas
│   ├── config/                        # ⚙️ Project configuration
│   │   └── settings.py
│   └── frontend/                      # 🌐 Web frontend (basic UI)
│       ├── index.html
│       ├── app.js
│       ├── crypta_data.json
│       └── style.css
├── scripts/                           # 🛠️ CLI entrypoints
│   ├── train.py                       # Launch model training
│   └── app.py                         # Launch FastAPI server
├── README.md                          # 📖 Project overview and instructions
├── Docker.train                       # Dockerfile for training environment
├── Docker.api                         # Dockerfile for API environment
├── requirements_CPU.txt               # Minimal dependencies
├── requirements_GPU.txt               # Dev dependencies for GPU runs
├── .gitignore
└── LICENSE
```

---

## 📒 Jupyter Notebooks

### 🧩 Key Pipeline Stages
- 📥 **Data ingestion and cleaning**
- ⚖️ **Model performance comparison**: `BERTweet` vs `DistilBERT`

The `notebooks/` directory includes exploratory and comparative notebooks:

- `preparation_dataset.ipynb`: Text cleaning, EDA, statistics
- `BERTweet_vs_distilBERT.ipynb`: Model benchmarking
- `notebooks/models/`: Model checkpoints generated during experimentation
- `notebooks/data/` : cleaned dataset for training, validation and test

### 📈 Using TensorBoard for Training Visualization

To visualize training progress (loss, accuracy, learning curves), we use **TensorBoard** via a Keras callback.

To launch TensorBoard in your terminal:
```bash
tensorboard --logdir notebooks/logs
```

> Logs are automatically saved in `notebooks/logs/` with a timestamped subdirectory.

---

## 🧠 Pretrained Model

The integrated model is a fine-tuned `distilbert-base-uncased` for binary sentiment classification. It is stored under:

```
models/distilbert_sentiment/
├── config.json
├── tokenizer/
└── tf_model.h5
```

---

## 🌐 API Usage (FastAPI)

The FastAPI server is containerized and exposes sentiment prediction as a REST endpoint.

### 🔍 Example `POST /predict` request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Bitcoin is going to the moon 🚀"}'
```

Response:
```json
{
  "label": "positive",
  "confidence": 0.987
}
```

### 🔁 API health check (`GET /health`)

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok"
}
```

---

## 🐳 Docker

This project uses two separate Dockerfiles to manage training and deployment environments independently:

- `Docker.train`: for training the DistilBERT model on GPU (L4, Lightning AI-compatible)
- `Docker.api`: for serving the FastAPI + frontend on CPU

---

### 🐳 1. Build Docker Images

From the project root:

```bash
# Build training image (GPU)
docker build -f Docker.train -t lucen-train .

# Build API image (CPU)
docker build -f Docker.api -t lucen-api .
```

---

### 🎓 2. Run Training (GPU-enabled)

```bash
docker run --gpus all lucen-train
```

> This will:
> - Load and clean BTC tweet data
> - Tokenize with DistilBERT
> - Fine-tune the model
> - Save it to `/app/lucenai/models/`

---

### 🌐 3. Serve the API (CPU)

```bash
docker run -p 8000:8000 lucen-api
```

Then visit:

- 🖥️ Frontend: [http://localhost:8000](http://localhost:8000)
- 📚 Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📚 Libraries Used

This project is built upon a robust and modern machine learning stack:

| Library | Description |
|--------|-------------|
| 🤗 [Transformers](https://huggingface.co/docs/transformers) | Pretrained NLP models and tokenizers (e.g., DistilBERT) |
| 🧠 [TensorFlow](https://www.tensorflow.org/) | Deep learning framework for model fine-tuning and inference |
| 📊 [Scikit-learn](https://scikit-learn.org/) | Utilities for splitting, metrics, and preprocessing |
| 🐍 [Pandas](https://pandas.pydata.org/) | Data handling and manipulation of tweet datasets |
| 📈 [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) | Visualization of model performance and data distributions |
| 🌐 [FastAPI](https://fastapi.tiangolo.com/) | High-performance web framework for exposing the model as an API |
| 🚀 [Uvicorn](https://www.uvicorn.org/) | ASGI server to run the FastAPI backend |

---

## 📬 Contact

- **Author**: Anthony Morin  
- **License**: MIT

---

© 2025 – Anthony Morin. All code is open-sourced under the MIT License.