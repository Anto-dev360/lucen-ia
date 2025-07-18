# 🚀 LucenAI – Sentiment Analysis on Crypto Tweets

## 🔍 Objective

This project aims to analyze **Bitcoin (BTC)** sentiment using public Twitter data. It leverages pre-trained NLP models—especially **DistilBERT**—through a full pipeline from preprocessing to deployment via FastAPI.


## 🧩 Key Pipeline Stages

- 📥 **Data ingestion and cleaning**
- 🔧 **Fine-tuning** of `DistilBERT` for binary sentiment classification (positive/negative)
- 📈 **Imbalance handling**, overfitting prevention, and ambiguous cases filtering
- 🧪 **Evaluation** on a dedicated test set
- 🌐 **Deployment** of a REST API with FastAPI


## 📁 Project Structure

```bash
lucen-ai/
├── data/                              # 📊 Dataset (raw)
│   └── BTC_Tweets_Sentiments.csv
├── docs/
│   ├── MVP_Sprints.txt                # Agile sprints 
│   └── project_pitch.pdf              # Presentation of the full lucen-ai project
├── notebooks/                         # 📓 Jupyter notebooks (EDA & experimentation)
│   ├── data/                          # Intermediate files used during experimentation
│   ├── logs/                          # Training logs (TensorBoard, etc.)
│   ├── models/                        # Temporary model outputs
│   ├── 01-preparation_dataset.ipynb      # Data cleaning and preparation
│   ├── 02-BERTweet_vs_distilBERT.ipynb   # Model comparison notebook
│   ├── 03-analyse_post_finetuning.ipynb  # Explore result of fine-tuned model
│   └── 04-compare_teacher_student.ipynb  # Compare models performance and RAM/Disk occupation
├─│ ─ lucenai/                           # 🧠 Core Python package
│   ├── training/                      # 🔧 Data pipeline and model logic
│   │   ├── preprocess.py              # Text preprocessing
│   │   ├── tokenizer.py               # Tokenizer loading/wrapping
│   │   ├── model.py                   # Model architecture & compilation
│   │   ├── evaluation.py              # Evaluation functions
│   │   ├── distillation.py            # Distillation of the fine-tuned model
│   │   ├── utils.py                   # Utility functions
│   │   └── models/distilbert_sentiment    # 🧠 Generated models
│   │       ├── checkpoint/best_model/     # Fine-tuned model weight
│   │       ├── tokenizer/                 # Fine-tuned model tokenizer
│   │       ├── logs/teacher/              # Fine-tuned model logs
│   │       ├── student_model/
│   │       │   ├── checkpoint/            # Student weights
│   │       │   └── tokenizer/             # Student tokenizer
│   │       └── logs/student/              # Student logs 
│   ├── api/                           # 🚀 FastAPI application
│   │   ├── predict.py                 # Inference endpoint
│   │   ├── utils.py
│   │   └── schemas.py                 # Pydantic schemas
│   ├── config/                        # ⚙️ Project configuration
│   │   └── settings.py
│   ├── frontend/                      # 🌐 Web frontend (basic UI)
│   │   ├── favicon.ico
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   └── tests/ 
│       └──  crypto_data.json          # Example BTC tweets
├── scripts/                           # 🛠️ CLI entrypoints
│   ├── train.py                       # Launch model training
│   ├── serve_and_tunnel.sh            # Launch FastAPI server
│   └── app.py                         # FastAPI entry point
├── README.md                          # 📖 Project overview and instructions
├── Dockerfile                         # Dockerfile for training environment (GPU-enabled)
├── Dockerfile.api                     # Dockerfile for inference via API
├── requirements.txt                   # Dependencies for model building
├── .gitignore
└── LICENSE
```


## 📒 Jupyter Notebooks

### 🧩 Key Pipeline Stages
- 📥 **Data ingestion and cleaning**
- ⚖️ **Model performance comparison**: `BERTweet` vs `DistilBERT`

The `notebooks/` directory includes exploratory and comparative notebooks:

- `preparation_dataset.ipynb`: Text cleaning, EDA, statistics
- `BERTweet_vs_distilBERT.ipynb`: Model benchmarking
- `notebooks/models/`: Model checkpoints generated during experimentation
- `notebooks/data/` : cleaned dataset for training, validation and test

📘 **Note:** The notebooks are written in French.

### 📈 Using TensorBoard for Training Visualization

To visualize training progress (loss, accuracy, learning curves), we use **TensorBoard** via a Keras callback.

To launch TensorBoard in your terminal:
```bash
tensorboard --logdir notebooks/logs
```

> Logs are automatically saved in `notebooks/logs/` with a timestamped subdirectory.


## 🧠 Pretrained Model

The sentiment analysis model is built on top of **`distilbert-base-uncased`**, a lightweight and fast version of BERT. It has been fine-tuned specifically for **binary classification** of Bitcoin-related tweet sentiments (`positive` vs `negative`).

### 🏗️ Architecture Summary

The model architecture defined in `model.py` follows this structure:

1. **DistilBERT Backbone**  
   - Loaded from Hugging Face via `TFAutoModel.from_pretrained("distilbert-base-uncased")`  
   - Outputs contextualized embeddings for each token

2. **[CLS] Token Pooling**  
   - The first token (index `[0]`) from the last hidden state is used as a global sentence representation

3. **Dropout Layer**  
   - Applied after the `[CLS]` token for regularization  
   - Dropout rate defined in `TRAINING_PARAMS.dropout_rate` (default: `0.3`)

4. **Dense Projection Layer**  
   - A fully-connected layer with `256` units and `ReLU` activation  
   - Adds non-linearity and improves learning capacity

5. **Binary Classification Head**  
   - A final `Dense` layer with `1` output unit and `sigmoid` activation  
   - Outputs a probability ∈ [0, 1] for the `positive` class

### 📦 Saved Model Format

After training, the model and tokenizer are saved in the following structure:

```
lucenai/models/distilbert_sentiment/
├── config.json                # Custom config (e.g., training parameters)
├── model.keras                # Full trained Keras model (TensorFlow format)
└── tokenizer/                 # Hugging Face tokenizer artifacts
    ├── vocab.txt
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

### ✅ Notes

- The model supports inference with TensorFlow's `.keras` format (recommended for Keras 3+).
- Tokenizer artifacts are compatible with Hugging Face Transformers and ensure consistent preprocessing for inference and deployment.


## 🧪 Distillation Support

This project supports knowledge distillation to compress the fine-tuned DistilBERT model into a smaller student model for faster inference and reduced memory usage.

### 🔧 How to use

To train a student model via distillation:

```bash
python scripts/train.py --distill
```

This will:
- Load the fine-tuned teacher model and tokenizer
- Train a smaller student model using soft predictions from the teacher
- Save the distilled model and tokenizer to the `models/distilbert_sentiment/student_model` directory
  The structure is the same as `models/distilbert_sentiment/` (including model, weight tokenizer)

To force retraining, including overwriting any previously saved model checkpoints:

```bash
python scripts/train.py --distill --force
```

The `--force` flag ensures the training starts from scratch, even if saved models exist.

### ✅ Notes

- Both teacher and student models use the same preprocessing pipeline.
- The student model can be used for inference with the same FastAPI endpoint.


## 🌐 API Usage (FastAPI)

The FastAPI server is containerized and exposes sentiment prediction as REST endpoints.

---

### 🔍 `POST /predict` — Single Tweet Sentiment

Predict the sentiment of a single tweet or short text.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Bitcoin is going to the moon 🚀"}'
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 0.9874
}
```

- `label`: Predicted sentiment (`"positive"`, `"negative"`, or `"invalid"`).
- `confidence`: Confidence score of the prediction (0.0–1.0).

---

### 📁 `POST /analyze` — Analyze JSON File of Tweets

Upload a JSON file containing a list of tweets for batch sentiment analysis.

**File format:**

```json
[
  { "id": 1, "text": "Bitcoin is breaking new highs again! 🚀 #BTC" },
  { "id": 2, "text": "I'm not convinced about crypto anymore. So unstable... 😓" },
  { "id": 3, "text": "Huge potential in blockchain tech, especially Bitcoin!" },
  { "id": 4, "text": "BTC just crashed 5% in one hour. That's scary." }
]
```

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
     -F "file=@crypto_data.json"
```

**Response:**
```json
{
  "positive": 0.65,
  "negative": 0.35,
  "total": 20
}
```

---

### 🩺 `GET /health` — API Health Check

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🖥️ Frontend Interface

LucenAI also includes a built-in frontend interface.

- You can **submit a single tweet** to analyze its sentiment.
- Or **upload a `.json` file** with a list of tweet objects.

**Accepted file format:**

```json
[
  { "id": 1, "text": "Example tweet here..." },
  { "id": 2, "text": "Another one." }
]
```

The interface provides real-time feedback and visualizes the overall sentiment distribution.


## 🚀 Run Locally with Tunnel

To make the FastAPI server publicly accessible via a tunnel (e.g., for frontend testing or mobile integration), a helper script `serve_and_tunnel.sh` is provided.

### Prerequisites

1. Create a free account on [ngrok.com](https://ngrok.com/).
2. Retrieve your authentication token.
3. Create a `.env` file at the root of the project with the following:

```
NGROK_AUTH_TOKEN=your_token_here
```

### Launch the API with tunnel

```bash
./scripts/serve_and_tunnel.sh
```

This will:
- Start the FastAPI backend (`uvicorn`)
- Expose it via an ngrok tunnel
- Display the public URL you can use to interact with the API


## 🐳 Docker

This project provides Docker images for:

- 📦 Training the DistilBERT model
- 🌐 Serving predictions via a FastAPI + Ngrok-powered API

### 🔧 Build the training image

```bash
docker build -f Dockerfile -t lucen-ia-train .
```

Train with GPU:

```bash
docker run --rm --gpus all \
  -v $(pwd)/lucenai/models:/app/lucenai/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/lucenai/config:/app/lucenai/config \
  lucen-ia-train
```

Train with CPU:

```bash
docker run --rm \
  -v $(pwd)/lucenai/models:/app/lucenai/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/lucenai/config:/app/lucenai/config \
  lucen-ia-train
```

### 🌐 Build and run the API

```bash
docker build -f Dockerfile.api -t sentiment-api .
```

Run with public Ngrok tunnel (requires .env):

```bash
docker run --rm --env-file .env -v $(pwd)/lucenai/models:/app/lucenai/models -p 8000:8000 sentiment-api
```

Sample request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bitcoin is amazing"}'
```

#### 📄 .env file required
Create a .env file with:
```env
NGROK_AUTH_TOKEN=your_token_here
```

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


## 📬 Contact

- **Author**: Anthony Morin  
- **License**: MIT

---

© 2025 – Anthony Morin. All code is open-sourced under the MIT License.