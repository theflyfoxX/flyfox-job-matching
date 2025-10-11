# 🦊 Flyfox Job Matching Engine

A machine learning-driven job matching engine that connects applicants with the most suitable job opportunities using natural language processing, text embeddings, and structured data features.

## 🚀 Key Features

- **🔄 Data Ingestion**: Load applicant profiles, job descriptions, and labeled pairs
- **🧠 Feature Engineering**: Combine text-based embeddings and structured metadata (location, experience, skills)
- **🎯 Model Training**: Train predictive models using logistic regression, XGBoost, and LightGBM
- **📈 Prediction**: Rank jobs for applicants or find best-fit candidates for positions
- **🌐 API Integration**: Serve predictions via FastAPI (optional)


## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/theflyfoxX/flyfox-job-matching.git
cd flyfox-job-matching
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv wrangler-env

# Activate on Windows
./wrangler-env/Scripts/activate

# Activate on macOS/Linux
source wrangler-env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
flyfox/
├── config.yaml                 # Central configuration
├── predict.py                  # Main prediction script
├── test.py                     # Test runner
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
│
├── data/
│   ├── raw/                   # Raw CSV files
│   │   ├── Combined_Jobs_Final.csv
│   │   ├── Experience.csv
│   │   ├── Positions_Of_Interest.csv
│   │   └── labeled_applicant_job_pairs.csv
│   ├── interim/               # Processed intermediate data
│   └── features/              # Final feature matrices
│
├── embeddings/
│   ├── jobs/                  # Job embeddings (.npy)
│   └── applicants/            # Applicant embeddings (.npy)
│
├── features/
│   └── build_features.py      # Feature engineering scripts
│
├── src/
│   ├── features/              # Feature builders
│   ├── io/                    # File I/O utilities
│   ├── models/                # Model training & evaluation
│   ├── prep/                  # Data preparation helpers
│   ├── preprocessing/         # Text/vector preprocessing
│   ├── utils/                 # Shared utilities
│   └── api/                   # FastAPI application
│
└── docker/                    # Docker configurations
```

## 🚀 Usage

### Generate Predictions

Run the main prediction script:

```bash
python predict.py
```

### Run Tests

Execute the test suite:

```bash
python test.py
```

## 📊 Data Requirements

### Required Files

Place the following files in `data/raw/`:

- `Combined_Jobs_Final.csv` - Job postings with descriptions and metadata
- `Experience.csv` - Applicant work experience records
- `Positions_Of_Interest.csv` - Applicant job preferences
- `labeled_applicant_job_pairs.csv` - Training data with applicant-job matches

### Required Embeddings

Pre-generated embeddings must be stored as `.npy` dictionary files:

- `embeddings/jobs/embeddings_dict.npy` - Job description embeddings
- `embeddings/applicants/embeddings_dict.npy` - Applicant profile embeddings

## 📦 Dependencies

### Core Libraries

- **Data Processing**: `pandas`, `numpy`, `pyarrow`, `fastparquet`
- **Machine Learning**: `scikit-learn`, `lightgbm`, `xgboost`
- **NLP & Embeddings**: `sentence-transformers`, `transformers`, `torch`, `gensim`
- **API**: `fastapi`, `uvicorn`
- **Database**: `psycopg2-binary` (PostgreSQL support)

See `requirements.txt` for complete list with versions.


## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
python test.py

# Run specific test modules
pytest tests/test_features.py
pytest tests/test_models.py
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model parameters
- Feature engineering settings
- API configuration
- File paths and data sources

## 📝 Notes

- Embeddings must be generated before running predictions
- Ensure all required data files are present in `data/raw/`
- The virtual environment (`wrangler-env/`) is excluded from version control
- GPU acceleration recommended for embedding generation and model training

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 👤 Author

**Ali Rassas**

- 📧 Email: [rassasali01@gmail.com](mailto:rassasali01@gmail.com)
- 🔗 GitHub: [@theflyfoxX](https://github.com/theflyfoxX)


---

