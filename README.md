# 🌊 Earthquake-Tsunami Prediction System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> A production-ready machine learning platform for real-time tsunami risk assessment from seismic data

## 📖 Overview

The Earthquake-Tsunami Prediction System is an end-to-end ML pipeline that predicts tsunami occurrence probability within minutes of earthquake detection. Built with enterprise MLOps best practices, it processes real-time USGS seismic data to provide actionable early warnings with 85% accuracy.

### Key Features

- **Real-time Predictions**: < 5 second inference latency for immediate risk assessment
- **High Accuracy**: 85% overall accuracy with optimized recall for tsunami detection
- **Production Ready**: Full CI/CD pipeline with automated testing and deployment
- **MLOps Integration**: Model versioning with DVC + MLflow, feature stores, and drift detection
- **Explainable AI**: SHAP-based model interpretability for transparent decision-making
- **RESTful API**: FastAPI service with OpenAPI specification and authentication

## 🎯 Quick Start

### Prerequisites

- Python 3.11 or 3.12
- 8GB RAM minimum
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/ayoub1999hannachi/earthquake-tsunami.git
cd earthquake-tsunami

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### Configuration

```bash
# Copy environment template
cp .env.template .env

# Configure required variables
# USGS_API_KEY=your_api_key
# MLFLOW_TRACKING_URI=http://localhost:5000
# POSTGRES_DSN=postgresql://user:pass@localhost:5432/tsunami_db
```

### Running the Pipeline

```bash
# Full pipeline execution
make pipeline

# Or run individual steps
make ingest-data      # Fetch latest USGS data
make validate-data    # Run data quality checks
make preprocess       # Feature engineering
make train            # Train model
make evaluate         # Generate evaluation report
make serve           # Start API server
```

## 📊 Dataset

### Data Source

- **Primary**: [USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/) (1970-2024)
- **Update Frequency**: Hourly incremental ingestion
- **Total Records**: ~500,000 earthquake events
- **Class Distribution**: 97.3% no tsunami, 2.7% tsunami

### Features

| Feature | Type | Description | Transformation |
|---------|------|-------------|----------------|
| `latitude` | float64 | Geographic coordinate (WGS84) | RobustScaler |
| `longitude` | float64 | Geographic coordinate (WGS84) | RobustScaler |
| `depth` | float64 | Focal depth (km) | Yeo-Johnson |
| `mag` | float64 | Moment magnitude (Mw) | StandardScaler |
| `sig` | int64 | Significance score (0-1000) | MinMaxScaler |
| `mmi` | float64 | Modified Mercalli Intensity | Log1p |
| `cdi` | float64 | Community Decimal Intensity | Log1p |
| `tsunamigenic_ratio` | float64 | Engineered: `mag / depth` | QuantileTransformer |
| `distance_to_coast` | float64 | Engineered: Haversine distance | RobustScaler |

### Data Splits

- **Train**: 1970-2018 (70%)
- **Validation**: 2019-2021 (15%)
- **Test**: 2022-2024 (15%)
- **Strategy**: GroupKFold by tectonic plate boundaries to prevent spatial leakage



## 📈 Model Performance

### XGBoost Classifier (Production Model)

| Metric | Class 0 (No Tsunami) | Class 1 (Tsunami) | Weighted Avg |
|--------|---------------------|-------------------|--------------|
| **Precision** | 0.87 | 0.68 | 0.86 |
| **Recall** | 0.98 | 0.31 | 0.96 |
| **F1-Score** | 0.92 | 0.43 | 0.90 |
| **ROC AUC** | - | 0.82 | - |

### Feature Importance

1. **Magnitude (mag)**: 32% - Most critical predictor
2. **Depth**: 21% - Shallow earthquakes more likely to cause tsunamis
3. **Distance to Coast**: 18% - Proximity affects tsunami generation
4. **Significance (sig)**: 12% - Overall event severity
5. **MMI**: 9% - Ground shaking intensity




## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run integration tests
make test-integration

# Run linting and type checking
make lint
make type-check
```

**Coverage Targets:**
- Unit Tests: > 90%
- Integration Tests: > 80%
- Overall: 95%


## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Security scan
make security-scan
```

**Tools:**
- **Formatting**: Black, isort
- **Linting**: Ruff (150+ rules)
- **Type Checking**: MyPy (strict mode)
- **Security**: Bandit, Safety





### Prometheus Metrics

- `model_prediction_latency_seconds`
- `model_accuracy_score`
- `api_request_duration_seconds`
- `data_drift_detected`

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests
4. Run quality checks: `make lint test`
5. Commit using conventional commits: `feat: add new feature`
6. Push and open a Pull Request

### Code Review Process

- 2 approvals required for merge
- All tests must pass
- Code coverage must not decrease
- Documentation must be updated

## 📝 Citation

If you use this work in research, please cite:

```bibtex
@software{hannachi2024tsunami,
  title={Earthquake-Tsunami Prediction System},
  author={Hannachi, Ayoub},
  year={2024},
  url={https://github.com/ayoub1999hannachi/earthquake-tsunami},
  version={2.1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **USGS** for providing comprehensive seismic data
- **MLflow Community** for model registry support
- **PyData Global** for review and feedback
- **Tunisian AI Society** for compute resources

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ayoub1999hannachi/earthquake-tsunami/issues)
- **Email**: ahannachi193@gmail.com
- **LinkedIn**: [Ayoub Hannachi](https://www.linkedin.com/in/ayoub-hannachi-0727931b0/)

## 🗺️ Roadmap

- [ ] Ensemble stacking with LightGBM + CatBoost
- [ ] Real-time waveform analysis integration
- [ ] Transformer-based spatial encoding
- [ ] Kubernetes-native deployment
- [ ] Model uncertainty quantification
- [x] XGBoost with class imbalance handling
- [x] SHAP explainability integration
- [x] FastAPI production service
- [x] Prometheus monitoring

---

<div align="center">
  <strong>Built with ❤️ for safer communities</strong>
  <br>
  <sub>⭐ Star this repo if you find it useful!</sub>
</div>
