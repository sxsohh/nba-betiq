# NBA BetIQ - System Architecture

## Overview

NBA BetIQ is an end-to-end machine learning system for NBA betting analysis. It demonstrates the full ML pipeline from data ingestion to production API deployment.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Vegas Odds   │  │ Game Scores  │  │  Shot Data   │      │
│  │  (raw/)      │  │   (raw/)     │  │   (raw/)     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      ETL PIPELINE                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. ingest.py       → Load & clean raw data         │   │
│  │  2. clean_merge.py  → Merge datasets                │   │
│  │  3. features.py     → Feature engineering           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    games_master_2018_19.csv
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML TRAINING                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  training.py  → Train 3 models:                     │   │
│  │                 • Home Win (Moneyline)               │   │
│  │                 • Spread Cover                       │   │
│  │                 • Over/Under                         │   │
│  │                                                       │   │
│  │  Models:                                             │   │
│  │    - XGBoost Classifier                             │   │
│  │    - Calibrated Logistic Regression                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   ml/models/   │
                    │   • *.pkl      │
                    └────────┬───────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODEL EVALUATION                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  evaluation.py  → Metrics:                          │   │
│  │                   • ROC AUC, Accuracy               │   │
│  │                   • Calibration (ECE)               │   │
│  │                   • Brier Score, Log Loss           │   │
│  │                                                       │   │
│  │  visuals.py     → Generate plots:                   │   │
│  │                   • ROC curves                       │   │
│  │                   • Calibration curves               │   │
│  │                   • Feature importance               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION API (FastAPI)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                          │   │
│  │    GET  /health          → Health check             │   │
│  │    POST /predict         → Home win probability     │   │
│  │    POST /predict/spread  → Spread probability       │   │
│  │    POST /predict/ou      → O/U probability          │   │
│  │    POST /ev              → Calculate EV             │   │
│  │    POST /predict_ev      → Predict + EV combined    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   Deployment   │
                    │   (Docker)     │
                    └────────────────┘
```

---

## Component Breakdown

### 1. Data Layer

**Location**: `data/`

- **Raw Data** (`data/raw/`):
  - Vegas betting odds (regular + playoffs)
  - Game scores and box scores
  - Shot-level data

- **Processed Data** (`data/processed/`):
  - Merged, cleaned datasets
  - Feature-engineered master dataset

**Key Features**:
- Historical betting lines (moneyline, spread, total)
- Public betting percentages
- Team statistics (FG%, rebounds, assists, etc.)
- Implied probabilities from odds
- Vig (house edge) calculations

---

### 2. ETL Pipeline

**Location**: `etl/`

#### `ingest.py`
- Loads raw CSV/TXT files
- Initial data type conversion
- Filters 2018-19 season
- Outputs intermediate files

#### `clean_merge.py`
- Pivots team-level data to game-level
- Merges odds + scores on GAME_ID
- Maps team names to abbreviations

#### `features.py`
- Engineers betting features (implied probs, vig)
- Creates target variables (home_win, spread_cover, ou_over_win)
- Computes advanced metrics (pace, efficiency)
- Saves to database (SQLite)

---

### 3. Machine Learning

**Location**: `ml/`

#### `training.py`
- Trains 3 classification models (home_win, spread, ou)
- Algorithms:
  - **XGBoost**: Gradient boosting (300 trees, depth=5)
  - **Calibrated LR**: Logistic regression + Platt scaling
- Cross-validation for calibration
- Saves models as `.pkl` files

#### `evaluation.py`
- Comprehensive metrics:
  - Accuracy, Precision, Recall, F1
  - ROC AUC, Log Loss, Brier Score
  - Expected Calibration Error (ECE)
- Model comparison (XGB vs LR)
- Outputs evaluation report

#### `visuals.py`
- Visualization functions:
  - ROC curves
  - Calibration plots
  - Public betting distributions
  - Feature importance
  - EV over time

---

### 4. Backend API

**Location**: `backend/`

**Framework**: FastAPI

#### `api.py`
- RESTful endpoints for predictions
- Model loading on startup (lifespan events)
- Error handling & logging
- CORS configuration

#### `schemas.py`
- Pydantic models for request/response validation
- Type safety & automatic docs

#### `utils.py`
- Odds conversion utilities
- EV calculation
- Kelly Criterion
- ROI metrics

#### `config.py`
- Environment-based configuration
- Model paths, database paths
- CORS settings

---

### 5. Deployment

#### Docker
- **Dockerfile**: Multi-stage build for production
- **docker-compose.yml**: Orchestrates API + dependencies

#### Cloud Deployment
- **Procfile**: For Heroku/Railway deployment
- Environment variables for configuration
- Health check endpoints for monitoring

---

## Data Flow

1. **Raw Data** → ETL Pipeline → **Master Dataset**
2. **Master Dataset** → ML Training → **Trained Models**
3. **Trained Models** → API Startup → **Loaded in Memory**
4. **API Request** → Model Inference → **Prediction + EV** → Response

---

## Key Design Decisions

### Why XGBoost + Calibrated LR?

- **XGBoost**: Best for prediction accuracy (handles non-linear patterns)
- **Calibrated LR**: Best for probability calibration (critical for EV calculations)
- **Ensemble Approach**: Use both and compare

### Why Calibration Matters?

For betting, we need **accurate probabilities**, not just predictions. A poorly calibrated model might say "70% confident" but only win 55% of the time. This destroys EV calculations.

### Why SQLite?

- Lightweight, serverless
- Perfect for single-season dataset (~1,200 games)
- Easy to deploy (no external DB needed)

### Why FastAPI?

- Modern, fast, async support
- Automatic OpenAPI docs
- Type safety with Pydantic
- Production-ready (Uvicorn ASGI server)

---

## Scalability Considerations

### Current Limitations
- Single season (2018-19) only
- Models trained on ~1,000 games
- Static models (no online learning)

### Future Enhancements
1. **Multi-Season Training**: Expand to 5+ seasons
2. **Live Odds Integration**: Real-time line movement tracking
3. **Model Monitoring**: Drift detection, auto-retraining
4. **Advanced Features**: Player injuries, lineups, rest patterns
5. **Distributed Training**: Hyperparameter tuning with Ray/Dask

---

## Testing Strategy

### Unit Tests (`tests/`)
- `test_ev.py`: EV calculation logic
- `test_calibration.py`: Calibration metrics
- `test_model_loading.py`: Model file integrity
- `test_healthcheck.py`: API endpoints

### Integration Tests
- Full ETL pipeline execution
- End-to-end prediction workflow

### CI/CD
- GitHub Actions: Auto-run tests on push
- Docker build validation
- Code quality checks (black, flake8)

---

## Security

- No credentials in code (`.env` for secrets)
- API rate limiting (future)
- Input validation via Pydantic
- CORS restrictions

---

## Monitoring & Observability

- Health check endpoint (`/health`)
- Structured logging (INFO/ERROR levels)
- Model versioning in responses
- Future: Prometheus metrics, Grafana dashboards

---

## Tech Stack Summary

| Layer | Technologies |
|-------|-------------|
| **Data** | Pandas, NumPy |
| **ML** | scikit-learn, XGBoost |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Database** | SQLite |
| **Visualization** | Matplotlib, Seaborn |
| **Testing** | Pytest |
| **Deployment** | Docker, Heroku/Railway |
| **CI/CD** | GitHub Actions |

---

## File Structure

```
nba-betiq/
├── backend/           # FastAPI application
├── data/              # Raw & processed data
├── etl/               # Data pipeline scripts
├── ml/                # ML training & evaluation
│   ├── models/        # Saved model files
│   ├── training.py
│   ├── evaluation.py
│   └── visuals.py
├── notebooks/         # Jupyter exploration
├── tests/             # Unit & integration tests
├── visuals/           # Generated plots
├── docs/              # Documentation
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition
└── docker-compose.yml # Multi-container setup
```

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Expected Value in Sports Betting](https://en.wikipedia.org/wiki/Expected_value)
- [Platt Scaling for Calibration](https://en.wikipedia.org/wiki/Platt_scaling)
