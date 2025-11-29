# NBA BetIQ API Documentation

**Base URL (local)**: `http://localhost:8000`

**Base URL (production)**: `https://your-app.herokuapp.com` (or Railway/Render)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
   - [GET /](#get-)
   - [GET /health](#get-health)
   - [POST /predict](#post-predict)
   - [POST /predict/spread](#post-predictspread)
   - [POST /predict/ou](#post-predictou)
   - [POST /ev](#post-ev)
   - [POST /predict_ev](#post-predict_ev)
3. [Error Handling](#error-handling)
4. [Rate Limits](#rate-limits)

---

## Authentication

Currently, no authentication required. API is open for testing and demo purposes.

**Future**: API keys for production use.

---

## Endpoints

### GET /

**Description**: Root endpoint providing API information.

**Response**:
```json
{
  "name": "NBA BetIQ API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

---

### GET /health

**Description**: Health check endpoint to verify API status and model availability.

**Response**:
```json
{
  "status": "healthy",
  "message": "NBA BetIQ API is running. Models loaded: 3/3"
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `200 OK` (degraded): Service running but models not loaded

---

### POST /predict

**Description**: Predict home team win probability (moneyline).

**Request Body**:
```json
{
  "features": {
    "HOME_FG_PCT": 0.462,
    "AWAY_FG_PCT": 0.448,
    "HOME_FG3_PCT": 0.365,
    "AWAY_FG3_PCT": 0.342,
    "HOME_REB": 45,
    "AWAY_REB": 42,
    "HOME_AST": 24,
    "AWAY_AST": 22,
    "HOME_TOV": 12,
    "AWAY_TOV": 14,
    "home_spread": -5.5,
    "pinnacle_total": 218.5
  }
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.6234,
  "prob_home_win": 0.6234,
  "model_version": "1.0.0"
}
```

**Field Descriptions**:
- `prediction`: Binary prediction (1 = home wins, 0 = away wins)
- `probability`: Win probability (0-1)
- `prob_home_win`: Same as probability (for clarity)
- `model_version`: Version of model used

**Status Codes**:
- `200 OK`: Successful prediction
- `422 Unprocessable Entity`: Invalid input (missing features)
- `503 Service Unavailable`: Model not loaded

---

### POST /predict/spread

**Description**: Predict probability that home team covers the point spread.

**Request Body**:
```json
{
  "features": {
    "HOME_FG_PCT": 0.462,
    "AWAY_FG_PCT": 0.448,
    "home_spread": -5.5,
    ...
  }
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.5823,
  "prob_home_covers": 0.5823,
  "model_version": "1.0.0"
}
```

---

### POST /predict/ou

**Description**: Predict over/under probability.

**Request Body**:
```json
{
  "features": {
    "HOME_FG_PCT": 0.462,
    "AWAY_FG_PCT": 0.448,
    "pinnacle_total": 218.5,
    ...
  }
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.5412,
  "prob_over": 0.5412,
  "model_version": "1.0.0"
}
```

---

### POST /ev

**Description**: Calculate expected value (EV) for a bet given win probability and odds.

**Request Body**:
```json
{
  "probability": 0.58,
  "odds": -110,
  "stake": 100.0
}
```

**Field Descriptions**:
- `probability`: Win probability from model (0-1)
- `odds`: American odds (e.g., -110, +150)
- `stake`: Bet amount in dollars (default: 100)

**Response**:
```json
{
  "ev": 5.41,
  "ev_percent": 5.41,
  "implied_prob": 0.5238,
  "model_prob": 0.58,
  "edge": 0.0562,
  "recommendation": "BET"
}
```

**Field Descriptions**:
- `ev`: Expected value in dollars
- `ev_percent`: EV as percentage of stake
- `implied_prob`: Implied probability from odds (accounting for vig)
- `model_prob`: Your model's predicted probability
- `edge`: Advantage over bookmaker (model_prob - implied_prob)
- `recommendation`: "BET" if EV > 0, else "PASS"

**Example Calculation**:

If model predicts 58% chance of winning, but odds imply only 52.38%:
- **Edge**: 5.62%
- **EV**: $5.41 per $100 bet
- **Recommendation**: BET (positive expected value)

**Status Codes**:
- `200 OK`: Successful calculation
- `422 Unprocessable Entity`: Invalid input

---

### POST /predict_ev

**Description**: Combined endpoint - makes prediction AND calculates EV in one call.

**Request Body**:
```json
{
  "features": {
    "HOME_FG_PCT": 0.462,
    "AWAY_FG_PCT": 0.448,
    "home_spread": -5.5,
    ...
  },
  "odds": -110,
  "stake": 100.0,
  "bet_type": "home_win"
}
```

**Field Descriptions**:
- `features`: Same as `/predict`
- `odds`: American odds
- `stake`: Bet amount
- `bet_type`: One of `"home_win"`, `"spread"`, or `"ou"`

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.6234,
  "ev": 18.92,
  "ev_percent": 18.92,
  "implied_prob": 0.5238,
  "edge": 0.0996,
  "recommendation": "BET",
  "model_version": "1.0.0"
}
```

**Status Codes**:
- `200 OK`: Successful prediction + EV
- `400 Bad Request`: Invalid `bet_type`
- `503 Service Unavailable`: Model not loaded

---

## Error Handling

All errors return JSON with `error` and `detail` fields:

```json
{
  "error": "Model not available",
  "detail": "home_win_xgb.pkl not found"
}
```

**Common Error Codes**:
- `400`: Bad request (invalid input)
- `422`: Validation error (Pydantic)
- `500`: Internal server error
- `503`: Service unavailable (models not loaded)

---

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to:
- View all endpoints
- Test requests directly
- See request/response schemas
- Download OpenAPI spec

---

## Example Workflow

### 1. Check API Health
```bash
curl http://localhost:8000/health
```

### 2. Get Prediction + EV
```bash
curl -X POST http://localhost:8000/predict_ev \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "HOME_FG_PCT": 0.462,
      "AWAY_FG_PCT": 0.448,
      "HOME_FG3_PCT": 0.365,
      "AWAY_FG3_PCT": 0.342,
      "home_spread": -5.5,
      "pinnacle_total": 218.5
    },
    "odds": -110,
    "stake": 100,
    "bet_type": "home_win"
  }'
```

### 3. Interpret Results

If response shows:
```json
{
  "probability": 0.58,
  "ev": 5.41,
  "edge": 0.0562,
  "recommendation": "BET"
}
```

**Interpretation**:
- Model estimates 58% chance home team wins
- Betting $100 has expected profit of **$5.41**
- You have a **5.62% edge** over the bookmaker
- **Recommendation**: Place the bet

---

## Rate Limits

**Current**: No rate limits

**Future**:
- 100 requests/minute per IP
- 1,000 requests/day per API key

---

## Deployment URLs

| Environment | URL |
|-------------|-----|
| Local | `http://localhost:8000` |
| Heroku | `https://nba-betiq.herokuapp.com` |
| Railway | `https://nba-betiq.up.railway.app` |
| Render | `https://nba-betiq.onrender.com` |

---

## SDK / Client Libraries

**Python Example**:
```python
import requests

url = "http://localhost:8000/predict_ev"
payload = {
    "features": {"HOME_FG_PCT": 0.462, ...},
    "odds": -110,
    "stake": 100,
    "bet_type": "home_win"
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Win Probability: {data['probability']:.2%}")
print(f"Expected Value: ${data['ev']:.2f}")
print(f"Recommendation: {data['recommendation']}")
```

---

## Support

For issues or questions:
- GitHub Issues: `https://github.com/yourusername/nba-betiq/issues`
- Email: `your.email@example.com`

---

## Changelog

### v1.0.0 (2024)
- Initial release
- 3 prediction models (home_win, spread, ou)
- EV calculation endpoint
- Combined predict + EV endpoint
- Health check & monitoring
