# Backend - FastAPI + MongoDB + Qdrant

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run with auto-reload:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Folder Structure
```
app/
├── main.py          # FastAPI app entry point
├── api/             # API routes
├── core/            # Core config and utilities
├── models/          # Pydantic models
├── services/        # Business logic
└── db/              # Database connections
```
