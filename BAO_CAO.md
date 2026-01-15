# BÁO CÁO LAB: VERTEX AI PIPELINE

**Họ và tên:** [Điền tên]  
**MSSV:** [Điền MSSV]  
**Ngày:** 15/01/2026

---

## 1. Mô tả bài toán

Xây dựng ML Pipeline dự đoán giá nhà sử dụng dataset Housing.csv (545 mẫu, 13 features).

---

## 2. Pipeline Flow

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│                  │      │                  │      │                  │      │                  │
│  DATA INGESTION  │ ──→  │  PREPROCESSING   │ ──→  │    TRAINING      │ ──→  │   EVALUATION     │
│                  │      │                  │      │                  │      │                  │
│  • Load CSV      │      │  • Encode        │      │  • Train/Test    │      │  • Calculate     │
│  • 545 rows      │      │    categorical   │      │    split 80/20   │      │    MSE, R2       │
│  • 13 columns    │      │  • Scale         │      │  • RandomForest  │      │  • Visualize     │
│                  │      │    numerical     │      │    100 trees     │      │                  │
└──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘
        ↓                         ↓                         ↓                         ↓
   raw_data.csv           preprocessed_data.csv        model.joblib            metrics.txt
                                                                            evaluation_plots.png
```

---

## 3. Minh chứng Docker

### 3.1. Docker Image đã build

```bash
$ docker images | grep house-prediction
REPOSITORY               TAG       IMAGE ID       SIZE
house-prediction-local   latest    7f860884802f   1.2GB
```

### 3.2. Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir pandas scikit-learn matplotlib joblib numpy
COPY run_local_pipeline.py /app/
COPY data /app/data
CMD ["python", "run_local_pipeline.py"]
```

### 3.3. Lệnh chạy

```bash
docker build -f Dockerfile.local -t house-prediction-local .
docker run --name house-pipeline -v $(pwd)/output:/app/output house-prediction-local
```

---

## 4. Kết quả

| Metric | Giá trị |
|--------|---------|
| **R² Score** | **0.8451** (84.5%) |
| RMSE | 735,551.63 |
| MAE | 460,204.79 |

---

## 5. Screenshots

*(Chèn screenshots vào đây)*

1. **Docker build/run:**
   
2. **Output pipeline:**

3. **Visualization:**

---

## 6. Ghi chú

Do quota limit của Google Cloud (RESOURCE_EXHAUSTED), pipeline được triển khai trên Docker local với logic tương đương Vertex AI.
