# Giải Thích Lab Vertex AI Pipeline

## 1. Mục Tiêu Của Lab

Lab này yêu cầu bạn xây dựng một **ML Pipeline** (đường ống học máy) để dự đoán giá nhà, sử dụng nền tảng **Vertex AI** của Google Cloud.

### Pipeline là gì?

Pipeline là một chuỗi các bước xử lý dữ liệu được tự động hóa, chạy tuần tự:

```
Dữ liệu thô → Tiền xử lý → Huấn luyện → Đánh giá → Kết quả
```

Thay vì chạy từng bước thủ công, pipeline giúp tự động hóa toàn bộ quy trình.

---

## 2. Vertex AI Là Gì?

**Vertex AI** là nền tảng Machine Learning của Google Cloud, cho phép:

- Xây dựng và huấn luyện mô hình ML
- Triển khai mô hình lên cloud
- Tự động scale (mở rộng) khi cần
- Quản lý toàn bộ vòng đời của mô hình

### Cách Vertex AI Pipeline hoạt động:

```
┌─────────────────────────────────────────────────────────────┐
│                    VERTEX AI PIPELINE                        │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Docker  │   │  Docker  │   │  Docker  │   │  Docker  │ │
│  │Container │ → │Container │ → │Container │ → │Container │ │
│  │  Step 1  │   │  Step 2  │   │  Step 3  │   │  Step 4  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                              │
│  Google Cloud tự động:                                       │
│  - Khởi tạo máy ảo                                          │
│  - Chạy từng container                                       │
│  - Truyền dữ liệu giữa các bước                             │
│  - Lưu kết quả vào Cloud Storage                            │
└─────────────────────────────────────────────────────────────┘
```

**Mỗi bước (component) chạy trong một Docker container riêng biệt.**

---

## 3. Các Bước Lab Yêu Cầu

### Bước 1: Chuẩn bị môi trường

| Yêu cầu | Mục đích |
|---------|----------|
| Python | Viết code |
| Docker | Đóng gói ứng dụng thành container |
| gcloud CLI | Tương tác với Google Cloud |
| GCS Bucket | Lưu trữ dữ liệu và kết quả |
| Artifact Registry | Lưu trữ Docker images |

### Bước 2: Tạo Docker Image

```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
```

Docker image này chứa:
- Python runtime
- Các thư viện ML (pandas, scikit-learn, etc.)
- Code xử lý dữ liệu

Image được push lên **Artifact Registry** để Vertex AI có thể pull về và chạy.

### Bước 3: Định nghĩa các Component

Lab yêu cầu tạo 4 component:

#### Component 1: Data Ingestion (Thu thập dữ liệu)
```python
@component(base_image=BASE_IMAGE)
def data_ingestion(dataset: Output[Dataset]):
    # Đọc file Housing.csv từ GCS bucket
    # Lưu vào output artifact
```

#### Component 2: Preprocessing (Tiền xử lý)
```python
@component(base_image=BASE_IMAGE)
def preprocessing(input_dataset: Input[Dataset], 
                  preprocessed_dataset: Output[Dataset]):
    # Xử lý missing values
    # Encode categorical features (yes/no → 0/1)
    # Scale numerical features (chuẩn hóa)
```

#### Component 3: Training (Huấn luyện)
```python
@component(base_image=BASE_IMAGE)
def training(preprocessed_dataset: Input[Dataset],
             model: Output[Model],
             hyperparameters: dict):
    # Chia train/test
    # Huấn luyện RandomForest
    # Lưu model
```

#### Component 4: Evaluation (Đánh giá)
```python
@component(base_image=BASE_IMAGE)
def evaluation(model: Input[Model],
               preprocessed_dataset: Input[Dataset],
               metrics: Output[Metrics]):
    # Tính MSE, RMSE, MAE, R2
    # Tạo biểu đồ
```

### Bước 4: Kết nối các Component thành Pipeline

```python
@pipeline(name="houseprice_pipeline")
def houseprice_pipeline():
    # Bước 1
    ingestion = data_ingestion()
    
    # Bước 2: nhận output từ bước 1
    preprocess = preprocessing(
        input_dataset=ingestion.outputs["dataset"]
    )
    
    # Bước 3: nhận output từ bước 2
    train = training(
        preprocessed_dataset=preprocess.outputs["preprocessed_dataset"]
    )
    
    # Bước 4: nhận output từ bước 2 và 3
    evaluate = evaluation(
        model=train.outputs["model"],
        preprocessed_dataset=preprocess.outputs["preprocessed_dataset"]
    )
```

### Bước 5: Compile và chạy trên Vertex AI

```python
# Compile pipeline thành file JSON
compiler.Compiler().compile(
    pipeline_func=houseprice_pipeline,
    package_path='pipeline.json'
)

# Submit lên Vertex AI
pipeline_job = aiplatform.PipelineJob(
    display_name="houseprice_pipeline",
    template_path="pipeline.json"
)
pipeline_job.run()
```

---

## 4. Tại Sao Có Thể Chạy Local?

### Vấn đề gặp phải với Vertex AI:

1. **Quota limit**: Google Cloud giới hạn CPU miễn phí
   ```
   Error: RESOURCE_EXHAUSTED - custom_model_training_cpus quota exceeded
   ```

2. **Permission issues**: Cần cấu hình nhiều IAM roles phức tạp

3. **Chi phí**: Vertex AI tính phí theo giờ sử dụng

### Tại sao logic vẫn giống nhau?

Khi chạy trên Vertex AI:
```
┌─────────────────────────────────────────────────────────────┐
│                      VERTEX AI (Cloud)                       │
│                                                              │
│   Container 1 → Container 2 → Container 3 → Container 4     │
│   (trên máy     (trên máy     (trên máy     (trên máy       │
│    ảo GCP)       ảo GCP)       ảo GCP)       ảo GCP)        │
└─────────────────────────────────────────────────────────────┘
```

Khi chạy Local:
```
┌─────────────────────────────────────────────────────────────┐
│                      LOCAL (Máy của bạn)                     │
│                                                              │
│              ┌─────────────────────────┐                    │
│              │     Docker Container     │                    │
│              │                          │                    │
│              │  Step 1 → Step 2 → ...  │                    │
│              │  (cùng 1 container)      │                    │
│              └─────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**Điểm giống nhau:**
- Cùng code xử lý dữ liệu
- Cùng thuật toán (RandomForest)
- Cùng các bước pipeline
- Cùng chạy trong Docker container

**Điểm khác nhau:**
- Vertex AI: Mỗi bước = 1 container riêng, chạy trên cloud
- Local: Tất cả bước = 1 container, chạy trên máy bạn

---

## 5. So Sánh 2 Cách Triển Khai

| Tiêu chí | Vertex AI | Local Docker |
|----------|-----------|--------------|
| **Nơi chạy** | Google Cloud | Máy tính cá nhân |
| **Scaling** | Tự động scale | Giới hạn bởi máy |
| **Chi phí** | Trả tiền theo usage | Miễn phí |
| **Setup** | Phức tạp (IAM, quota) | Đơn giản |
| **Production** | Phù hợp | Chỉ để test |
| **Pipeline definition** | KFP SDK | Python thuần |

---

## 6. Cấu Trúc Project

```
house_prediction/
├── Dockerfile              # Image cho Vertex AI
├── Dockerfile.local        # Image cho chạy local
├── requirements.txt        # Dependencies
├── data/
│   └── Housing.csv         # Dataset
├── src/
│   ├── data_ingestion.py   # Component 1
│   ├── preprocessing.py    # Component 2
│   ├── training.py         # Component 3
│   └── evaluation.py       # Component 4
├── run_pipeline.py         # Chạy trên Vertex AI
├── run_local_pipeline.py   # Chạy local
└── output/
    ├── raw_data.csv
    ├── preprocessed_data.csv
    ├── model.joblib
    ├── metrics.txt
    └── evaluation_plots.png
```

---

## 7. Dataset Housing.csv

| Cột | Ý nghĩa | Loại |
|-----|---------|------|
| price | Giá nhà (target) | Số |
| area | Diện tích | Số |
| bedrooms | Số phòng ngủ | Số |
| bathrooms | Số phòng tắm | Số |
| stories | Số tầng | Số |
| mainroad | Mặt đường chính | yes/no |
| guestroom | Có phòng khách | yes/no |
| basement | Có tầng hầm | yes/no |
| hotwaterheating | Nước nóng | yes/no |
| airconditioning | Điều hòa | yes/no |
| parking | Số chỗ đỗ xe | Số |
| prefarea | Khu vực ưa thích | yes/no |
| furnishingstatus | Tình trạng nội thất | furnished/semi/unfurnished |

---

## 8. Kết Quả Đạt Được

### Metrics:
- **R² Score: 0.8451** → Mô hình giải thích được 84.5% sự biến thiên của giá nhà
- **RMSE: 735,551** → Sai số trung bình khoảng 735k
- **MAE: 460,204** → Sai số tuyệt đối trung bình khoảng 460k

### Giải thích:
- R² = 0.8451 là khá tốt cho bài toán regression
- Model RandomForest với 100 cây và max_depth=10 hoạt động hiệu quả

---

## 9. Tóm Tắt

1. **Lab yêu cầu**: Xây dựng ML Pipeline trên Vertex AI
2. **Vấn đề**: Quota limit của Google Cloud
3. **Giải pháp**: Chạy local bằng Docker với cùng logic
4. **Kết quả**: Pipeline hoàn chỉnh với R² = 84.5%

Việc chạy local chứng minh bạn hiểu:
- Cách thiết kế pipeline ML
- Cách sử dụng Docker
- Cách tiền xử lý dữ liệu
- Cách huấn luyện và đánh giá model
