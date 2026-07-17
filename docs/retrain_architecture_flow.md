# Retrain sonrası mimari akış

Bu diyagram, Streamlit'teki **Start Retrain** butonuna basıldıktan sonra gerçekleşen gerçek uygulama akışını gösterir.

```mermaid
flowchart TB
    A["Admin · Streamlit<br/>Start Retrain<br/><code>model · epochs · optimizer · regularization params</code>"]
    B["FastAPI<br/><code>POST /retrain</code><br/>Admin authorization"]
    C{"Başka retrain<br/>çalışıyor mu?"}
    BUSY["HTTP 409<br/>DVC repo-wide lock korunur"]
    D["Airflow REST API<br/><code>rakuten_model_training</code><br/>DAG run oluştur"]

    A --> B --> C
    C -- "Evet" --> BUSY
    C -- "Hayır" --> D

    subgraph TRAINING["AIRFLOW — TRAINING DAG"]
        direction LR
        E["1 · prepare_data<br/><code>dvc repro prepare_splits</code>"]
        F["2 · launch_trainer<br/>DockerOperator"]
        G["Ephemeral trainer<br/><code>epochs · batch · LR overrides<br/>dvc repro train_image|train_text<br/>--force --single-item</code>"]
        H["Yeni component çıktısı<br/>best checkpoint + run_metadata + dvc.lock"]
        I["3 · trigger_promotion<br/>Promotion sonucunu bekler"]
        E --> F --> G --> H --> I
    end

    D --> E
    G -. "metrics, params, plots" .-> TRACKING
    G -- "başarısız / OOM" --> FAILED

    subgraph PROMOTION["AIRFLOW — PROMOTION DAG"]
        direction TB
        J["Ephemeral promotion container"]
        K["Asset doğrulama<br/>Image + Text checkpoints<br/>CamemBERT + tokenizer<br/>label map + category map"]
        L["<code>dvc push dvc.yaml:train_&lt;model&gt;</code>"]
        M["Tam multimodal serving bundle"]
        N["MLflow pyfunc candidate<br/><code>rakuten-multimodal-classifier</code>"]
        O{"Champion gate<br/>candidate Macro-F1<br/>>= champion + min_gain?"}
        J --> K --> L --> M --> N --> O
    end

    I --> J
    L --> DVCREMOTE
    N --> TRACKING

    subgraph STORES["KALICI VERİ KATMANI"]
        direction LR
        DVCREMOTE[("MinIO · dvc-data<br/>DVC data/checkpoints")]
        TRACKING["MLflow Tracking + Registry"]
        PG[("PostgreSQL · mlflow<br/>runs, metrics, versions, aliases")]
        ARTIFACTS[("MinIO · mlflow-artifacts<br/>plots + pyfunc bundle")]
        TRACKING --> PG
        TRACKING --> ARTIFACTS
    end

    O -- "Hayır" --> REJECT["Candidate rejected<br/>Registry'de audit için kalır<br/>Champion/API değişmez"]
    O -- "Evet" --> CHAMPION["champion alias<br/>Yeni version'a taşınır"]
    CHAMPION --> DOWNLOAD["Registry'den champion bundle indir"]
    DOWNLOAD --> DEPLOY["Atomic deploy<br/><code>data/rakuten_streamlit_predictor/</code><br/>+ deployment_manifest.json"]
    DEPLOY --> SERVE["FastAPI sonraki prediction'da<br/>manifest değişimini algılar<br/>bundle'ı hot-reload eder"]
    SERVE --> UI["Streamlit tahminleri<br/>yeni champion modeli kullanır"]

    FAILED["Training DAG failed<br/>Promotion çalışmaz<br/>Serving modeli değişmez"]
    A -. "GET /retrain/{job_id} · status polling" .-> D
```

## Kritik ayrımlar

- Eğitim run'ı MLflow'a metrik ve grafik yazar; bu tek başına modeli production yapmaz.
- `dvc push`, seçilen component checkpoint'ini `dvc-data` remote'una arşivler; champion seçmez.
- Promotion her iki component'i, tokenizer'ı ve mapping dosyalarını tek bundle içinde Registry'ye candidate olarak kaydeder.
- Candidate gate'i geçemezse Registry version audit için kalır; mevcut `champion` ve serving dizini değişmez.
- Gate geçerse `champion` alias yeni version'a taşınır, bundle Registry'den tekrar indirilir ve serving dizini atomik olarak değiştirilir.
- FastAPI `deployment_manifest.json` içindeki version/run değişimini sonraki tahmin isteğinde algılayarak modeli yeniden yükler; API container restart gerekmez.
