# NetraEdge — System Architecture

## Overview

NetraEdge is an offline-first facial recognition and liveness detection system designed for zero-network environments. It runs entirely on-device using lightweight TFLite models, with optional sync to AWS when connectivity is restored.

## System Architecture

```mermaid
graph TB
    subgraph App["React Native App"]
        CAM["Camera Frame<br/>react-native-vision-camera v5"]
        FP["Frame Processor<br/>ML Kit Face Detection"]
        CORE["@netraedge/core<br/>Pure TypeScript"]
        SYNC["SyncManager<br/>AWS Upload + Purge"]
    end

    subgraph Pipeline["FacePipeline"]
        ENC["Encoder<br/>TFLite → 128-d embedding"]
        LIVE["LivenessOrchestrator"]
        STORE["EmbeddingStore<br/>SQLite"]
    end

    subgraph Liveness["Liveness Checks"]
        BLINK["BlinkDetector<br/>EAR (Eye Aspect Ratio)"]
        TEXTURE["TextureAnalyzer<br/>CNN real vs fake"]
        DEPTH["DepthEstimator<br/>3D face mesh variance"]
    end

    CAM --> FP
    FP --> CORE
    CORE --> Pipeline
    LIVE --> BLINK
    LIVE --> TEXTURE
    LIVE --> DEPTH
    CORE --> SYNC

    style App fill:#1e293b,stroke:#334155,color:#fff
    style Pipeline fill:#1e3a5f,stroke:#2563eb,color:#fff
    style Liveness fill:#1a332e,stroke:#22c55e,color:#fff
```

## Data Flow

### Enrollment Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CAM as Camera
    participant ML as ML Kit
    participant ENC as Encoder
    participant DB as SQLite
    participant AWS as AWS

    U->>CAM: Tap "Enroll"
    CAM->>ML: Stream frames
    ML-->>CAM: Face detected + landmarks
    CAM->>ENC: 10 face crops
    ENC-->>DB: 128-d embeddings (averaged)
    Note over DB: Stored locally
    U->>AWS: Network restored
    AWS-->>DB: Purge local data
```

### Verification Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CAM as Camera
    participant ML as ML Kit
    participant LIV as Liveness
    participant ENC as Encoder
    participant DB as EmbeddingStore

    U->>CAM: Tap "Verify"
    CAM->>ML: Stream frames
    ML-->>CAM: Face + 478 mesh points
    CAM->>LIV: Blink + Texture + Depth
    LIV-->>CAM: LIVE / SPOOF
    alt LIVE
        CAM->>ENC: Face crop
        ENC-->>DB: 128-d probe embedding
        DB-->>U: Match result + confidence
    else SPOOF
        U-->>U: Access denied
    end
```

## Model Architecture

### MobileFaceNet (Face Recognition)

```mermaid
graph LR
    A["Input<br/>112×112×3"] --> B["Conv 3×3<br/>64, stride 2"]
    B --> C["13× MobileBlock<br/>Depthwise Separable + SE"]
    C --> D["Global Avg Pool<br/>512"]
    D --> E["FC Layer<br/>128-d"]
    E --> F["L2 Normalize<br/>Unit vector"]

    style A fill:#1e293b,stroke:#334155,color:#fff
    style F fill:#1a332e,stroke:#22c55e,color:#fff
```

### Liveness Detection Pipeline

```mermaid
graph TB
    subgraph Input["Frame Input"]
        FACE["Face Crop 112×112"]
        MESH["478 Mesh Points"]
    end

    subgraph Blink["Blink Detection"]
        EAR["Eye Aspect Ratio"]
        HIST["EAR History"]
        BLINK_DET{"EAR < 0.21<br/>for 100ms?"}
    end

    subgraph Texture["Texture Analysis"]
        CNN["LivenessCNN"]
        PROB["Real / Print / Screen"]
        TEXT_DET{"Real > 0.80?"}
    end

    subgraph Depth["Depth Estimation"]
        Z["Z-coordinates"]
        VAR["Variance"]
        DEP_DET{"Variance > 0.25?"}
    end

    subgraph Decision["Final Decision"]
        PASS{"2+ checks<br/>passed?"}
        LIVE["LIVE ✓"]
        SPOOF["SPOOF ✗"]
    end

    FACE --> EAR
    MESH --> EAR
    EAR --> HIST
    HIST --> BLINK_DET
    FACE --> CNN
    CNN --> PROB
    PROB --> TEXT_DET
    MESH --> Z
    Z --> VAR
    VAR --> DEP_DET
    BLINK_DET --> PASS
    TEXT_DET --> PASS
    DEP_DET --> PASS
    PASS -->|Yes| LIVE
    PASS -->|No| SPOOF

    style Input fill:#1e293b,stroke:#334155,color:#fff
    style Blink fill:#1e3a5f,stroke:#2563eb,color:#fff
    style Texture fill:#3b1f4e,stroke:#a855f7,color:#fff
    style Depth fill:#1a332e,stroke:#22c55e,color:#fff
    style Decision fill:#4a2c1a,stroke:#f59e0b,color:#fff
```

## Models

| Model | Architecture | Input | Output | Size (INT8) |
|-------|-------------|-------|--------|-------------|
| Face Recognition | MobileFaceNet | 112×112×3 | 128-d vector | ~4.8MB |
| Liveness Detection | Custom CNN | 112×112×3 | 3-class softmax | ~2.8MB |
| Face Detection | ML Kit BlazeFace | Variable | Bounding box + landmarks | Built-in |

**Total model size: ~7.6MB** (well under 20MB target)

## Concurrency Model

```mermaid
graph LR
    subgraph MainThread["JS Thread"]
        UI["React UI"]
        HOOKS["Hooks"]
    end

    subgraph FrameThread["Frame Processor Thread"]
        FP["Frame Processing"]
        DET["Face Detection"]
    end

    subgraph NativeThread["Native Thread Pool"]
        TFL["TFLite Inference"]
        DB_OPS["SQLite Operations"]
    end

    subgraph SyncThread["Background Thread"]
        NET["Network Check"]
        UPLOAD["AWS Upload"]
        PURGE["Local Purge"]
    end

    UI --> HOOKS
    HOOKS --> FP
    FP --> DET
    DET --> TFL
    TFL --> DB_OPS
    HOOKS --> NET
    NET --> UPLOAD
    UPLOAD --> PURGE

    style MainThread fill:#1e293b,stroke:#334155,color:#fff
    style FrameThread fill:#1e3a5f,stroke:#2563eb,color:#fff
    style NativeThread fill:#1a332e,stroke:#22c55e,color:#fff
    style SyncThread fill:#4a2c1a,stroke:#f59e0b,color:#fff
```

## Error Handling

All errors use the `Result<T, E>` pattern — no exceptions in business logic.
Error codes are structured enums for programmatic handling.

```mermaid
graph LR
    A["Operation"] -->|Success| B["Result.ok"]
    A -->|Failure| C["Result.error"]
    B --> D["Use value"]
    C --> E["Handle error"]
    E --> F["Log + Recover"]

    style B fill:#1a332e,stroke:#22c55e,color:#fff
    style C fill:#4a1a1a,stroke:#ef4444,color:#fff
```

## Security

- All data stored locally on device
- No network calls without explicit user action
- Embeddings are L2-normalized (unit vectors)
- No raw images stored — only embeddings
- Sync uses TLS encryption
- Purge is immediate after successful upload
