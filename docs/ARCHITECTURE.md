# NetraEdge — System Architecture

## Overview

NetraEdge is a layered monorepo architecture designed for offline face recognition and liveness detection. The system follows clean architecture principles with dependency injection, functional error handling, and clear separation of concerns.

```mermaid
graph TB
    subgraph "Presentation Layer"
        APP[Demo App]
        SCREENS[Screens]
        COMPONENTS[Components]
        HOOKS[Hooks]
    end
    
    subgraph "Bridge Layer"
        RN["@netraedge/react-native"]
        NATIVE[Native Modules]
        CONTEXT[App Context]
    end
    
    subgraph "Core Layer"
        CORE["@netraedge/core"]
        PIPELINE[FacePipeline]
        ENCODER[Encoder]
        LIVENESS[LivenessOrchestrator]
        STORE[EmbeddingStore]
        SYNC[SyncManager]
    end
    
    subgraph "Platform Layer"
        ANDROID[Kotlin TFLite]
        IOS[Swift TFLite]
        MODELS[TFLite Models]
    end
    
    APP --> SCREENS
    SCREENS --> HOOKS
    HOOKS --> RN
    RN --> CORE
    CORE --> NATIVE
    NATIVE --> ANDROID
    NATIVE --> IOS
    ANDROID --> MODELS
    IOS --> MODELS
```

## Data Flow — Enrollment

```mermaid
sequenceDiagram
    participant U as User
    participant S as EnrollScreen
    participant C as FaceCamera
    participant FD as Face Detection
    participant P as FacePipeline
    participant E as Encoder (TFLite)
    participant ST as EmbeddingStore
    participant SQ as SyncQueue
    
    U->>S: Enter User ID
    S->>C: Start camera
    C->>FD: Detect faces (ML Kit)
    FD->>S: Face detected + landmarks
    
    loop 10 frames
        S->>S: Capture frame
        S->>S: Extract mesh points
    end
    
    S->>P: enroll(userId, frames, meshPoints)
    
    loop For each frame
        P->>E: encode(faceData)
        E->>E: TFLite inference (112x112x3 → 128-d)
        E->>E: L2 normalize
        E-->>P: embedding
    end
    
    P->>P: Average embeddings
    P->>ST: Store(userId, avgEmbedding)
    P->>SQ: Queue for sync
    P-->>S: EnrollmentResult
    S->>U: Success!
```

## Data Flow — Verification

```mermaid
sequenceDiagram
    participant U as User
    participant V as VerifyScreen
    participant C as FaceCamera
    participant FD as Face Detection
    participant P as FacePipeline
    participant L as LivenessOrchestrator
    participant E as Encoder (TFLite)
    participant ST as EmbeddingStore
    
    U->>V: Open camera
    V->>C: Start camera
    C->>FD: Detect faces
    FD->>V: Face detected
    
    V->>L: processFrame(faceData, meshPoints)
    
    Note over L: 3 Independent Checks
    
    par Check 1: Blink
        L->>L: EAR calculation
    and Check 2: Texture
        L->>E: LivenessCNN inference
    and Check 3: Depth
        L->>L: Z-variance analysis
    end
    
    L-->>V: LivenessResult
    
    alt Liveness PASSED (2+ checks)
        V->>P: verify(faceData, meshPoints)
        P->>E: encode(faceData)
        E-->>P: 128-d embedding
        P->>ST: identify(embedding, threshold)
        ST-->>P: Best match
        P-->>V: VerificationResult
        V->>U: Identity Verified
    else Liveness FAILED
        V->>U: SPOOF DETECTED
    end
```

## Module Dependencies

```mermaid
graph LR
    subgraph "@netraedge/core"
        T[Types]
        C[Config]
        E[Embedding]
        L[Liveness]
        P[Pipeline]
        S[Sync]
    end
    
    T --> C
    T --> E
    T --> L
    E --> P
    L --> P
    S --> P
    
    subgraph "@netraedge/react-native"
        H[Hooks]
        CO[Components]
        SC[Screens]
        ST[Storage]
        SY[Sync]
        CTX[Context]
    end
    
    P --> H
    H --> SC
    CO --> SC
    CTX --> SC
    ST --> SY
```

## Native Module Architecture

### Android (Kotlin)

```mermaid
graph TB
    subgraph "React Native Bridge"
        NM[NetraEdgeModule.kt]
        NP[NetraEdgePackage.kt]
    end
    
    subgraph "TFLite Runtime"
        TI[Interpreter]
        RF[Recognition Model]
        LF[Liveness Model]
    end
    
    NM --> NP
    NP --> TI
    TI --> RF
    TI --> LF
    
    RF --> |"112x112x3 → 128-d"| OUT1[Embedding]
    LF --> |"112x112x3 → 3-class"| OUT2[Probabilities]
```

### iOS (Swift)

```mermaid
graph TB
    subgraph "React Native Bridge"
        SW[NetraEdgeModule.swift]
        OC[NetraEdgeModule.m]
    end
    
    subgraph "TensorFlow Lite"
        TI[Interpreter]
        RF[Recognition Model]
        LF[Liveness Model]
    end
    
    SW --> OC
    OC --> TI
    TI --> RF
    TI --> LF
```

## Error Handling

```mermaid
graph TD
    OP[Operation] --> CHECK{Success?}
    CHECK -->|Yes| OK[Result.ok value]
    CHECK -->|No| ERR[Result.error]
    
    ERR --> CODE[ErrorCode]
    CODE --> HANDLER[Handle Error]
    
    subgraph "Error Codes"
        FACE_NOT_FOUND
        MODEL_NOT_LOADED
        LIVENESS_FAILED
        ENROLLMENT_FAILED
        VERIFICATION_FAILED
        SYNC_FAILED
    end
```

## Sync Architecture

```mermaid
graph TB
    subgraph "Offline"
        ENROLL[Enrollment] --> QUEUE[Sync Queue]
        QUEUE --> SQLITE[SQLite Storage]
    end
    
    subgraph "Sync Manager"
        NM[Network Monitor] --> |Online| SM[Sync Manager]
        SM --> BATCH[Batch Upload]
        BATCH --> TRANSPORT[REST Transport]
    end
    
    subgraph "Cloud"
        TRANSPORT --> AWS[AWS Endpoint]
        AWS --> CONFIRM[Confirmation]
    end
    
    subgraph "Purge"
        CONFIRM --> PURGE[Purge Manager]
        PURGE --> DELETE[Delete Local Data]
    end
    
    SQLITE --> SM
```

## Testing Strategy

```mermaid
graph LR
    subgraph "Unit Tests (86)"
        UT1[CosineSimilarity]
        UT2[BlinkDetector]
        UT3[DepthEstimator]
        UT4[EmbeddingStore]
        UT5[Encoder]
        UT6[FacePipeline]
        UT7[LivenessOrchestrator]
        UT8[SyncManager]
        UT9[SyncQueue]
        UT10[DataPurge]
        UT11[Result]
        UT12[Constants]
    end
    
    subgraph "Integration"
        IT1[Pipeline E2E]
        IT2[Sync Flow]
    end
    
    subgraph "Manual"
        MT1[Camera Test]
        MT2[Device Test]
    end
```

## Performance Characteristics

```mermaid
graph LR
    subgraph "Model Size"
        REC[Recognition: 10.6 MB]
        LIV[Liveness: 0.5 MB]
        TOTAL[Total: 11.1 MB]
    end
    
    subgraph "Inference Time"
        T1[Encode: ~100ms]
        T2[Liveness: ~50ms]
        T3[Total: ~400ms]
    end
    
    subgraph "Memory"
        M1[Models: ~20 MB]
        M2[Buffers: ~10 MB]
        M3[Peak: ~30 MB]
    end
```

## Security Considerations

| Concern | Mitigation |
|---------|------------|
| Face data storage | L2-normalized embeddings only (not raw images) |
| Local storage | SQLite with app-private directory |
| Network sync | HTTPS with auth token |
| Model tampering | Models bundled in APK assets |
| Spoofing | Multi-modal liveness (blink + texture + depth) |

## Future Enhancements

```mermaid
graph LR
    NOW[Current] --> F1[Federated Learning]
    NOW --> F2[Smile Detection]
    NOW --> F3[Head Turn]
    NOW --> F4[Template Encryption]
    NOW --> F5[Model Fine-tuning]
```

---

**NetraEdge** — Secure. Offline. Lightweight.

NHAI Hackathon 7.0 | Datalake 3.0 Integration
