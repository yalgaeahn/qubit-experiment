# IQ Cloud Instruction

## 1. Scope

`iq_cloud`는 readout calibration 전용 실험/분석 모듈입니다.

- 지원 상태: `g/e` only
- 지원 qubit 수:
  - 1개: single-qubit readout (`g`, `e`)
  - 2개: multiplexed readout (`gg`, `ge`, `eg`, `ee`)
- 기본 acquisition:
  - `AveragingMode.SINGLE_SHOT`
  - `AcquisitionType.INTEGRATION`

파일:
- `experiments/iq_cloud.py`
- `analysis/iq_cloud.py`
- `experiments/iq_cloud_common.py`

## 2. Experiment Workflow

엔트리포인트:
- `experiments.iq_cloud.experiment_workflow(session, qpu, qubits, temporary_parameters=None, options=None)`

기능:
- 1Q/2Q prepared states를 순회하면서 integrated IQ single shots 획득
- `options.do_analysis=True`이면 분석 수행
- `options.update=True`이면 threshold를 qpu 파라미터에 반영

핸들 규칙:
- `"{qubit_uid}/iq_cloud/{prepared_label}"`

## 3. Analysis Model

엔트리포인트:
- `analysis.iq_cloud.analysis_workflow(result, qubits, options=None)`

각 qubit마다 다음을 추정합니다.

1. 클래스 평균
- `mu_g`, `mu_e` (IQ 2D 벡터)

2. Shared covariance (pooled)
- `S = ((n_g-1)Cov_g + (n_e-1)Cov_e) / (n_g+n_e-2)`

3. Trace-scaled ridge
- `Sigma = S + lambda I`
- `lambda`는 고정값이 아니라 데이터 기반으로 자동 추정
- 내부적으로 `ridge_alpha = lambda / (trace(S)/2)` 저장

4. Bayes decision rule (equal prior)
- prior: `P(g)=P(e)=0.5`
- `w = Sigma^{-1}(mu_e - mu_g)`
- `b = -0.5*(mu_e^T Sigma^{-1} mu_e - mu_g^T Sigma^{-1} mu_g)`
- 판정: `w^T x + b >= 0 -> e`, else `g`

5. Threshold representation
- `w` (2D)
- `b` (scalar)
- `t` (axis threshold scalar, `t = -b/||w||`)

## 4. Metrics

### 4.1 Assignment fidelity / confusion matrix

- confusion matrix는 모두 저장:
  - raw counts
  - row-normalized probabilities
- 축 정의:
  - 행: prepared
  - 열: predicted
  - 클래스 순서: `[g(0), e(1)]`

fidelity:
- per-qubit fidelity: `trace(counts_2x2) / sum(counts_2x2)`
- 2Q multiplexed:
  - joint fidelity: `trace(counts_4x4) / sum(counts_4x4)`
  - average fidelity: `(fid_q0 + fid_q1)/2`

### 4.2 Readout separation / SNR

qubit별 저장:
- `delta_mu_over_sigma = |w^T(mu_e-mu_g)| / sqrt(w^T Sigma w)`
- `mahalanobis_distance = sqrt((mu_e-mu_g)^T Sigma^{-1}(mu_e-mu_g))`

## 5. Bootstrap Error Bars

대상 metric:
- fidelity
- threshold (`t`)
- `delta_mu_over_sigma`
- `mahalanobis_distance`

기본 설정:
- `bootstrap_samples = 2000`
- `bootstrap_confidence_level = 0.95`
- percentile CI
- prepared-state stratified resampling
- 2Q에서는 pair-resampling으로 qubit 간 shot alignment 유지
- `bootstrap_seed` 옵션으로 재현성 제어

## 6. Output Schema (analysis_workflow)

`analysis_workflow(...).run().output`의 핵심 키:

- `decision_model`
- `thresholds`
- `confusion_matrices`
- `assignment_fidelity`
- `separation_metrics`
- `bootstrap`
- `qubit_parameters`

### 6.1 `decision_model`

```python
{
  q_uid: {
    "mu_g": [float, float],
    "mu_e": [float, float],
    "sigma": [[float, float], [float, float]],
    "inv_sigma": [[float, float], [float, float]],
    "w": [float, float],
    "b": float,
    "t": float,
    "axis_unit": [float, float],
    "ridge_lambda": float,
    "ridge_alpha": float,
  },
  ...
}
```

### 6.2 `thresholds`

```python
{
  q_uid: {
    "w": [float, float],
    "b": float,
    "t": float,
    "axis_unit": [float, float],
  },
  ...
}
```

### 6.3 `confusion_matrices`

```python
{
  "per_qubit": {
    q_uid: {
      "counts": [[int, int], [int, int]],
      "normalized": [[float, float], [float, float]],
      "labels": ["g", "e"],
    },
    ...
  },
  # 2Q only
  "joint": {
    "counts": [[int]*4]*4,
    "normalized": [[float]*4]*4,
    "labels": ["gg", "ge", "eg", "ee"],
  },
}
```

### 6.4 `assignment_fidelity`

```python
{
  "per_qubit": {q_uid: float, ...},
  # 2Q only
  "joint": float,
  "average": float,
}
```

### 6.5 `separation_metrics`

```python
{
  "per_qubit": {
    q_uid: {
      "delta_mu_over_sigma": float,
      "mahalanobis_distance": float,
    },
    ...
  }
}
```

### 6.6 `bootstrap`

```python
{
  "per_qubit": {
    q_uid: {
      "fidelity": {
        "mean": float,
        "std": float,
        "ci_low": float,
        "ci_high": float,
        "confidence_level": float,
      },
      "threshold": {...},
      "delta_mu_over_sigma": {...},
      "mahalanobis_distance": {...},
    },
    ...
  },
  # 2Q only
  "joint": {"fidelity": {...}},
  "average": {"fidelity": {...}},
  "settings": {
    "bootstrap_samples": int,
    "confidence_level": float,
    "seed": int | None,
  },
}
```

### 6.7 `qubit_parameters`

```python
{
  "old_parameter_values": {
    q_uid: {
      "readout_integration_discrimination_thresholds": ...,
      "readout_integration_kernels_type": ...,
      "readout_integration_kernels": ...,
    },
    ...
  },
  "new_parameter_values": {
    q_uid: {
      "readout_integration_discrimination_thresholds": [t],
      "readout_integration_kernels_type": "default",   # enforce_constant_kernel=True
      "readout_integration_kernels": None,               # enforce_constant_kernel=True
    },
    ...
  }
}
```

## 7. Minimal Usage

```python
from experiments import iq_cloud

opts = iq_cloud.experiment_workflow.options()
opts.do_analysis(True)
opts.update(False)

# 1Q
res_1q = iq_cloud.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=[qubits[0]],
    options=opts,
).run()

# 2Q multiplexed
res_2q = iq_cloud.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=[qubits[0], qubits[1]],
    options=opts,
).run()
```
