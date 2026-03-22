# 포세이돈 (Poseidon) 🌊⚓

> **자율 선박 동역학 엔진 — LOS 항법 × COLREGs × EKF × Fossen 3-DOF**
> 엣지 AI 플랫폼용 자율운항 런타임

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Edge_AI-orange)](marine_autonomy/)
[![Stdlib](https://img.shields.io/badge/Core-stdlib_only-lightgrey)](marine_autonomy/)

**English version:** [README_EN.md](README_EN.md)

---

## 무엇인가

Poseidon은 자율 선박 제어를 위한 순수 Python 런타임 패키지다.

[Autonomy_Runtime_Stack](https://github.com/qquartsco-svg/Autonomy_Runtime_Stack)의 형제 패키지로,
동일한 Edge AI 설계원칙을 해양 환경에 적용했다.
외부 의존 없이 Jetson Nano, Raspberry Pi 같은 엣지 디바이스에서 바로 구동된다.

```
핵심 기능:
  Fossen 3-DOF 선박 동역학   — surge / sway / yaw 물리 모델 (RK4 적분)
  LOS 경로 추종              — Line-of-Sight 헤딩 제어 + 웨이포인트 순차 추종
  COLREGs 충돌 회피 FSM      — Rule 8/13/14/15/16 (HEAD-ON / CROSSING / OVERTAKING)
  Marine EKF 상태 추정       — GPS + 헤딩 + 속도 융합 (4-state, 순수 Python 행렬)
  Ω 안전 판정               — 위험·연료·시야·수심 복합 감쇠 모델
  도시별 확장 프리셋          — 항만 / 연안 / 외양 / 강 환경
```

> **주의:** 이 패키지는 COLREGs 인증 시스템이 아니다.
> COLREGs 논리는 운영용 내부 휴리스틱이며, 공식 해사 인증 대체재가 아니다.

---

## 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                        Poseidon                              │
│                                                              │
│  [센서 / CARLA / ROS2 / AIS]                                 │
│           ↓                                                  │
│  MarinePerception ──── MarineTickContext                     │
│  VesselState      ──────────────────────┐                   │
│           ↓                             │                   │
│  ┌──────────────────────────────────┐   │                   │
│  │     VesselOrchestrator           │   │                   │
│  │  L1  MarineEKF    상태 추정      │   │                   │
│  │  L2  COLREGs FSM  충돌 회피      │   │                   │
│  │  L3  LOS Guidance 경로 추종      │   │                   │
│  │      Ω Safety     안전 판정      │   │                   │
│  └──────────────┬───────────────────┘   │                   │
│                 │ VesselActuator        │                   │
│                 ↓                       │                   │
│     throttle / rudder_norm / reverse    │                   │
│                 ↓                       │                   │
│  ┌─────────────────────────────────┐   │                   │
│  │  marine-propulsion-engine 연동  │◄──┘                   │
│  │  PropulsionOrchestrator         │                        │
│  │  SHA-256 CommandChain           │                        │
│  └─────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 핵심 수식

### 1. Fossen 3-DOF 선박 동역학

```
M · ν̇ = τ − D·ν + τ_dist
η̇ = J(ψ)·ν

η = [x, y, ψ]ᵀ   (위치·헤딩)
ν = [u, v, r]ᵀ   (surge·sway·yaw rate)

M = diag(m−X_u̇, m−Y_v̇, Iz−N_ṙ)   부가질량 포함 관성 행렬
D = diag(−X_u, −Y_v, −N_r)          선형 감쇠
```

수치 적분: **4차 Runge-Kutta (RK4)**

---

### 2. LOS 경로 추종

```
ψ_d = α_k + atan2(−e, Δ)

α_k = atan2(y_{k+1}−y_k, x_{k+1}−x_k)    경로각
e   = −(x−x_k)·sin(α_k) + (y−y_k)·cos(α_k)   크로스트랙 오차
Δ   = lookahead distance (전방 주시 거리)

PD 헤딩 제어:  δ = Kp·ψ_e − Kd·r
```

---

### 3. COLREGs 상황 분류

```
상황 분류:
  HEAD_ON          — |β| < 15° and ΔCOG > 150°
  CROSSING_GIVEWAY — 10° ≤ β ≤ 112.5° (우현)
  OVERTAKING       — 112.5° < |β| < 247.5° (선미)
  SAFE             — d > d_safe

상태 전이:
  EMERGENCY_STOP   — d < d_emg
  GIVE_WAY         — HEAD_ON or CROSSING_GIVEWAY
  STAND_ON         — CROSSING_STANDON
  CRUISE           — otherwise
```

---

### 4. Marine EKF 상태 추정

```
상태 벡터: x̂ = [x, y, ψ, u]ᵀ

예측:  P_{k|k-1} = F·P·Fᵀ + Q
칼만 이득:  K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
갱신:  x̂_k = x̂_{k|k-1} + K·(z − H·x̂_{k|k-1})

측정: GPS(z_xy), 헤딩(z_ψ), 속도(z_u) 독립 갱신
```

---

### 5. Ω 안전 판정

```
Ω = ∏ ωk,   ωk ∈ (0, 1]
```

| 조건 | 감쇠 ωk |
|------|---------|
| `risk_score > 0.85` | × 0.55 |
| `risk_score > 0.35` | × 0.78 |
| `fuel < 0.20` | × 0.70 |
| `visibility < 200m` | × 0.80 |
| `depth < 3 × draft` | × 0.65 |

```
판정:
  HEALTHY  — Ω ≥ 0.82
  STABLE   — Ω ≥ 0.55
  FRAGILE  — Ω ≥ 0.30
  CRITICAL — Ω < 0.30
```

> Ω는 운영용 내부 risk heuristic이며, ISO 15739 / SOLAS 인증 지표가 아니다.

---

## 환경 프리셋

| 키 | 순항 속도 | 최대 속도 | 환경 |
|----|---------|---------|------|
| `harbor` | 3 kn | 5 kn | 항만 내 기동, 소형선 혼재 |
| `coastal` | 10 kn | 15 kn | 연안 순찰, COLREGs 상시 적용 |
| `ocean` | 20 kn | 25 kn | 외양 항해, 대형 선박 조우 가능 |
| `river` | 5 kn | 8 kn | 내륙 수로, 수심 제약 |

### 새 환경 프리셋 추가 (4줄)

```python
from marine_autonomy.presets import SydneyRoadPreset, PRESET_REGISTRY

PRESET_REGISTRY["busan_port"] = SydneyRoadPreset(
    name="busan_port", speed_limit_ms=2.57, cruise_speed_ms=1.54,
    lookahead_m=25.0, acceptance_radius_m=12.0,
    safe_range_m=120.0, action_range_m=250.0, emergency_range_m=25.0,
    draft_m=2.0, description="부산항 내항 — 5kn 제한",
)
```

---

## 빠른 시작

### 설치

```bash
git clone https://github.com/qquartsco-svg/Poseidon.git
cd Poseidon
pip install -e .          # 코어 (stdlib only)
pip install -e ".[full]"  # Autonomy_Runtime_Stack 포함
```

### 드라이런

```bash
python examples/run_harbor.py --preset harbor --steps 200 --waypoints "0,0 100,50 200,0"
python examples/run_harbor.py --preset coastal --steps 500
python examples/run_harbor.py --preset ocean --steps 300
```

### Python API

```python
from marine_autonomy import VesselOrchestrator, MarineTickContext, get_preset
from marine_autonomy.contracts.schemas import MarinePerception, ContactVessel

orch = VesselOrchestrator(preset=get_preset("coastal"))

ctx = MarineTickContext(
    waypoints=((0.0, 0.0), (500.0, 200.0), (1000.0, 0.0)),
    perception=MarinePerception(
        contacts=(ContactVessel(
            id="TGT-01", range_m=600.0, bearing_rad=0.3,
            cog_rad=3.14, sog_ms=5.0,
        ),),
        visibility_m=2000.0,
        depth_m=30.0,
    ),
)
for step in range(100):
    ctx = orch.tick(ctx, dt_s=0.1)
    print(f"[{step:3d}] COLREGs={ctx.colregs_state:15s} Ω={ctx.omega:.3f} {ctx.verdict}")
```

### marine-propulsion-engine 연동

```python
from marine_autonomy import VesselOrchestrator, MarineTickContext, get_preset
from propulsion import MarineBridge   # marine-propulsion-engine 설치 필요

nav    = VesselOrchestrator(preset=get_preset("coastal"))
bridge = MarineBridge()

ctx = MarineTickContext(waypoints=((0.0, 0.0), (500.0, 200.0)))

for step in range(200):
    ctx      = nav.tick(ctx, dt_s=0.1)
    prop_ctx = bridge.tick_from_vessel(
        vessel_actuator=ctx.actuator,
        measured_angle=180.0,    # 실제 축 센서값
        t_s=step * 0.1,
    )
    print(f"마모={prop_ctx.wear_risk:.3f} | {prop_ctx.verdict}")

bridge.export_audit("shaft_audit.json")
```

---

## 파일 구조

```
Poseidon/
├── marine_autonomy/
│   ├── __init__.py            — 공개 API
│   ├── dynamics.py            — Fossen 3-DOF 동역학 (Euler + RK4)
│   ├── guidance.py            — LOS 경로 추종 + PD 헤딩 제어
│   ├── estimation.py          — Marine EKF (4-state, stdlib only)
│   ├── colregs.py             — COLREGs FSM (Rule 13/14/15/16)
│   ├── orchestrator.py        — VesselOrchestrator (tick 기반)
│   ├── presets.py             — 환경 프리셋 레지스트리
│   ├── contracts/
│   │   └── schemas.py         — VesselState, MarinePerception, MarineTickContext
│   └── adapters/
│       └── ais_adapter.py     — AIS 메시지 파서
├── tests/
│   └── test_marine.py         — 36 passed
├── examples/
│   └── run_harbor.py          — CLI 드라이런
├── pyproject.toml
├── README.md                  — 한국어 (이 파일)
└── README_EN.md               — English version
```

---

## 확장성

### 브릿지 교체

```python
class ROS2Bridge:
    def tick(self, ctx, *, path=None, ekf=None):
        # ROS2 publish/subscribe 구현
        ...

orch = VesselOrchestrator(bridge=ROS2Bridge(), preset=get_preset("coastal"))
```

### 커스텀 AIS 파서

```python
from marine_autonomy.adapters.ais_adapter import parse_ais_contact

contact = parse_ais_contact({
    "id": "MMSI-123456789",
    "range_m": 800.0,
    "bearing_deg": 45.0,
    "cog_deg": 270.0,
    "sog_kn": 12.0,
})
```

---

## 의존 관계

```
Poseidon (코어)
│  stdlib only — math, hashlib, json, dataclasses, typing
│
├── 선택 의존
│   └── Autonomy_Runtime_Stack (pip install -e ".[full]")
│
└── 연동 가능
    ├── marine-propulsion-engine  → MarineBridge (추진축 제어)
    ├── SYD_DRIFT                 → SHA-256 감사 체인 (동일 설계)
    ├── CARLA 0.9.x               → CarlaSimBridge
    └── ROS2 / AirSim             → 커스텀 bridge.tick() 구현
```

---

## 테스트

```bash
pytest tests/ -v
# 36 passed, 0 failed (stdlib only)
```

---

## 관련 레포

| 레포 | 역할 |
|------|------|
| [Autonomy_Runtime_Stack](https://github.com/qquartsco-svg/Autonomy_Runtime_Stack) | 자율주행 기초 엔진 |
| [SYD_DRIFT](https://github.com/qquartsco-svg/SYD_DRIFT) | 자율주행 + SHA-256 감사 체인 |
| [marine-propulsion-engine](https://github.com/qquartsco-svg/marine-propulsion-engine) | 선박 추진축 마모 제어 |
| **Poseidon** | 자율 선박 항법 런타임 (이 레포) |

---

## 라이선스

MIT

---

*Poseidon — 바다의 신. 자율선박이 항로를 찾는다.*
