# 포세이돈 (Poseidon) 🌊⚓

> **범용 해양 자율운항 엔진 v0.2.0**
> 수상함 · 잠수함 · 요트 · 보트 · 자율USV — 단일 엔진

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-82_passed-brightgreen)](tests/)
[![Stdlib](https://img.shields.io/badge/Core-stdlib_only-lightgrey)](marine_autonomy/)

**English version:** [README_EN.md](README_EN.md)

---

## 무엇인가

Poseidon은 모든 해양 기체를 위한 **범용 자율운항 제어 엔진**이다.

수상함, 잠수함, 요트, 보트, 자율 무인수상정(USV) — 선체 클래스만 바꾸면 동일한 엔진이 작동한다.
외부 의존 없이 Jetson Nano, Raspberry Pi 같은 엣지 디바이스에서 바로 구동된다.

```
핵심 레이어 (v0.2.0):
  비선형 Fossen 3-DOF 동역학  — Coriolis 행렬 + 비선형 감쇠 + RK4 적분
  외란 모델                   — 파도(정현파) + 바람(동압) + 조류(상대속도 보정)
  LOS 경로 추종               — Line-of-Sight 헤딩 제어 + 웨이포인트 순차 추종
  다중 선박 COLREGs FSM       — 전체 contacts 동시 판정 + 최대 회피각 합성
  해양 A* 경로 계획           — 수심도 기반 장애물 회피 (수심 보너스 비용)
  Marine EKF 상태 추정        — GPS + 헤딩 + 속도 융합 (4-state, 순수 Python 행렬)
  Ω 안전 판정                — 위험·연료·시야·수심·다중조우 복합 감쇠
  범용 선체 클래스            — 수상함 / 잠수함(4-DOF) / 요트 / 보트 / USV
```

> **주의:** COLREGs 논리는 운영용 내부 휴리스틱이며 공식 해사 인증 대체재가 아니다.

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         Poseidon v0.2.0                         │
│                                                                 │
│  [센서 / AIS / CARLA / ROS2 / 자체 시뮬레이터]                   │
│              ↓                                                  │
│  DisturbanceState  MarinePerception  MarineTickContext           │
│  (파도·바람·조류)   (다중 contacts)    (hull_class 포함)          │
│              ↓                                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │               VesselOrchestrator                      │      │
│  │                                                       │      │
│  │  [HullClass 분기]                                     │      │
│  │   SURFACE / YACHT / BOAT / USV → 3-DOF               │      │
│  │   SUBMARINE                    → 4-DOF (depth 추가)  │      │
│  │                                                       │      │
│  │  L1  MarineEKF         상태 추정 (GPS+헤딩+속도)       │      │
│  │  L2  COLREGs FSM       다중 선박 동시 충돌 회피        │      │
│  │  L3  maritime_astar    수심도 A* 전역 경로 계획        │      │
│  │  L4  LOS Guidance      경로 추종 + COLREGs 헤딩 보정  │      │
│  │  L5  Fossen Dynamics   비선형 3/4-DOF + 외란 적용     │      │
│  │      Ω Safety          복합 안전 판정                 │      │
│  └────────────────────────┬──────────────────────────────┘      │
│                           │ VesselActuator                      │
│                           ↓                                     │
│               throttle / rudder_norm / reverse                  │
│                           ↓                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │         marine-propulsion-engine 연동 (선택)        │         │
│  │  PropulsionOrchestrator  ←  ShaftEKF + FSM         │         │
│  │  SHA-256 CommandChain    ←  불변 감사 기록          │         │
│  └────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 범용 선체 클래스

| HullClass | DOF | 대표 기체 | 기본 프리셋 |
|-----------|-----|---------|-----------|
| `SURFACE_VESSEL` | 3 | 10m 연안 순찰함 | harbor / coastal / ocean / river |
| `SUBMARINE` | 4 | 50m 잠수함 | sub_shallow / sub_deep |
| `YACHT` | 3 | 12m 요트 | yacht_racing / yacht_cruising |
| `BOAT` | 3 | 6m 고속정 | boat_patrol / boat_harbor |
| `AUTONOMOUS_USV` | 3 | 3m 무인수상정 | usv_survey |

```python
from marine_autonomy.dynamics import submarine_params, yacht_params, boat_params
from marine_autonomy.presets  import get_hull_preset

# 잠수함 심해 순항
sub  = submarine_params()                         # VesselParams
pset = get_hull_preset("submarine", "deep")       # MarinePreset (20kn)

# 레이싱 요트
ycht = yacht_params()
pset = get_hull_preset("yacht", "racing")         # 16kn

# 고속 순찰정
bt   = boat_params()
pset = get_hull_preset("boat", "patrol")          # 30kn
```

---

## 핵심 수식

### 1. 비선형 Fossen 3-DOF (v0.2.0 완성)

```
M·ν̇ = τ_ctrl + τ_dist − C(ν)·ν − D(ν)·ν

Coriolis 행렬 C(ν):         [  0       0      −m22·v−m26·r ]
  m11 = m − X_u̇              [  0       0       m11·u       ]
  m22 = m − Y_v̇              [  m22·v+m26·r  −m11·u    0   ]
  m26 = −Y_ṙ

비선형 감쇠 D(ν)·ν:
  X방향: Xu·u + Xuu·|u|·u
  Y방향: Yv·v + Yvv·|v|·v
  N방향: Nr·r + Nrr·|r|·r

수치 적분: RK4
```

### 2. 외란 합력 τ_dist = τ_wave + τ_wind (조류는 상대속도 보정)

```
파도 (단순 정현파 근사):
  τ_wave_X = k·Hs²·Awp·cos(ψ_wave−ψ)·sin(2π·t/Tp)
  k = 0.05·ρ_w·g·Awp / Lpp

바람 (동압 모델):
  q = 0.5·ρ_air·Vw²
  τ_wind_X =  q·Cx·A_lat·cos(ψ_wind−ψ)
  τ_wind_Y = −q·Cy·A_lat·sin(ψ_wind−ψ)
  τ_wind_N = −q·Cn·A_lat·Lpp·sin(ψ_wind−ψ)

조류 상대속도 보정:
  u_r = u − curr_u·cos(ψ) − curr_v·sin(ψ)
  v_r = v + curr_u·sin(ψ) − curr_v·cos(ψ)
  → 감쇠 계산 시 u, v 대신 u_r, v_r 사용
```

### 3. 잠수함 깊이 제어 (4-DOF)

```
e_z   = target_depth − depth
ẇ     = Bz·e_z − Kz·w
depth' = depth + w·dt
```

### 4. LOS 경로 추종

```
ψ_d = α_k + atan2(−e, Δ) + δ_colregs

α_k = atan2(Δy, Δx)           경로각
e   = −Δx·sin(α_k)+Δy·cos(α_k)  크로스트랙 오차
δ_colregs = COLREGs 회피 헤딩 보정값 (다중 선박 합성)

PD 헤딩: δ_rudder = Kp·(ψ_d−ψ) − Kd·r
```

### 5. 다중 선박 COLREGs (v0.2.0)

```
모든 contacts 동시 분류:
  HEAD_ON          — |β| < 15° and ΔCOG > 150°
  CROSSING_GIVEWAY — 10° ≤ β ≤ 112.5° (우현)
  OVERTAKING       — 112.5° < |β| < 247.5° (선미)
  SAFE             — d > d_safe

우선순위 판정:
  1. EMERGENCY_STOP — any contact d < d_emg
  2. GIVE_WAY       — give_way contacts 중 최대 회피각 적용
                      HEAD_ON→20°, CROSSING→15°, OVERTAKING→10°
  3. STAND_ON       — crossing stand-on
  4. CRUISE         — 위험 없음

반환: state, avoid_heading_offset_rad, situations[], dominant_contact
```

### 6. 해양 A* 경로 계획 (v0.2.0)

```
비용 함수:
  g(n) = Σ step_cost + depth_penalty
  step_cost: 직선=1.0, 대각선=1.414
  depth_penalty = max(0, 1−(depth−draft)/10) × 0.2  ← 얕을수록 비쌈

휴리스틱: h(n) = 유클리드 거리 (8방향)
통항 조건: depth[row][col] ≥ draft + min_depth_m
```

### 7. Ω 안전 판정 (v0.2.0 확장)

```
Ω = ω_risk × ω_fuel × ω_vis × ω_depth × ω_contacts
```

| 조건 | 감쇠 ω |
|------|--------|
| `risk_score > 0.85` | × 0.55 |
| `risk_score > 0.35` | × 0.78 |
| `fuel < 0.20` | × 0.70 |
| `visibility < 200m` | × 0.80 |
| `depth < 3×draft` | × 0.65 |
| `contacts ≥ 3척` | × 0.90 |

```
HEALTHY  — Ω ≥ 0.82
STABLE   — Ω ≥ 0.55
FRAGILE  — Ω ≥ 0.30
CRITICAL — Ω < 0.30
```

---

## 프리셋 전체 목록

### 수상함 / 보트 / 요트 / USV

| 키 | 선체 | 순항 | 최대 | 환경 |
|----|------|------|------|------|
| `harbor` | 수상함 | 3 kn | 5 kn | 항만 내 기동 |
| `coastal` | 수상함 | 10 kn | 15 kn | 연안 순찰 |
| `ocean` | 수상함 | 20 kn | 25 kn | 외양 항해 |
| `river` | 수상함 | 5 kn | 8 kn | 내륙 수로 |
| `yacht_racing` | 요트 | 12 kn | 16 kn | 레이싱 |
| `yacht_cruising` | 요트 | 7 kn | 10 kn | 크루징 |
| `boat_patrol` | 보트 | 20 kn | 30 kn | 고속 순찰 |
| `boat_harbor` | 보트 | 3 kn | 5 kn | 항만 기동 |
| `usv_survey` | USV | 4 kn | 6 kn | 수중 탐사 |

### 잠수함

| 키 | 순항 | 최대 | 환경 |
|----|------|------|------|
| `sub_shallow` | 6 kn | 10 kn | 천해 작전 (draft 5m) |
| `sub_deep` | 15 kn | 20 kn | 심해 순항 (draft 8m) |

---

## 빠른 시작

### 설치

```bash
git clone https://github.com/qquartsco-svg/Poseidon.git
cd Poseidon
pip install -e .            # 코어 (stdlib only)
pip install -e ".[full]"    # Autonomy_Runtime_Stack 포함
```

### 드라이런

```bash
# 수상함 항만 프리셋
python examples/run_harbor.py --preset harbor --steps 200

# 연안 프리셋
python examples/run_harbor.py --preset coastal --steps 500

# 외양 프리셋
python examples/run_harbor.py --preset ocean --steps 300 --waypoints "0,0 1000,500 2000,0"
```

### 수상함 — 다중 선박 COLREGs + 외란

```python
from marine_autonomy import VesselOrchestrator, MarineTickContext, get_preset
from marine_autonomy.contracts.schemas import (
    MarinePerception, ContactVessel, DisturbanceState, HullClass
)

orch = VesselOrchestrator(preset=get_preset("coastal"))

ctx = MarineTickContext(
    hull_class=HullClass.SURFACE_VESSEL,
    waypoints=((0.0, 0.0), (500.0, 200.0), (1000.0, 0.0)),
    perception=MarinePerception(
        contacts=(
            ContactVessel(id="TGT-01", range_m=400.0, bearing_rad=0.1,
                          cog_rad=3.14, sog_ms=6.0),   # HEAD-ON
            ContactVessel(id="TGT-02", range_m=600.0, bearing_rad=1.2,
                          cog_rad=1.5,  sog_ms=4.0),   # CROSSING
        ),
        visibility_m=1500.0,
        depth_m=25.0,
    ),
    disturbance=DisturbanceState(
        wave_height_m=1.5, wave_period_s=8.0, wave_dir_rad=0.5,
        wind_speed_ms=8.0, wind_dir_rad=1.0,
        current_u_ms=0.5,  current_v_ms=0.2,
        t_s=0.0,
    ),
)

for step in range(200):
    ctx = orch.tick(ctx, dt_s=0.1)
    print(f"[{step:3d}] COLREGs={ctx.colregs_state:15s} Ω={ctx.omega:.3f} {ctx.verdict}")
```

### 잠수함 — 4-DOF 잠항 제어

```python
from marine_autonomy.contracts.schemas import HullClass, SubmarineState
from marine_autonomy.dynamics import submarine_params, submarine_depth_step
from marine_autonomy.presets  import get_hull_preset

params = submarine_params()
pset   = get_hull_preset("submarine", "deep")

depth, w = 0.0, 0.0
for step in range(300):
    depth, w = submarine_depth_step(depth, w, target_depth_m=50.0,
                                    params=params, dt_s=0.1)
    print(f"depth={depth:.1f}m  w={w:.3f}m/s")
```

### 수심도 A* 경로 계획

```python
from marine_autonomy.guidance import DepthChart, maritime_astar

# 수심도 생성 (50×50 격자, 10m 해상도)
grid = [[20.0] * 50 for _ in range(50)]
# 얕은 구역 (장애물)
for r in range(20, 30):
    for c in range(10, 40):
        grid[r][c] = 1.0   # 1m 수심 → 통항 불가

chart = DepthChart(grid=grid, resolution_m=10.0, min_depth_m=3.0)
path  = maritime_astar(chart, start_xy=(0, 0), goal_xy=(490, 490), draft_m=2.0)
print(f"경로 웨이포인트: {len(path)}개")
```

### marine-propulsion-engine 연동

```python
from marine_autonomy import VesselOrchestrator, MarineTickContext, get_preset
from propulsion import MarineBridge

nav    = VesselOrchestrator(preset=get_preset("coastal"))
bridge = MarineBridge()
ctx    = MarineTickContext(waypoints=((0.0, 0.0), (500.0, 200.0)))

for step in range(200):
    ctx      = nav.tick(ctx, dt_s=0.1)
    prop_ctx = bridge.tick_from_vessel(
        vessel_actuator=ctx.actuator,
        measured_angle=180.0,
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
│   ├── dynamics.py            — 비선형 Fossen + 외란 + 잠수함 4-DOF
│   ├── guidance.py            — LOS 추종 + 해양 A* (DepthChart)
│   ├── estimation.py          — Marine EKF (4-state, stdlib only)
│   ├── colregs.py             — 다중 선박 COLREGs FSM
│   ├── orchestrator.py        — VesselOrchestrator (hull-class aware)
│   ├── presets.py             — PRESET_REGISTRY + HULL_PRESETS
│   ├── contracts/
│   │   └── schemas.py         — HullClass, DisturbanceState, SubmarineState,
│   │                            VesselState, MarinePerception, MarineTickContext
│   └── adapters/
│       └── ais_adapter.py     — AIS 메시지 파서
├── tests/
│   ├── test_marine.py         — 36 tests (기본 레이어)
│   └── test_poseidon_upgrade.py — 46 tests (v0.2.0 업그레이드)
├── examples/
│   └── run_harbor.py          — CLI 드라이런
├── pyproject.toml
├── README.md                  — 한국어 (이 파일)
└── README_EN.md               — English version
```

---

## 확장성

### 새 선체 클래스 추가

```python
from marine_autonomy.dynamics import VesselParams
from marine_autonomy.presets  import MarinePreset, PRESET_REGISTRY, HULL_PRESETS

# 수중 드론 (소형 AUV)
HULL_PRESETS["auv"] = {
    "shallow": MarinePreset(
        name="auv_shallow", max_speed_ms=2.06, cruise_speed_ms=1.03,
        lookahead_m=10.0, acceptance_radius_m=3.0,
        safe_range_m=30.0, action_range_m=60.0, emergency_range_m=5.0,
        draft_m=0.3, description="소형 AUV 천해 탐사",
    ),
}

def auv_params(**kw) -> VesselParams:
    return VesselParams(
        hull_class="auv", name="AUV-mini",
        mass_kg=50.0, Iz_kgm2=5.0,
        max_thrust_n=200.0, Lpp_m=1.5, **kw
    )
```

### 커스텀 브릿지 (ROS2 / AirSim)

```python
class ROS2Bridge:
    def tick(self, ctx, *, path=None, ekf=None):
        # ROS2 publish/subscribe
        ...

orch = VesselOrchestrator(bridge=ROS2Bridge(), preset=get_preset("coastal"))
```

---

## 의존 관계

```
Poseidon (코어)
│  stdlib only — math, heapq, json, dataclasses, typing
│
├── 선택 의존
│   └── Autonomy_Runtime_Stack  (pip install -e ".[full]")
│
└── 연동 가능
    ├── marine-propulsion-engine  → MarineBridge (추진축 SHA-256 감사)
    ├── SYD_DRIFT                 → CommandChain (동일 설계)
    ├── CARLA 0.9.x               → CarlaSimBridge
    └── ROS2 / AirSim             → 커스텀 bridge.tick()
```

---

## 테스트

```bash
pytest tests/ -v
# 82 passed, 0 failed (stdlib only)
# test_marine.py          36 tests — 기본 레이어
# test_poseidon_upgrade.py 46 tests — v0.2.0 업그레이드
```

---

## 관련 레포

| 레포 | 역할 |
|------|------|
| [Autonomy_Runtime_Stack](https://github.com/qquartsco-svg/Autonomy_Runtime_Stack) | 자율주행 기초 엔진 (EKF·FSM·Stanley) |
| [SYD_DRIFT](https://github.com/qquartsco-svg/SYD_DRIFT) | 자율주행 + SHA-256 감사 체인 |
| [marine-propulsion-engine](https://github.com/qquartsco-svg/marine-propulsion-engine) | 선박 추진축 마모 제어 + 감사 체인 |
| **Poseidon** | 범용 해양 자율운항 엔진 (이 레포) |

---

## 라이선스

MIT

---

*Poseidon — 바다의 신. 수상함부터 잠수함까지, 모든 해양 기체의 항로를 찾는다.*
