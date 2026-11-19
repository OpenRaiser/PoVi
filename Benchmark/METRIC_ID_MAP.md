# PoVi-Bench metrics (paper FINAL)

Registry version `0.6.0`. Primary scored metrics are exactly these **17** IDs.

| ID | Name | Code domain | Scale |
|---|---|---|---|
| S1 | Information Coverage | content | [0,1] |
| S2 | Metaphor Source | content | [0,1] |
| S3 | Metaphor Target | content | [0,1] |
| S4 | Mapping Bridge | content | [0,1] |
| S5 | Metaphor Contextual Support | content | [0,1] |
| S6 | Interpretive Calibration | content | [0,1] |
| D1 | Narrative Structure | content | 1–5 |
| D2 | Prelude–Explication Linkage | content | 1–5 |
| D3 | Modality Complementarity | av | 1–5 |
| D4 | Audiovisual Pacing | av | 1–5 |
| R1 | Visual Requirement Coverage | visual | [0,1] |
| R2 | Visual Semantic Correctness | visual | [0,1] |
| R3 | Static Visual Quality | visual | 1–5 |
| R4 | Character Identity Continuity | visual | [0,1] |
| R5 | Visual Metaphor Realization | visual | 1–5 |
| R6 | Cross-modal Correspondence | av | [0,1] |
| R7 | Narration–Event Synchrony | av | [0,1] |

## Normalization

- `[0,1]` scores are used as-is.
- `1–5` scores are normalized as **score / 5** (paper convention).

## Report groups

| Paper group | Metrics |
|---|---|
| Narration | S1–S6, D1–D2 |
| Video | R1–R5 |
| Cross-modal | D3–D4, R6–R7 |

Gates (`G*`) are validity checks only and are not paper metrics.
