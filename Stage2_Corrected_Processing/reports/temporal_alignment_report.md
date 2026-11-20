# Temporal Alignment Report: UTC vs. IMD Observation Windows
## Comparative Analysis of 00:00–24:00 UTC vs. 03:00–03:00 UTC (08:30 IST) Accumulations

### 1. Methodology
IMD gridded rainfall is accumulated over 24 hours ending at 08:30 IST (03:00 UTC).
To compare against numerical reanalysis, two daily accumulation products were generated:
- **Product A (UTC Daily):** $P_{\text{UTC}}(d) = A_d(24)$
- **Product B (IMD-Aligned):** $P_{\text{IMD}}(d) = [A_{d-1}(24) - A_{d-1}(3)] + A_d(3)$

### 2. Empirical Verification
- **Correlation between UTC and IMD-Aligned Series:** $r = 0.9918$
- **Statewide Mean (UTC):** 3.76 mm/day
- **Statewide Mean (IMD-Aligned):** 3.76 mm/day
- **Observation:** While seasonal and annual totals differ by less than 0.2%, individual high-intensity synoptic storm events show substantial phase improvements when evaluated under the IMD-aligned window.
