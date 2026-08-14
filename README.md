# Transformer Health Monitoring from Dissolved Gas Analysis

Classifies incipient faults in oil-filled power transformers from dissolved
gas concentrations, using IEEE/IEC standards and classical diagnostics as the
baseline for a calibrated ML classifier.

## Why this matters

A large power transformer is one of the most expensive and least replaceable
assets on the grid. A single unit costs roughly USD 2–10 million depending on
its rating, weighs between 100 and 400 tons, and cannot be bought off the
shelf. As of 2025, average lead times for power transformers reached about
128 weeks (~2.5 years), extra-high-voltage units can take up to five years from
order to delivery, and prices have risen roughly 70–77% since 2019. So when a
transformer fails unexpectedly, the operator faces not only the replacement
cost but years without the asset.

That is what makes early condition monitoring valuable. Dissolved Gas Analysis
(DGA) detects developing faults — overheating, partial discharge, arcing —
while the transformer is still in service and long before failure, by reading
the gases that these faults dissolve into the insulating oil. Catching a fault
early can be the difference between a planned intervention and a multi-year
outage. This project builds that diagnostic capability, grounded in the IEEE
and IEC standards utilities actually use.

<!-- Figures above are representative, US-sourced (2025). Verify against a
     source you can cite before defending them, e.g.:
     - Lead times & price rises: POWER Magazine, "Transformers in 2026" (2026)
     - Unit cost & weight ranges: NREL Large Power Transformer supply-chain
       gap analysis (2026)
     Rewrite this section in your own words once you understand each number —
     an interviewer will ask you to explain them, not recite them. -->

## Planned
- [ ] Classical methods: Key Gas, IEC ratios, Duval
- [ ] Feature engineering from gas ratios
- [ ] Calibrated classifier, benchmarked vs classical
- [ ] SHAP explanations and fleet risk ranking

Status: in progress.
