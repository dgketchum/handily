# Inference Leakage Prevention Policy

This policy governs every pipeline that turns the handily water-table model (or any
well-trained successor) into rasters, validation numbers, or shared products. It
exists because of a July 2026 incident: rendered DTW/WTE maps for NM, UCRB, NV, and
MT carried training-well "bulls-eyes" — the inference runner rebuilt a trained IDW
feature from **all wells including each well's own observation**, while the
checkpoints had been trained on leave-fold-out crossfit versions. Maps scored ~5×
better at training wells than the model's honest out-of-fold error, the artifact
passed every existing gate, and contaminated products shipped before it was caught.

## The first question a reviewer will ask

*"You held out folds during training, then used every well at inference — isn't
that still leakage?"*

No — and the discriminating test is about **values, not participants**. Leakage
is never "which wells does the interpolator touch?"; it is "does any well's
**observation** influence the feature at its own location (or anywhere the model
is evaluated against it) in a way the training features never had?" Walk the
fixed pipeline against that test:

- **What flows through the inference interpolation is never an observation.**
  The lattice R interpolates each well's archived leave-fold-out feature value —
  computed, at training time, with the well's entire fold (its own HUC12 cluster
  and everything grouped with it) excluded. Observations enter nowhere in the
  inference feature path. "All wells" just means all wells carry their
  pre-audited out-of-fold values onto the lattice.
- **At a training well, the inference feature is identical to the training
  feature.** The map's R pins to exactly the value the checkpoint saw for that
  well during training (0.000 m pin on all 34,503 wells). There is nothing extra
  at wells for the model's residual correction to exploit — precisely what the
  all-observation IDW violated.
- **Between wells, the contract also matches.** At training time, a well's
  feature was informed by every well outside its fold, including near neighbors
  across HUC12 boundaries; the inference lattice has the same property. The
  tightest same-site information (the well's own HUC12 cluster) was excluded
  from its archived value, so it cannot sneak back in through interpolation of
  the well's own pin.
- **The empirical confirmation is the leak gate (Rule 2).** If this construction
  leaked, the map would beat the model's honest out-of-fold error at training
  wells. It does not: pilot map-at-well MAD 5.43 m vs archived OOF MAD 5.64 m,
  and the lattice |R − obs| distribution matches the archived crossfit residuals
  (median 5.03 m) by construction.

One honest caveat: the *map* near training wells is of course informed by
training-well data — that is the point of a well-conditioned product, not
leakage. What makes it legitimate is that (a) accuracy measured at those wells
is the model's real out-of-fold accuracy, not an artifact, and (b) any external
accuracy claim comes from the OOF panel (Rule 4), evaluated on spatially
buffered holdout wells (Rule 3), never from sampling the map where it is best
informed.

Where "all wells at inference" *would* be — and was — leakage is when the raw
observations themselves are interpolated, so the feature at a well contains its
own label. That mode shipped in the July 2026 incident, is prohibited by Rule 1,
and is what the Rule-2 gate exists to catch.

The rules below are requirements, not guidance. A change that cannot satisfy them
does not render, does not score, and does not ship.

## 1. Feature-contract parity

Every well-derived input computed at inference time must be built under the same
exclusion scheme used to build that input at training time (crossfit /
leave-fold-out / leave-radius-out with a calibrated radius). Building a trained
feature from the full well set — including observations the model was trained
against — is prohibited, no matter how reasonable "use all the data at deployment"
sounds. Observation pinning is permitted only as an explicitly labeled post-hoc
assimilation layer with its own uncertainty accounting, never through the feature
path of a trained model.

*Incident tie-in: the lattice R was "upgraded" to all-well IDW on exactly that
reasoning; the checkpoints, calibrated to OOF-quality R, double-corrected and
printed 6-km-radius label bumps into flat terrain.*

The **preferred mechanism is crossfit-value interpolation**: interpolate the
archived leave-fold-out feature values themselves (never raw observations) onto
the render lattice with the training estimator's kernel and parameters. It is
exact by construction — at well locations it reproduces the training feature to
0.000 m, so the rendered map is the seamless spatial extension of the audited OOF
field — and it introduces no exclusion boundary, hence no seam artifacts.

Be precise about where the exclusion lives, because the two stages are easy to
conflate. All fold exclusion is in the archived **values**: each well's value was
computed only from wells outside its own CV fold. The lattice **interpolation**
then runs over the full well set with no exclusion of any kind — every well
participates, including the nearest one — and pins each well's archived value
exactly. That pinning is safe, and intended, precisely because the archived value
is not the observation: it carries the feature's honest out-of-fold error, so the
map contains no islands of impossible accuracy for a trained residual correction
to exploit. Do not "fix" the inference interpolator by adding fold or radius
exclusion on top of crossfit values — that re-introduces exclusion-boundary
artifacts while removing nothing. Leave-radius-out on raw observations
is the fallback where archived crossfit values do not exist; its radii must be
**calibrated, not guessed**: choose the radius whose residual distribution at the
training wells matches the archived crossfit feature (median and p90 of
|feature − obs|). A radius of zero must reproduce the all-well result exactly —
that is the sanity check that the exclusion is actually wired in. Know the
fallback's cost: a hard exclusion disk draws arc discontinuities where wells
cross its edge, concentrated around extreme-residual wells in sparse areas.

## 2. The leak gate is mandatory

Every inference/render run must sample its output at the training wells inside each
tile/basin and compare against the archived out-of-fold predictions. The leak
signature is two-part, and the gate requires both: the map beats its own OOF MAD at
training wells beyond a declared threshold, **and** the map departs from the OOF
predictions per-well (median |map − OOF| above a declared tolerance) — labels pull a
leaking map off the model's honest predictions toward the observations (5.29 m
median in the July 2026 incident, versus ~0.55 m for clean maps). A map failing both
**fails the run** — it is not a better map; it is a leaking one. The MAD ratio alone
can false-positive on small bimodal well panels where fold-median ensembling
legitimately beats the single held-out checkpoint (a diagnosed case: ratio 0.41 with
the R feature pinned to within 0.010 m of the archived values and zero wells
collapsed onto observations); a ratio trip that still tracks the OOF predictions
passes flagged for review. The gate result (n wells, map MAD, OOF MAD, per-well
tracking median, thresholds, status) is persisted with the run metadata.

*Incident tie-in: the mixture-identity and coarse-consistency gates both passed
throughout, because both sides of those comparisons shared the leak. Only a
map-versus-OOF comparison can catch this class.*

### 2a. Source-assimilation arms: the gate is informational, not waived

A `--source-edges` (assimilation) arm reads the observation pool at inference time
**by design** — a map cell sitting at a well reads that well, which is the product,
not a defect. Both gate signals therefore fire on intended behaviour, and the gate's
premise ("the map must not beat its own OOF at training wells") no longer holds. For
these arms the panel is still computed and persisted with the run metadata, but is
recorded with status `informational_source_arm` and never fails the run. Two
obligations replace the enforcement:

- **The assimilation pool must be declared** in the run metadata: source counts
  (bundle + admitted external), the external table's path, the kNN k, the median
  rank-0 source distance (km) and the same-basin fraction. A map whose accuracy comes
  from assimilated observations must say so, with the pool it assimilated.
- **Accuracy claims still come from held-out evaluation** (§3, §4) — the arm's
  own OOF, or wells outside the assimilated pool and spatially buffered from it.
  Map-at-well error for a source arm measures the pin, not skill.

## 3. "Held out" means spatially buffered

Validation of rasters against wells must exclude wells within the interpolation
support radius of any training well — not merely wells at identical coordinates.
Metrics must additionally be reported stratified by distance to the nearest
training well (km), so residual contamination is visible rather than averaged away.

*Incident tie-in: a coordinate-match-only "holdout" kept 99.8% of wells, most of
them inside training-well bulls-eyes, and the resulting report claimed the
comparison was "conservative" for the model. The claim was inverted.*

## 4. External claims quote OOF, not map-at-well

Headline accuracy numbers for any audience outside this project come from the
cross-validated out-of-fold panel (with its metric panel: MAD, bias and median
residual, RMSE, depth-banded and spatial strata, units on every number).
Render-sampled numbers are internal QA only, and are always labeled with the gate
result of Rule 2.

*Incident tie-in: every OOF number survived the audit unchanged; every
render-sampled number was inflated.*

## 5. Self- and nest-exclusion must be spatial

Identity keys do not survive cross-dataset joins: the same physical well appears
under different canonical IDs in monitoring and construction records. Any
"exclude the well's own record" guard must therefore be location-based, with an
exclusion radius no smaller than the geocoding jitter between record pairs.
Identity-based exclusion may be kept as a supplement, never as the mechanism.

*Incident tie-in: the drilled-depth feature's canonical-ID self-exclusion matched
0 of 34,503 training wells, while 88.5% had a same-site record under a different
ID <1 m away; only the radius clause was doing anything.*

## 6. Target-derived label screens are cross-fit by fold

Any screen that decides *whether a training label is admitted* by consulting the
observed target (here, well water-table elevations) is target-derived and is
subject to the same leave-one-fold-out contract as a trained feature — the screen
is just a feature that gates a loss weight instead of entering the forward pass.
The failure mode it prevents is subtle: a held-out well could make one of its own
fold's pseudo-labels "pass," so that label would train the model that is then
evaluated against that very well, importing the held-out observation through the
loss even though it never appears as a feature.

The rule, therefore:

- **Fold f's inclusion mask is computed from fold f's training wells only** (the
  `cv_fold != f` pool — the exact pool `crossfit_idw` uses for the regional prior
  R). A held-out well (`cv_fold == f`) can never influence which labels fold f
  trains on. The mask is per-fold (one bit per fold), not a single global flag.
- **Cross-border passing is legitimate.** A label in fold f's held-out HUC4s may
  still pass for fold f via training wells (`cv_fold != f`) within the search
  radius — that is training-side information only, exactly like R being informed
  by near neighbors across block boundaries.
- **Full pool at deployment.** At inference the screen may use every well —
  strictly more evidence than any CV fold had — mirroring the
  crossfit-at-wells / full-pool-at-inference pattern of Rule 1. A `*_full` mask
  computed from all wells is emitted for deployment and diagnostics, and is
  **never** used to weight a CV fold's training loss.
- **No evidence → no label.** Inclusion requires positive supporting evidence
  (a well within the radius that supports the label). Absence of wells is not
  evidence for the label; the conservative default is to drop it. A permissive
  "keep the label where we cannot check it" default injects exactly the regional
  bias this screen exists to remove.
- **The screen decides admission, not value.** The pseudo-label's value stays
  fixed from its target-blind source (here DTW = 0 from NHD + GSW shore evidence);
  the wells decide only whether that claim stands. Setting the value from well
  observations would be well-IDW — already a cross-fit feature — smuggled in as a
  label.

*Verification is mandatory and matches Rule 5's spirit: a unit test with synthetic
geometry where a held-out well would flip a pass bit must assert the fold-f bit is
unchanged; and the per-fold pass counts must differ across folds (a constant count
across folds would prove the mask collapsed to a global flag and lost its fold
awareness). Current implementation: `utils/build_glr_labels.py` (the GLR shoreline
screen) and `utils/train_conus_gnn.py --glr-labels` (per-fold shore-label weight);
tests in `tests/test_glr_labels.py`.*

## Scope and enforcement

- Applies to: inference runners, renderers, mosaic/COG builders, validation
  scripts, and any notebook or report that samples a well-trained surface.
- Current implementations: `utils/infer_conus_gnn.py` (`--r-source crossfit`
  default = Rule 1; per-basin leak gate, `--leak-gate-frac` = Rule 2) and
  `utils/validate_fac_gwx_wells.py` (`--holdout-buffer-km` + distance-to-training
  strata = Rule 3). Training-side crossfit features:
  `utils/build_conus_graph_inputs.py` (`crossfit_idw` / `crossfit_deep_idw`).
- Review checklist for any PR touching these paths: which rule does each
  well-derived input satisfy, and where is the Rule-2 gate invoked?
- Incident record and full technical detail: internal audit, July 2026
  (`notes/LEAKAGE_AUDIT.md`).
