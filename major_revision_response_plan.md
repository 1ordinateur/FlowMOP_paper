# FlowMOP Major Revision Response Plan

This document groups the editor, associate editor, and reviewer comments into shared revision items. Comment labels use:

- `E1`: Editor / Associate Editor comments
- `R1`: Reviewer 1 comments
- `R2`: Reviewer 2 comments

The goal is to convert this into a detailed response letter after the analyses and manuscript edits are completed.

## Editor Comment 1: Algorithmic Motivation and Comparative Framing

**Related comments:** `E1` algorithmic motivation and theoretical justification; `R1` comment 5; `R2` comments 1, 5, 6, 7, 8, 18, 20, 21, 22, 25, 28.

**Reviewer concern:** The manuscript does not yet clearly justify why FlowMOP was needed, what limitations of FlowCut, PeacoQC, or other automated gating tools it was designed to address, or which advantages are intrinsic to the algorithm rather than implementation choices. Reviewers also ask that related tools be introduced more clearly and that claims such as "first automated approach capable of debris cleaning" be revised.

**Proposed revisions:**

- Rework the Introduction to describe the current tool landscape, including FlowCut, PeacoQC, GateNet, UNITO, and broader automated gating approaches.
- Clarify that many existing automated gating or classification tools can remove debris-like populations, while FlowMOP is intended specifically as a preprocessing workflow integrating time, debris, and doublet removal.
- Add a clearer algorithmic rationale for each FlowMOP module:
  - time gating using multi-resolution smoothing and parameter voting;
  - debris gating using FSC-A thresholding as a conservative low-size debris filter;
  - doublet gating using FSC-A/FSC-H ratio structure.
- Distinguish the algorithmic contribution from the Dask/Python implementation contribution.
- Add explicit limitations for sample contexts where the current algorithm may be less suitable, including high-debris tumor digests, large morphological debris, high-scatter aggregates, and cytometers with multiple scatter configurations.
- Clarify the rationale for the `>10 parameter` voting threshold and revise if the threshold is empirical rather than theoretically derived.

**Additional analyses proposed:**

- Add FlowMOP MAD smoothing ablation analysis using the existing `FlowMOP/benchmarks/benchmark_flowmop_mad_smoothing.py` script. This now supports existing generated synthetic-combo FCS datasets through `--dataset-dir`, so the analysis can be run on the already-generated bimix/trimix/segment datasets rather than regenerating all inputs.
- Add parameter sensitivity analysis for PeacoQC and FlowCut where feasible to show that comparisons are not driven solely by default settings. This remains a manuscript-analysis task; the current committed benchmark infrastructure establishes API-correct baseline runners for both tools.
- Use the smoothing ablation results to replace speculative language such as performance differences that "may be attributed to smoothing" with direct evidence from on/off or grid-based smoothing comparisons.

**PeacoQC versus FlowMOP mechanistic framing to add:**

- Make the limitation being tested explicit: PeacoQC defines quality through the stability of density-peak positions across acquisition bins. This is well matched to acquisition artifacts that move marker peaks, but peak instability is not uniquely caused by poor acquisition quality.
- Link this to real use cases: bin-level peak composition can change because of random finite-bin sampling, uneven sample mixing, settling/stratification in the tube, differential uptake of sticky or fragile populations, transient bolus release after partial obstruction, carryover at acquisition start, or heterogeneous tissue suspensions. In these settings, a bin can have altered peak presence, peak height, or peak position even when the events are biologically valid.
- Link this directly to the synthetic results: the Bimix and Trimix benchmarks deliberately create temporally localized source-composition changes with event-level source labels. PeacoQC may detect real peak instability in those files, but under the benchmark ground truth, peak instability is not always equivalent to "events that should be removed." This explains the observed pattern: high sensitivity when altered bins are large and obvious, but reduced specificity in smaller-bin or three-source mixtures where ordinary bin-to-bin composition variation and true source perturbation are harder to distinguish.
- Use careful wording: do not say that PeacoQC is "wrong" to detect local peak instability. Instead state that it optimizes for a different quality concept, namely stable density peaks over acquisition time, whereas this benchmark scores source-label removal and target-source retention.

**Draft manuscript wording for PeacoQC comparison:**

> PeacoQC detects acquisition instability by identifying density peaks per channel and assessing whether those peak positions remain stable across acquisition bins. This approach is powerful when instrument or sample-flow artifacts produce marker-peak shifts. However, in heterogeneous samples, bin-level peak structure can also vary because the sampled cell composition changes over time, because of finite-bin sampling variation, uneven mixing, sample settling, transient bolus release, or differential uptake of specific populations. The Bimix and Trimix benchmarks intentionally model this ambiguity by creating temporally localized source-composition changes with event-level source labels. In this setting, peak instability is a real signal, but it is not always equivalent to low-quality events under the source-label ground truth. This provides a plausible explanation for PeacoQC's strong sensitivity in larger-bin perturbations and reduced specificity in smaller or compositionally more complex mixtures. FlowMOP's benchmarked time-gating mode instead uses globally anchored positive thresholds and per-bin positive-fluorescence summaries, which may reduce over-removal when local peak structure varies because of composition rather than acquisition failure.

**Draft response-letter wording for PeacoQC comparison:**

> We agree that the manuscript should more clearly distinguish FlowMOP's behavior from PeacoQC rather than treating all comparator differences as a smoothing effect. We have revised the discussion to explain that PeacoQC defines high-quality acquisition as stability of density-peak positions over acquisition time. This is appropriate for many acquisition artifacts, but in heterogeneous cytometry samples peak structure can also change because of bin-level composition variation, sample settling, uneven uptake, transient bolus release, or stochastic sampling of rare populations. The Bimix and Trimix simulations were designed to test this ambiguity using event-level source labels. Thus, PeacoQC may detect genuine peak instability while still losing specificity under our source-label scoring, because not all peak-instability events are benchmark-labelled contaminants. We now explicitly connect this mechanism to the observed pattern of high sensitivity in larger-bin perturbations and reduced specificity in smaller-bin or trimix settings.

**Draft response-letter wording for FlowCut versus FlowMOP comparison:**

> We have also clarified that FlowMOP's advantage over FlowCut is mechanistically distinct from its difference with PeacoQC. FlowCut is sensitive to acquisition-density and time-versus-fluorescence structure, which is valuable when flow-rate disturbances are coupled to signal abnormalities, but can remove valid events when acquisition rate changes without a corresponding fluorescence-quality defect. FlowMOP's time-gating module excludes Time, FSC, SSC, and source-label channels from the marker set and tests fluorescence summaries of globally defined positive populations across acquisition order. To support this interpretation, we added a matched mechanism benchmark that preserves the raw synthetic-combo fluorescence and source composition while changing only the Time channel. This tests whether each method responds to source-linked fluorescence/composition structure, local acquisition-rate structure, or both.

**Comprehensive mechanism experiment to delineate FlowCut versus FlowMOP:**

- Build the primary mechanism benchmark from the existing source-labelled smallcut synthetic-combo files rather than from newly simulated fluorescence perturbations. The Bimix and Trimix files already contain source-linked fluorescence/composition differences via `SampleIDInt` and have approximately normalized Time from synthetic-combo construction; Segment files preserve stronger source/time-density structure. This makes the benchmark directly interpretable: only the Time channel is experimentally changed.
- Use 15 high-count inputs: five Bimix, five Trimix, and five Segment files. Each benchmark input contains 500,000 acquisition-order-preserving events. For Bimix and Trimix, use the first 500,000 events because the sources are already shuffled through the acquisition. For Segment files, use a contiguous 500,000-event window centered on the source transition so that both segment sources are present.
- Generate three matched variants for every file:
  - raw: unchanged events and unchanged Time;
  - source-time-warped: multiply local Time increments by source identity while leaving fluorescence, scatter, labels, and event order unchanged;
  - random-time-warped: apply the same Time-increment multipliers to contiguous 2,000-event chunks independently of source identity.
- Use 1.0x and 2.0x local Time multipliers for two-source Bimix/Segment files, and 1.0x, 1.5x, and 2.0x for three-source Trimix files. Rescale the final Time range to match the raw input so that the perturbation tests local acquisition-rate structure rather than total acquisition duration.
- Run FlowMOP, FlowCut, and PeacoQC on exactly the same generated FCS inputs. FlowMOP should use the current default MAD smoothing (`0.01,0.05`) and positive-geomean fluorescence summaries for the main mechanism run. Historical smoothing (`0.1,0.9`) remains a parameter-sensitivity comparison, not the main interpretation.
- Primary metrics should match the MAD-smoothing source-label scoring: filename proportions identify the target source or sources with the largest mixture proportion; sensitivity is retained target-source events divided by retained events; specificity is removed non-target-source events divided by removed events; balanced score is the mean of sensitivity and specificity. Also report removal fractions and deltas relative to the matched raw variant.
- Expected discriminator:
  - if FlowCut's weakness is rate/density sensitivity, it should show changed removal under Time-only source or random warping despite unchanged fluorescence values;
  - if FlowMOP's advantage is fluorescence/population-summary anchoring, it should be comparatively less affected by Time-only warping while retaining source-label specificity;
  - if PeacoQC's weakness is peak-stability sensitivity to composition changes, it may react to real source-linked peak instability even when that instability is not labelled as low-quality under the source-label truth.
- Present this as a mechanistic supplement, not as another broad leaderboard. The goal is to explain when each algorithm's signal is appropriate: FlowCut for density/time-linked abnormalities, PeacoQC for unstable marker-density peaks, and FlowMOP for fluorescence-population deviations that are less coupled to acquisition rate alone.

## Editor Comment 2: Algorithmic Versus Implementation Advantages

**Related comments:** `E1` algorithmic versus implementation advantages; `R1` comment 1; `R2` comment 7.

**Reviewer concern:** The manuscript emphasizes the Dask framework, but it is unclear whether FlowMOP's performance advantages are due to algorithmic design or simply to implementation and parallelization. Reviewers request runtime and memory benchmarking.

**Proposed revisions:**

- Reframe Dask as a scalability and implementation advantage, not direct evidence of improved gating accuracy.
- Add text explaining whether FlowMOP's workflow is naturally parallelizable at the file, sample, and event-matrix level.
- Avoid implying that Dask itself makes the algorithm more accurate.
- Explicitly state which reported advantages are:
  - algorithmic, such as preprocessing module design, smoothing, and voting;
  - computational, such as distributed execution and memory handling.

**Additional analyses proposed:**

- Benchmark FlowMOP, PeacoQC, and FlowCut across increasing event counts using the committed `FlowMOP/benchmarks/benchmark_qc_algorithms.py` script.
- Use the new clone-based scaling mode rather than purely random synthetic data where possible:
  - provide a real base FCS file using `--base-fcs`;
  - create larger matched inputs by concatenating/tile-cloning events;
  - preserve the original within-file `Time` density pattern while offsetting repeated blocks forward so acquisition time remains monotonic;
  - run all algorithms on the exact same cloned FCS inputs.
- Include runtime and peak memory usage from `/usr/bin/time -v`, summarized in `results.csv`, `summary.csv`, and `summary.md`.
- Run PeacoQC and FlowCut through generated Rscript wrappers. The wrappers have been checked against the current upstream APIs:
  - PeacoQC accepts channel indices or names via `channels`; we pass 1-based channel indices.
  - flowCut documents `Channels` as a vector of channel indices; we pass 1-based channel indices.
  - PeacoQC is run with `plot = FALSE`, `save_fcs = FALSE`, `report = FALSE`, and `output_directory = NULL`.
  - flowCut is run with `Plot = "None"`, `AllowFlaggedRerun = FALSE`, and `Verbose = FALSE`.
- Use the committed PBS wrapper `FlowMOP/benchmarks/qsub_qc_algorithm_clone_scaling.pbs.sh` for the cluster run. It requests 24 CPUs and can run:
  - `flowmop`;
  - `peacoqc`;
  - `flowcut`.
- Use `FlowMOP/benchmarks/qsub_mad_smoothing_existing_data.pbs.sh` for the MAD smoothing ablation on existing generated synthetic-combo datasets. This also requests 24 CPUs.
- Present results as speed/scalability benchmarking rather than evidence of biological superiority.
- In the manuscript, explicitly separate these implementation/scalability results from algorithmic performance claims.

## Editor Comment 3: Expert Rankings, Absolute Quality, and Downstream Biological Impact

**Related comments:** `E1` expert preference rankings and statistical interpretation; `R1` comment 3; `R2` comments 16, 24, 26.

**Reviewer concern:** Forced expert rankings are difficult to interpret because they only provide relative preference. A method ranked below human gating may still be adequate, or all automated methods may be unacceptable. The statistical model also has limited power with few experts.

**Proposed revisions:**

- Rename subjective "benchmarking" language to "expert comparison" or "expert evaluation".
- Reframe the Bayesian ranking analysis as exploratory expert preference, not a definitive measure of algorithmic superiority.
- Revise figure labels so the creator of the gate and the expert performing the ranking are unambiguous.
- Consider reversing the ranking scale or explaining it more clearly.
- Replace or supplement current ranking plots with more directly interpretable summaries, such as average score or absolute quality categories.
- Remove or soften contorted interpretations such as FlowMOP being "not least preferred" in selected datasets.

**Additional analyses proposed:**

- Add downstream biological impact validation by applying FlowMOP, FlowCut, and PeacoQC preprocessing while keeping later manual biological gates fixed.
- Add the planned regression analysis to quantify whether preprocessing method materially affects downstream population frequencies or conclusions.
- Add debris and doublet removal rates relative to expert gates.
- If feasible, collect or present an absolute quality assessment scale:
  - excellent gating;
  - adequate and suitable for analysis;
  - inadequate for analysis but not terrible;
  - unacceptable.

## Editor Comment 4: Dataset Complexity, Tumor Samples, and Scope of Claims

**Related comments:** `E1` smaller point on dataset complexity; `R2` comments 10, 22, 28.

**Reviewer concern:** The manuscript may overstate dataset complexity and does not validate FlowMOP on difficult but common samples such as digested tumor samples. Reviewers specifically question whether FSC-A-only debris gating will perform well on tumor samples, large debris, or complex scatter profiles.

**Proposed revisions:**

- Clarify that "datasets exceed human capabilities" primarily refers to file number, event count, and throughput burden, not necessarily the intrinsic complexity of a single scatter gate.
- Revise statements implying that analysis software itself makes debris or time gating more complex.
- State explicitly that FlowMOP's current debris module is designed for low-FSC debris removal and may be less effective for large debris, aggregates, necrotic tumor material, or non-biological high-scatter contaminants.
- Discuss how future versions could incorporate SSC-A, pulse width, margin events, or instrument-specific scatter channels.
- Discuss adaptation to instruments with multiple scatter measurements, such as 405 FSC/SSC, 488 FSC/SSC, and polar scatter configurations.

**Proposed response strategy for tumor datasets:**

- If no tumor dataset is added, do not ignore the point in the manuscript.
- Explain in the response letter that a sufficiently controlled tumor dataset was not available within the revision window.
- Add a manuscript limitation acknowledging that tumor digests are an important future validation case.
- Emphasize that the new downstream and regression analyses strengthen analytical validation within the datasets currently available.

## Editor Comment 5: Synthetic Time Perturbations and Simulation Strategy

**Related comments:** `E1` smaller point on bimix/trimix simulation and flow-rate variation; `R2` comments 3, 4, 14, 15, 17, 19, 20.

**Reviewer concern:** The synthetic time perturbation strategy is difficult to understand and may not map clearly onto common experimental flow-rate deviations. Reviewers ask for clearer definitions, rationale, visualization, and event-removal summaries.

**Proposed revisions:**

- Define "temporal artifacts" in the Abstract and Introduction.
- Introduce terms such as concatenated perturbations, mixed time perturbations, bimix, trimix, high-debris mixtures, and CFSE/CTV doublets before using them.
- Rewrite the Methods section describing synthetic time sample generation.
- Add an illustrative figure or schematic showing the three synthetic time perturbation strategies.
- Explain why each synthetic perturbation was designed and which real acquisition artifact it approximates.
- Clarify whether the simulated fluctuations represent subtle mid-acquisition effects, beginning/end acquisition instability, or both.
- Add text acknowledging that some real flow-rate deviations are visually obvious, whereas the synthetic mixtures were designed to test subtler artifacts.
- Add event removal rates for the precleaning datasets and justify the 5% threshold.
- Correct missing axis titles and unclear subplots in Figure 1A.

**Additional analyses proposed:**

- Add a supplementary table or figure reporting event-removal proportions across synthetic and real datasets.
- Where possible, report how often subtle synthetic perturbations were detectable by human reviewers.
- For speed/scaling figures, use clone-based real-FCS scaling rather than purely random synthetic matrices. This avoids making the runtime/memory benchmark depend on unrealistic marker distributions while still allowing controlled event-count scaling.

## Editor Comment 6: Debris Gating Methodology and Validation

**Related comments:** `R1` comment 2; `R2` comments 15, 22, 24.

**Reviewer concern:** FSC-A-only debris gating may be insufficient for difficult samples because debris can overlap with cells on FSC-A or appear as large debris, aggregates, clumps, or tumor-associated material. Reviewers request clearer methods and debris removal rates against human expert gates.

**Proposed revisions:**

- Clarify exactly how high-debris and low-debris samples were synthetically combined.
- Present FSC-A debris gating as a conservative low-FSC debris removal strategy rather than a universal debris solution.
- Add limitations explaining that SSC-A, pulse width, and margin-event metadata may improve debris discrimination in complex samples.
- Discuss that large-cell clumps, tumor debris, and high-scatter aggregates may not be captured by FSC-A thresholding alone.

**Additional analyses proposed:**

- Add debris removal rate relative to human expert gates.
- If available in existing data, compare debris classification against FSC/SSC manual gates.
- Add sensitivity or failure-case discussion for samples where debris and cells overlap strongly in FSC-A.

## Editor Comment 7: Doublet Gating Methodology and Validation

**Related comments:** `R2` comments 5, 25, 26, 27.

**Reviewer concern:** FlowMOP's FSC-A/FSC-H ratio approach assumes cells fall within the relevant FSC measurement range. Reviewers note that saturated or edge-collapsed populations could distort the ratio histogram. They also ask whether FlowMOP can handle aggregates and whether CTV-CFSE double-positive cells miss CTV-CTV or CFSE-CFSE doublets.

**Proposed revisions:**

- Clarify whether FlowMOP's doublet module is intended to remove classical doublets/aggregates and where it may fail.
- Add limitations for saturated FSC measurements, edge-collapsed populations, large myeloid cells, and highly heterogeneous scatter profiles.
- Clarify the CTV/CFSE doublet-generation protocol, including wash steps and sample preparation details.
- Explain that CTV-CFSE double positives provide a measurable doublet class but do not capture same-label doublets.

**Additional analyses proposed:**

- Add doublet removal rate relative to expert gates.
- Where possible, assess whether same-label doublets are expected to bias the validation.
- Provide the detailed CTV/CFSE protocol in the supplement if space is limited.

## Editor Comment 8: Error Costs, Sensitivity, and Practical Significance

**Related comments:** `R1` comment 4; `R1` comment 5; `E1` downstream biological impact concern.

**Reviewer concern:** Sensitivity/specificity trade-offs need biological interpretation. The manuscript should explain whether it is worse to leave transient artifacts in the data or to remove rare meaningful populations. Reviewers also want evidence that observed performance differences are practically significant.

**Proposed revisions:**

- Expand the Discussion of false positive and false negative preprocessing errors.
- Explain that aggressive cleaning may remove rare or transient biological populations, while conservative cleaning may leave artifacts that distort downstream analysis.
- Avoid framing higher removal as inherently better.
- Interpret FlowMOP, FlowCut, and PeacoQC differences in terms of downstream analytical consequences, not only event counts.

**Additional analyses proposed:**

- Use the planned regression analysis to test whether preprocessing method significantly shifts downstream population frequencies.
- Report effect sizes, confidence or credible intervals, and practical interpretation rather than only significance.
- Add parameter sensitivity analysis for competing methods to assess robustness across plausible settings.
- Use the clone-based scaling benchmark only to address computational feasibility and practical runtime/memory significance. Do not use speed differences as a proxy for biological correctness.

## Editor Comment 10: Benchmarking Implementation Now Available

**Related comments:** `E1` algorithmic versus implementation advantages; `R1` comment 1; `R1` comment 5; `R2` comment 7.

**Reviewer concern:** Runtime and memory claims need empirical support, and comparisons to PeacoQC and FlowCut must be technically fair despite those tools being implemented in R. The manuscript should avoid conflating Dask implementation advantages with algorithmic superiority.

**Current code now available:**

- `FlowMOP/benchmarks/benchmark_qc_algorithms.py`
  - Generates matched benchmark inputs.
  - Supports default synthetic FCS generation.
  - Now supports clone-based real-FCS scaling through `--base-fcs`.
  - Clone mode tiles a real base FCS to target event counts and preserves the original within-file `Time` density pattern while keeping time monotonic between repeated blocks.
  - Runs FlowMOP, PeacoQC, and FlowCut on matched inputs.
  - Captures wall time and peak RAM using `/usr/bin/time -v`.
  - Writes `benchmark_commands.txt`, `metadata.json`, `results.csv`, `summary.csv`, and `summary.md`.
- `FlowMOP/benchmarks/benchmark_flowmop_mad_smoothing.py`
  - Runs FlowMOP across MAD smoothing settings.
  - Can generate labeled synthetic inputs.
  - Can also consume existing generated synthetic-combo FCS datasets through `--dataset-dir`.
  - Scores time-gating sensitivity, specificity, and balanced score where source labels are available.
- `benchmarks/benchmark_rate_density_mechanism.py`
  - Runs the raw-file mechanism benchmark using 15 smallcut Bimix/Trimix/Segment FCS inputs.
  - Preserves fluorescence, scatter, source labels, and event order, and modifies only the Time channel for source-linked and random local Time-warp variants.
  - Runs FlowMOP, FlowCut, and PeacoQC on matched inputs and writes raw-delta summaries for removal fraction, sensitivity, specificity, and balanced score.
- `FlowMOP/benchmarks/qsub_qc_algorithm_clone_scaling.pbs.sh`
  - PBS wrapper for the FlowMOP/PeacoQC/FlowCut clone-scaling benchmark.
  - Requests 24 CPUs.
  - Requires `BASE_FCS`.
  - Allows override of `SIZES`, `REPEATS`, `ALGORITHMS`, `TIMEOUT`, `OUT_DIR`, and `ALLOW_MISSING`.
- `FlowMOP/benchmarks/qsub_mad_smoothing_existing_data.pbs.sh`
  - PBS wrapper for running the MAD smoothing ablation on existing generated synthetic-combo datasets.
  - Requests 24 CPUs.
  - Requires `DATASET_DIR`.
  - Allows override of dataset bin size, smoothing grid, output directory, file limit, and timeout.

**API alignment notes for response letter / methods:**

- PeacoQC API checked against the current Bioconductor/r-universe manual and source:
  - `PeacoQC(ff, channels, determine_good_cells = "all", plot, save_fcs, output_directory, name_directory, report, ...)`;
  - `channels` may be indices or names;
  - the benchmark passes 1-based channel indices.
- flowCut API checked against the current Bioconductor/r-universe manual and source:
  - `flowCut(f, Segment = 500, Channels = NULL, Directory = NULL, FileID = NULL, Plot, AllowFlaggedRerun, Verbose, ...)`;
  - `Channels` is documented as a vector of channel indices;
  - the benchmark passes 1-based channel indices.
- The benchmark intentionally disables plot/report/FCS output for PeacoQC and plotting for flowCut so measured runtime emphasizes algorithm execution rather than optional output generation.

**How this addresses the comments:**

- `R1` comment 1: provides the requested runtime and peak memory benchmarking table inputs across increasing dataset sizes.
- `E1` implementation-versus-algorithm concern: allows the manuscript to explicitly present Dask/Python as a scalability contribution while separately discussing algorithmic behavior.
- `R1` comment 5: provides infrastructure for FlowMOP MAD smoothing sensitivity/ablation and creates API-correct baseline wrappers for PeacoQC/FlowCut before parameter-sensitivity extensions.
- `R2` comment 7: supports revising the Introduction so Dask is described as computational infrastructure rather than the core gating algorithm.

## Editor Comment 9: Manuscript Structure, Methods Completeness, and Figure Corrections

**Related comments:** `R1` comment 3; `R2` comments 2, 3, 4, 9, 11, 12, 13, 16, 19, 23.

**Reviewer concern:** Several manuscript sections are hard to follow, some methods are incomplete, and some figures lack axis labels or clear annotations. Reviewers request structural editing by an experienced researcher.

**Proposed revisions:**

- Restructure the Abstract:
  - start with background and problem;
  - define temporal artifacts;
  - reduce unexplained technical terms;
  - state validation strategy and main findings more clearly.
- Add a final summary paragraph to the Introduction.
- Remove the duplicated Method line: "Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer."
- List the antibodies used in the PBMC staining cocktail.
- Add cytometer voltage settings or acquisition settings where available.
- Move or retitle the Bayesian modelling section so it is not incorrectly nested under "Non-synthetic samples".
- Begin the Bayesian modelling section by clearly stating its purpose.
- Correct missing axis titles in Figure 1A, Figure 1B, and Figure 1C.
- Update row-axis titles in ranking grids to "Gate Provided By" or equivalent wording.
- Consider replacing the scatter plot panel with an average-score column if this improves interpretability.

## Proposed High-Level Revision Package

The revision package should therefore contain four major workstreams:

1. **Manuscript alterations and reframing**
   - address related tools, algorithmic motivation, terminology, limitations, figure clarity, and methods completeness.

2. **Analytical and biological validation**
   - add downstream population-impact analysis;
   - include the planned regression analysis;
   - add debris and doublet removal rates relative to expert gates.

3. **Algorithmic robustness and sensitivity analysis**
   - ablate FlowMOP smoothing and parameter voting;
   - tune PeacoQC and FlowCut parameters across reasonable ranges;
   - revise claims based on observed results.

4. **Speed and scalability benchmarking**
   - benchmark runtime and peak memory using clone-based real-FCS scaling where possible;
   - run FlowMOP, PeacoQC, and FlowCut through API-checked wrappers;
   - compare local and Dask/distributed execution where feasible;
   - clearly separate computational scalability from gating accuracy.

## Response Letter Positioning

The response letter should make clear that the revision directly addresses the central editorial concerns:

- FlowMOP's algorithmic motivation is now explained more explicitly.
- Implementation advantages are separated from algorithmic claims.
- Expert rankings are reframed and supplemented with more interpretable analytical validation.
- Regression analysis is added to test downstream biological impact.
- Benchmarking is added to support speed and scalability claims.
- Limitations are acknowledged for tumor samples, complex debris, FSC saturation, and instrument-specific scatter configurations.
