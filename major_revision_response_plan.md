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

- Add FlowMOP ablation analysis for smoothing and parameter voting.
- Add parameter sensitivity analysis for PeacoQC and FlowCut to show that comparisons are not driven solely by default settings.

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

- Benchmark FlowMOP, PeacoQC, and FlowCut across increasing event counts and/or file counts.
- Include runtime and peak memory usage.
- Where practical, compare local execution with Dask/distributed execution.
- Present results as speed/scalability benchmarking rather than evidence of biological superiority.

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
   - benchmark runtime and peak memory;
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

