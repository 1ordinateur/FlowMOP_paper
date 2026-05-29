# Reviewer Comment Revision Matrix With Track-Change Options

## Legend

- 🔴 **Major comment / major revision**
- 🟠 **Targeted rewrite**
- 🟡 **Clarification or limitation**
- 🟢 **Already partly addressed**
- 🟦 **STAGED** - decision made; ready to apply to manuscript
- 🔵 **Reviewer comment**
- 🟣 **Suggested revision**

Track-change notation:

- <span style="color:#b00020">~~deleted text~~</span>
- <span style="color:#007a3d">added text</span>

## P01 - E1: Algorithmic Motivation

🔴 🔵 **Comment**

> The manuscript does not adequately motivate what deficiencies in FlowCut, PeacoQC, or related approaches FlowMOP was designed to address.

🟦 **Status: STAGED - prefer Option B.**

<details>
<summary><strong>Suggested revision options - Option B selected</strong></summary>

🟣 **Option A: concise Introduction replacement**

> <span style="color:#b00020">~~This paper proposes a novel cleaning algorithm, FlowMOP, that seeks to address these shortcomings.~~</span>
> <span style="color:#007a3d">This paper introduces FlowMOP as an integrated preprocessing workflow designed to address three limitations of existing approaches: most automated QC tools focus primarily on time-dependent acquisition artifacts; debris and doublet removal often remain manual or are handled by broader population-gating frameworks; and high-dimensional panels require safeguards against over-removal from single-channel artifacts.</span>

🟣 **Option B: more explicit comparative framing**

> <span style="color:#b00020">~~FlowMOP is the first algorithmic approach capable of debris removal, and the first algorithm to combine the three aforementioned cleaning steps into an integrated package.~~</span>
> <span style="color:#007a3d">FlowMOP was developed to combine automated time-gating, debris removal, and doublet exclusion in a single preprocessing pipeline. In contrast, FlowCut and PeacoQC primarily address time-dependent quality control, while broader automatic gating approaches such as GateNet and UNITO are aimed at population identification rather than standardized preprocessing cleanup.</span>

</details>

## P02 - E1 / R2.7: Dask, Parallelization, And Accuracy

🟢 🔵 **Comment**

> Clarify whether FlowMOP is intrinsically algorithmically suited to Dask/parallelization, and do not imply that Dask compatibility proves better gating accuracy.

🟦 **Status: STAGED - already addressed by recent manuscript changes.**

The recent revisions already separate Dask/scalability from gating accuracy and describe why FlowMOP's bin-summary algorithm is structurally compatible with chunked execution. No additional immediate rewrite is needed unless the benchmarking results require a more specific Results sentence.

<details>
<summary><strong>Suggested revision options - already addressed</strong></summary>

🟣 **Option A: implementation-safety wording**

> <span style="color:#b00020">~~This Dask-based design allows FlowMOP to run efficiently across distributed compute resources.~~</span>
> <span style="color:#007a3d">FlowMOP supports scalable execution through Dask-compatible, chunk-oriented computation. Computational scalability is evaluated separately from gating accuracy, as implementation speed does not by itself establish improved biological or analytical performance.</span>

🟣 **Option B: algorithm-structure wording**

> <span style="color:#b00020">~~FlowMOP is more vectorisable and more compatible with Dask's lazy, chunked execution model at an algorithmic level.~~</span>
> <span style="color:#007a3d">FlowMOP's time-gating algorithm reduces event-level measurements to per-bin, per-channel summaries before applying smoothing, MAD-based outlier detection, and channel voting. This regular map-reduce structure is naturally compatible with vectorised and distributed execution, while gating accuracy is evaluated independently.</span>

</details>

## P03 - E1 / R1.5: Smoothing Speculation

🔴 🔵 **Comment**

> The editor objects to the statement that FlowMOP's performance difference "may be attributed to smoothing" and asks for direct testing.

🟡 **Status: pending MAD smoothing ablation.**

This revision should be finalized after the MAD smoothing tests are available.

🟣 **Option A: pending ablation wording**

> <span style="color:#b00020">~~This difference may be attributed to FlowMOP's 'smoothing' implementation.~~</span>
> <span style="color:#007a3d">This difference is consistent with FlowMOP's use of both local and smoothed time-bin summaries, although the specific contribution of smoothing is evaluated separately in the ablation analysis.</span>

🟣 **Option B: post-ablation wording template**

> <span style="color:#b00020">~~This difference may be attributed to FlowMOP's 'smoothing' implementation.~~</span>
> <span style="color:#007a3d">In the MAD smoothing ablation, [removing/changing] the smoothing component [changed metric] in [scenario], supporting the interpretation that multi-resolution smoothing contributes to FlowMOP's performance under these acquisition perturbations.</span>

## P04 - E1: Expert Rankings Are Ambiguous

🔴 🔵 **Comment**

> Forced rankings only show relative preference and cannot show whether automated gates are adequate or inadequate in absolute terms.

🟦 **Status: STAGED - reframe ranking results and direct readers to quantitative downstream validation.**

Use this point to explain that forced human preference rankings are inherently limited because they either evaluate human opinion directly or attempt to reproduce human behavior. The revised manuscript should therefore treat the rankings as exploratory and point readers to synthetic ground-truth testing plus real-sample downstream population analyses.

<details>
<summary><strong>Suggested revision options - validation framing selected</strong></summary>

🟣 **Option A: Results reframe**

> <span style="color:#b00020">~~FlowMOP's outputs in these samples were benchmarked against a set of human experts.~~</span>
> <span style="color:#007a3d">FlowMOP's outputs in these samples were compared with expert-provided gates using a forced-ranking preference task. These rankings measure relative expert preference and should not be interpreted as an absolute measure of gating adequacy.</span>

🟣 **Option B: Discussion limitation plus validation pointer**

> <span style="color:#b00020">~~The subjective human rankings do though indicate that FlowMOP is an acceptable substitute for at scale automated debris and doublet gating.~~</span>
> <span style="color:#007a3d">The subjective human rankings suggest that FlowMOP's acceptability varies by dataset and task. Because forced rankings do not distinguish marginal preference from analytical inadequacy, and because human evaluation itself has inherent inter-operator limitations, we interpret these data as exploratory expert preference. Practical performance is therefore assessed alongside synthetic ground-truth tests and quantitative downstream analyses in real samples, including regression-based estimates of how preprocessing choice affects downstream population frequencies.</span>

</details>

## P05 - E1: Downstream Biological Impact

🔴 🔵 **Comment**

> The editor asks whether differences between FlowMOP, PeacoQC, and FlowCut materially affect downstream biological conclusions.

🟦 **Status: STAGED - fold into P04 and biological validation framing.**

This should be handled together with P04: expert rankings show preference, while downstream regression/population-frequency analyses address practical analytical impact in real samples.

<details>
<summary><strong>Suggested revision options - fold into P04</strong></summary>

🟣 **Option A: new Results subsection**

> <span style="color:#007a3d">To assess whether preprocessing differences altered downstream interpretation, FlowMOP, PeacoQC, and FlowCut gates were substituted for the corresponding preprocessing steps while subsequent biological gates were held constant. Regression models were then used to estimate the effect of preprocessing method on downstream population frequencies.</span>

🟣 **Option B: conservative Discussion wording**

> <span style="color:#007a3d">The practical relevance of preprocessing differences depends on whether they alter downstream biological conclusions. We therefore interpret event-removal differences alongside regression-based downstream population analyses in real samples rather than treating higher or lower event removal, or closer agreement with a single human gate, as inherently superior.</span>

</details>

## P06 - E1 / R2.10: Tumor Cell Validation

🔴 🔵 **Comment**

> Tumor digests are difficult, common samples, and the manuscript does not validate FlowMOP on them.

🟦 **Status: STAGED - add tumor-cell validation.**

We will run tumor-cell samples and add them as an additional biological validation dataset. The manuscript should use this to address the reviewer directly while still acknowledging that tumor-derived material can present difficult debris and aggregate structures.

<details>
<summary><strong>Suggested revision options - tumor validation selected</strong></summary>

🟣 **Option A: Results addition**

> <span style="color:#b00020">~~FlowMOP was evaluated on PBMC, liver, spleen, and related non-synthetic datasets.~~</span>
> <span style="color:#007a3d">FlowMOP was additionally evaluated on tumor-cell samples to test performance in a higher-debris biological setting. These samples were included because tumor-derived preparations can contain necrotic debris, aggregates, and heterogeneous scatter profiles that provide a more stringent test of automated debris and preprocessing behavior.</span>

🟣 **Option B: Discussion addition**

> <span style="color:#b00020">~~Future validation should specifically assess digested tumor samples and other high-debris tissues.~~</span>
> <span style="color:#007a3d">The added tumor-cell validation directly addresses a sample type expected to challenge automated debris removal. At the same time, tumor-derived material remains heterogeneous, and the results should be interpreted as validation in a difficult biological context rather than proof that a single FSC-A-centered debris strategy can resolve every tumor-digest debris or aggregate phenotype.</span>

</details>

## P07 - E1 / R2.14: Bimix And Trimix Simulation Meaning

🟠 🔵 **Comment**

> It is unclear how bimix/trimix samples represent realistic flow-rate or acquisition artifacts.

🟦 **Status: STAGED - use Option A with added micro-blockage rationale.**

<details>
<summary><strong>Suggested revision options - Option A selected</strong></summary>

🟣 **Option A: Methods clarification**

> <span style="color:#b00020">~~The synthetically generated Segment, Bimix, and Trimix samples sought to emulate the fluorescence fluctuations that time-gating should remove.~~</span>
> <span style="color:#007a3d">The Segment, Bimix, and Trimix samples were designed as complementary time-artifact stress tests. Segment samples model sustained acquisition shifts, whereas Bimix and Trimix samples model subtler mid-acquisition changes in event origin or fluorescence distribution that may be difficult to identify manually. While major blockages often produce segmental deviations, short self-resolving micro-blockages can generate smaller transient changes; the Bimix and Trimix simulations were designed to test detection of these less obvious distributional perturbations under controlled ground-truth conditions.</span>

🟣 **Option B: Discussion limitation**

> <span style="color:#007a3d">These mixed-source simulations should not be interpreted as reproducing every common flow-rate deviation. Rather, they provide controlled event-level ground truth for testing whether algorithms can detect subtle mid-acquisition distributional changes in addition to obvious start- or end-acquisition failures.</span>

</details>

## P08 - R1.1: Runtime And Memory Benchmarking

🔴 🔵 **Comment**

> Runtime and memory efficiency claims are not supported empirically.

🟡 **Status: pending benchmark results.**

The clone-based benchmark scripts now exist, but the manuscript wording should be finalized after the benchmarking data are generated and reviewed.

🟣 **Option A: Methods addition**

> <span style="color:#007a3d">Computational scalability was benchmarked using matched cloned FCS inputs generated from a real base file. Larger files were produced by concatenating event blocks while preserving the original within-file Time density and maintaining monotonic acquisition time. FlowMOP, PeacoQC, and FlowCut were run on identical inputs, and wall time and peak resident memory were recorded using `/usr/bin/time -v`.</span>

🟣 **Option B: Results addition**

> <span style="color:#007a3d">Across increasing event counts, runtime and peak memory were summarized for FlowMOP, PeacoQC, and FlowCut. These benchmarks evaluate computational scalability of the released implementations and are reported separately from gating accuracy metrics.</span>

## P09 - R1.2 / R2.22: FSC-A Debris Gating Concern

🟠 🔵 **Comment**

> FSC-A-only debris gating may miss larger debris, clumps, aggregates, tumor debris, or debris overlapping cells in FSC-A.

🟦 **Status: STAGED - clarify the deliberate conservative scope of FSC-A debris removal.**

<details>
<summary><strong>Suggested revision options - conservative FSC-A scope selected</strong></summary>

🟣 **Option A: Methods narrowing**

> <span style="color:#b00020">~~To debris gate, FlowMOP applies an FSC-A based threshold to exclude debris.~~</span>
> <span style="color:#007a3d">To debris gate, FlowMOP applies a conservative FSC-A-based threshold intended primarily to remove low-FSC debris. FSC-A was selected because low-forward-scatter material is a comparatively universal debris signal across sample types, whereas SSC-A patterns are more tissue- and instrument-dependent. This module is not intended to classify all possible debris morphologies.</span>

🟣 **Option B: Discussion rationale and limitation**

> <span style="color:#007a3d">FlowMOP's debris module deliberately targets the most generalizable debris phenotype: small, low-FSC events with undesirable staining characteristics. We do not assume that all debris can be identified from FSC-A alone. Larger aggregates or tissue-specific contaminants may resemble intact large cells, granulocytes, hepatocytes, or other biologically plausible populations in FSC/SSC space, and distinguishing these events generally requires domain knowledge or sample-specific gating. In such settings, SSC-A, pulse-width, margin-event metadata, or multi-parameter classifiers may provide useful extensions, but they are not treated here as universal debris criteria.</span>

</details>

## P10 - R1.3: Ranking Terminology

🟠 🔵 **Comment**

> "Benchmarking" is inappropriate for subjective expert preference rankings.

🟦 **Status: STAGED - use Option B.**

<details>
<summary><strong>Suggested revision options - Option B selected</strong></summary>

🟣 **Option A: terminology replacement**

> <span style="color:#b00020">~~FlowMOP's outputs in these samples were benchmarked against a set of human experts.~~</span>
> <span style="color:#007a3d">FlowMOP's outputs in these samples were compared with expert-provided gates.</span>

🟣 **Option B: figure-caption replacement**

> <span style="color:#b00020">~~Table showing ranking preferences of each cleanup method...~~</span>
> <span style="color:#007a3d">Table showing expert preference rankings for gates provided by each cleanup method or human operator...</span>

</details>

## P11 - R1.4: "Not Least Preferred" Interpretation

🟠 🔵 **Comment**

> The manuscript overinterprets FlowMOP not being least preferred as superiority to a human benchmarker.

🟦 **Status: STAGED - remove the claim.**

<details>
<summary><strong>Selected revision - delete overinterpretation</strong></summary>

> <span style="color:#b00020">~~FlowMOP was not the least preferred, indicating superiority to at least one human benchmarker.~~</span>

</details>

## P12 - R1.5: Biological Cost Of Errors

🟠 🔵 **Comment**

> Discuss whether it is worse to leave artifacts or remove rare biological populations.

🟦 **Status: STAGED - use Option A, with stronger time-gating context.**

<details>
<summary><strong>Suggested revision options - Option A selected</strong></summary>

🟣 **Option A: Discussion insertion**

> <span style="color:#007a3d">The biological cost of preprocessing errors is difficult to measure directly, and neither under-cleaning nor over-cleaning is preferable. Under-cleaning may allow acquisition-time artifacts with abnormal staining patterns to confound downstream results, including by creating, inflating, or obscuring apparent rare populations. Conversely, over-cleaning could remove rare or transient biological populations. Because FlowMOP's time-gating module operates on acquisition-time structure rather than population identity, rare populations are not expected to be systematically biased unless they are temporally confounded with an acquisition artifact.</span>

🟣 **Option B: link to downstream analysis**

> <span style="color:#007a3d">We therefore interpret sensitivity and specificity alongside downstream population analyses, because the most relevant question is whether preprocessing changes biological conclusions rather than whether one method removes more events.</span>

</details>

## P13 - R1.6 / R2.21: Parameter Sensitivity And Voting Ablation

🔴 🔵 **Comment**

> Show that results are not due to default parameters and show what happens without parameter voting.

🟦 **Status: STAGED - use Option B and explicitly explain default-parameter choice.**

We will not do extensive parameter tuning for FlowCut or PeacoQC. The comparison should use recommended/default settings for all methods, with FlowMOP also kept at fixed parameters, and use the MAD smoothing analysis as the targeted sensitivity analysis.

<details>
<summary><strong>Suggested revision options - Option B selected</strong></summary>

🟣 **Option A: analysis description**

> <span style="color:#007a3d">We performed sensitivity analyses across FlowMOP MAD smoothing settings and, where feasible, across PeacoQC and FlowCut parameters. We also evaluated the effect of channel-voting thresholds on event removal to assess whether voting prevents over-removal in high-dimensional panels.</span>

🟣 **Option B: limitation if incomplete**

> <span style="color:#007a3d">All algorithms were compared using recommended or default settings, including fixed FlowMOP parameters, to reflect typical unsupervised use. We did not perform extensive parameter tuning for FlowCut or PeacoQC because the original method descriptions do not provide dataset-specific guidance for how such tuning should be performed. Instead, we evaluate FlowMOP's sensitivity to the smoothing parameter in the MAD smoothing analysis, while treating full cross-method parameter optimization as outside the scope of this validation.</span>

</details>

## P14 - R2.1: "First Automated Debris Cleaning" Claim

🟠 🔵 **Comment**

> GateNet, UNITO, and other automatic gating methods may remove debris, so the "first automated approach capable of debris cleaning" claim is too broad.

🟦 **Status: STAGED - use Option A and distinguish headless preprocessing from supervised/annotated gating.**

<details>
<summary><strong>Suggested revision options - Option A selected</strong></summary>

🟣 **Option A: narrow novelty claim**

> <span style="color:#b00020">~~FlowMOP is the first automated approach capable of debris cleaning.~~</span>
> <span style="color:#007a3d">FlowMOP is, to our knowledge, the first integrated preprocessing workflow to combine automated time-gating, debris removal, and doublet exclusion in a single headless tool. Unlike broader automated gating frameworks that require user-defined training examples or manually annotated gates, FlowMOP is designed to operate without task-specific training or manual gate annotation.</span>

🟣 **Option B: distinguish tool classes**

> <span style="color:#b00020">~~the first algorithmic approach targeted at the removal of debris~~</span>
> <span style="color:#007a3d">a preprocessing-focused debris removal module, distinct from broader automatic gating frameworks that may classify debris-like populations as part of general population identification.</span>

</details>

## P15 - R2.2-R2.4: Abstract Structure And Terms

🟠 🔵 **Comment**

> The Abstract should start with background, define temporal artifacts, and avoid unexplained technical terms.

🟦 **Status: STAGED - use Option B.**

<details>
<summary><strong>Suggested revision options - Option B selected</strong></summary>

🟣 **Option A: simplified validation sentence**

> <span style="color:#b00020">~~Validation employed synthetic controls... (concatenated and mixed time perturbations, high-debris mixtures, and CFSE/CTV co-labeled doublets).~~</span>
> <span style="color:#007a3d">Validation used synthetic datasets with event-level ground truth for acquisition-time artifacts, debris enrichment, and doublet enrichment, together with expert comparison on biological datasets.</span>

🟣 **Option B: temporal artifact definition**

> <span style="color:#007a3d">Temporal artifacts are acquisition-time-dependent deviations in event quality or fluorescence signal caused by blockages, bubbles, flow instability, or instrument instability.</span>

</details>

## P16 - R2.5 / R2.25 / R2.26: Doublet And Aggregate Assumptions

🟠 🔵 **Comment**

> FlowMOP's FSC-A/FSC-H doublet method may fail with saturated scatter channels, aggregates, or same-label CTV/CFSE doublets.

🟦 **Status: STAGED - clarify correct acquisition/parameter assumptions.**

FlowMOP assumes that the relevant acquisition parameters have been correctly configured and measured. If voltages or scatter settings are incorrect enough to saturate or collapse the relevant channels, no preprocessing method can reliably recover the lost measurement information.

<details>
<summary><strong>Suggested revision options - acquisition assumption selected</strong></summary>

🟣 **Option A: doublet limitation**

> <span style="color:#007a3d">FlowMOP's doublet module assumes that FSC-A/FSC-H and SSC-A/SSC-H ratios remain informative and that acquisition voltages and scatter parameters have been set appropriately. If relevant scatter channels are saturated, edge-collapsed, or incorrectly configured at acquisition, the lost pulse-shape information cannot be recovered by FlowMOP or by other downstream preprocessing algorithms; such samples require acquisition review, manual intervention, or alternative pulse-shape features where available.</span>

🟣 **Option B: validation limitation**

> <span style="color:#007a3d">CTV-CFSE double-positive events provide an observable ground-truth doublet class, but same-label CTV-CTV and CFSE-CFSE doublets are not directly detectable in this validation design.</span>

</details>

## P17 - R2.6 / R2.9: Related Tools And Introduction Ending

🟠 🔵 **Comment**

> The Introduction does not adequately introduce related tools and lacks a summary paragraph.

🟦 **Status: STAGED - use Option B.**

<details>
<summary><strong>Suggested revision options - Option B selected</strong></summary>

🟣 **Option A: related-tools paragraph**

> <span style="color:#007a3d">Existing tools address different parts of this problem. flowAI, flowClean, FlowCut, and PeacoQC focus primarily on acquisition quality or time-dependent signal abnormalities, whereas GateNet, UNITO, and related automatic gating frameworks address broader population identification. FlowMOP differs by targeting integrated preprocessing cleanup across time artifacts, debris, and doublets.</span>

🟣 **Option B: final Introduction paragraph**

> <span style="color:#007a3d">Here, we introduce FlowMOP, an automated preprocessing tool for time-gating, debris removal, and doublet exclusion. We compare time-gating performance with PeacoQC and FlowCut, evaluate debris and doublet removal using synthetic ground-truth datasets and expert comparison, assess downstream biological impact, and benchmark computational scalability.</span>

</details>

## P18 - R2.8: Validation Novelty

🟡 🔵 **Comment**

> Simulation and real data are commonly used for validation, so the manuscript should not imply this is uniquely novel.

🟣 **Option A: remove novelty implication**

> <span style="color:#b00020">~~Conversely, this paper seeks to compare performance not only through traditional human-expert defined standards, but also the development of bespoke data generated explicitly for pre-processing validation.~~</span>
> <span style="color:#007a3d">We combine synthetic datasets with event-level ground truth and real biological datasets to evaluate complementary aspects of preprocessing performance.</span>

🟣 **Option B: emphasize ground truth**

> <span style="color:#007a3d">The synthetic datasets are used not because simulated validation is unique, but because they provide event-level labels for estimating sensitivity and specificity in preprocessing tasks where real ground truth is otherwise difficult to define.</span>

## P19 - R2.11-R2.13: Methods Completeness

🟠 🔵 **Comment**

> Remove duplicated acquisition sentence; list antibodies; include voltage/acquisition settings.

🟣 **Option A: clean acquisition wording**

> <span style="color:#b00020">~~Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer. . Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.~~</span>
> <span style="color:#007a3d">Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.</span>

🟣 **Option B: add missing panel/settings detail**

> <span style="color:#007a3d">The PBMC staining cocktail included [marker/clone/fluorophore/supplier]. Acquisition settings, including detector voltages where available, are provided in Supplementary Table [X].</span>

## P20 - R2.15: High/Low Debris Combination

🟠 🔵 **Comment**

> The method does not explain how high-debris and low-debris samples were combined.

🟣 **Option A: event-balanced wording**

> <span style="color:#b00020">~~For assessment of FlowMOP gating performance, 'high debris' and 'low debris' samples were synthetically combined.~~</span>
> <span style="color:#007a3d">For assessment of FlowMOP debris-gating performance, high-debris and low-debris samples were combined by sampling [N/%] events from each source and concatenating them into matched synthetic mixtures while retaining source labels for ground-truth quantification.</span>

🟣 **Option B: randomized-mixture wording**

> <span style="color:#007a3d">The combined debris files were generated by pooling high-debris and low-debris events at [ratio], randomizing event order, and retaining the source identity of each event to calculate post-gating enrichment of the low-debris component.</span>

## P21 - R2.16: Bayesian Modelling Placement

🟠 🔵 **Comment**

> Bayesian modelling is misplaced and its purpose is unclear.

🟣 **Option A: section move**

> <span style="color:#b00020">~~### Bayesian Modelling~~</span>
> <span style="color:#007a3d">### Statistical Analysis Of Expert Preference Rankings</span>

🟣 **Option B: purpose sentence**

> <span style="color:#007a3d">Bayesian modelling was used to summarize relative expert preference rankings across gate providers; it was not used to define objective ground-truth gating quality.</span>

## P22 - R2.17: Event Removal And 5% Threshold

🟠 🔵 **Comment**

> Report how many events are removed and justify the 5% threshold.

🟣 **Option A: supplementary table**

> <span style="color:#007a3d">Supplementary Table [X] reports the number and percentage of events removed by each preprocessing step for each dataset and method.</span>

🟣 **Option B: threshold rationale**

> <span style="color:#007a3d">The 5% threshold was selected as a conservative default to avoid triggering exclusion on very small event fractions; sensitivity to this threshold is reported in Supplementary Figure/Table [X] where available.</span>

## P23 - R2.18: `>10 Parameters` Voting Rule

🟡 🔵 **Comment**

> Explain why bins require two flagged parameters when more than 10 parameters are present.

🟣 **Option A: heuristic wording**

> <span style="color:#b00020">~~However, when >10 parameters are present, bins are rejected if they are flagged by two or more parameters.~~</span>
> <span style="color:#007a3d">For panels with more than 10 parameters, FlowMOP requires two or more parameters to flag a bin before rejection. This empirical safeguard reduces false-positive removal caused by isolated noisy channels in high-dimensional panels.</span>

🟣 **Option B: limitation wording**

> <span style="color:#007a3d">The 10-parameter threshold is an empirical default rather than a theoretically fixed boundary, and future work should optimize voting thresholds across panel sizes and staining designs.</span>

## P24 - R2.19 / R2.23: Figure 1 Labels

🟠 🔵 **Comment**

> Figure 1A/B/C lack axis titles and clear annotations.

🟣 **Option A: caption edit**

> <span style="color:#007a3d">Figure 1 has been revised to include x- and y-axis labels for all schematic plots, explicit labels for the smoothed and unsmoothed time-bin summaries, and a y-axis label for the doublet-ratio histogram.</span>

🟣 **Option B: response-letter wording**

> <span style="color:#007a3d">We revised Figure 1A-C to label all axes and clarify the two time-summary panels following time-binned fluorescence calculation.</span>

## P25 - R2.20: "Multiple Types Of Aberrations"

🟡 🔵 **Comment**

> Clarify what "multiple types of aberrations" means.

🟣 **Option A: replace phrase**

> <span style="color:#b00020">~~multiple types of aberrations~~</span>
> <span style="color:#007a3d">short spikes, sustained signal shifts, gradual drift, transient microblockages, and mixed-source acquisition irregularities</span>

🟣 **Option B: explanatory sentence**

> <span style="color:#007a3d">The short and long smoothing scales are intended to detect complementary artifact structures, including brief spikes and longer sustained shifts.</span>

## P26 - R2.24: Debris Removal Rate Against Expert

🔴 🔵 **Comment**

> Report debris removal rate against human expert gates.

🟣 **Option A: Results sentence**

> <span style="color:#007a3d">We additionally quantified FlowMOP debris removal relative to expert gates by calculating the proportion of expert-removed events also removed by FlowMOP and the proportion of FlowMOP-removed events outside the expert debris gate.</span>

🟣 **Option B: table description**

> <span style="color:#007a3d">Supplementary Table [X] reports expert-FlowMOP debris-gate agreement, including overlap, missed expert debris, and FlowMOP-only removal fractions.</span>

## P27 - R2.27: CTV/CFSE Protocol Detail

🟠 🔵 **Comment**

> The CTV/CFSE protocol is unclear, especially wash steps.

🟣 **Option A: Methods detail template**

> <span style="color:#007a3d">Cells were labelled with CTV or CFSE according to the manufacturer's instructions, quenched with [medium/serum], washed [number] times, recombined at [ratio], and incubated for 30 minutes at 37°C and 5% CO2 to enrich doublet formation.</span>

🟣 **Option B: supplement approach**

> <span style="color:#007a3d">A complete CTV/CFSE doublet-generation protocol, including dye concentrations, incubation conditions, quenching, wash steps, recombination ratio, and acquisition settings, is provided in Supplementary Methods.</span>

## P28 - R2.28: Multiple Scatter Channels

🟡 🔵 **Comment**

> Explain how FlowMOP could adapt to instruments with multiple scatter channels such as 405/488/polar scatter.

🟣 **Option A: limitation**

> <span style="color:#007a3d">The current implementation assumes canonical FSC/SSC channels for debris and doublet modules. Instruments with multiple scatter measurements may require user selection or algorithmic selection of the most informative scatter pair.</span>

🟣 **Option B: future extension**

> <span style="color:#007a3d">Future versions could extend FlowMOP to evaluate multiple scatter-channel pairs and select or combine the pair with the clearest debris/doublet separation.</span>
