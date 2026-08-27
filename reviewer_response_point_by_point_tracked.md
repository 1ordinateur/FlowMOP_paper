# Response to the Associate Editor and Reviewers

**Tracked against `7cb7dabfd6aa2a746a7ed13493ec6b1ac67934e2` (repository state on 24 August 2026; latest commit then was 7cb7dab from 19 August 2026).** <span style="color:#0066cc">Blue text was added.</span> <span style="color:#c00000"><s>Red struck-through text was removed.</s></span>


**Manuscript:** *FlowMOP: An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool*

The Associate Editor and reviewers raised several overlapping concerns. To avoid repeating the same response while ensuring that every point is addressed, closely related comments are reproduced verbatim and answered together below. A complete comment-coverage index is provided at the end of this response.

We thank the Associate Editor and reviewers for their considered feedback and believe that the requested amendments have significantly improved the manuscript. We hope that the revised manuscript now addresses the Associate Editor's and reviewers' concerns.

## Data Leakage: Correction to the Comparative Benchmark

During revision, we identified a data-leakage error in the original competitor benchmark: the source-label field retained for scoring had inadvertently been available to PeacoQC and FlowCut. The synthetic FCS files deliberately retained `SampleIDInt` so that event-level source truth remained attached for scoring, but the original PeacoQC and FlowCut benchmark paths did not remove this numeric field from the channels available for quality assessment. FlowMOP did not have this issue because its fluorescence-channel selection explicitly excluded channel names containing `sample`, together with Time and scatter channels.

We reran the full benchmark with the source label excluded from all quality-control inputs and used it only for scoring. The correction reinforced rather than reversed the comparative findings. FlowMOP had higher sensitivity than FlowCut for Segment at both tested bin sizes and for the 5000-event Bimix and Trimix benchmarks. At 5000 events, FlowMOP also had higher specificity than both competitors across Segment, Bimix, and Trimix; at 2000 events, it retained higher specificity than PeacoQC in Bimix and Trimix without a significant sensitivity disadvantage and had higher sensitivity and specificity than FlowCut in Segment. Figure 2, its violin distributions, p-values, significance brackets, and all reported comparisons now use these corrected outputs.

This correction is independent of the specific reviewer comments addressed below and is reported for transparency.

## Associate Editor's General Assessment

> Novel tools to expedite flow cytometry data analysis are always welcome, and I commend the authors for their effort in introducing these new methods. However, as currently submitted, the manuscript requires major revision. The reviewers have identified numerous areas requiring improvement, and I encourage the authors to address their comments carefully and comprehensively. In addition to the reviewers' critiques, I offer several observations of my own that I believe will strengthen the manuscript's suitability for publication.

**Response:** The revisions are organized around the algorithmic motivation, comparative mechanism, computational benchmarking, interpretation of expert evaluations, methodological clarity, and the limits of the current approach. Detailed responses follow.

## Combined Comment 1 — Algorithmic Motivation, Novelty, Related Tools, and Validation Framing

**Raised by:** Associate Editor; Reviewer 2, Comments 1, 6, and 8.

### Associate Editor — Algorithmic Motivation and Theoretical Justification

> The authors acknowledge the existence of other algorithms that perform substantially similar analyses to the components of FlowMop, yet they do not adequately motivate what deficiencies in the existing algorithms their new approach is designed to address. A compelling framing would follow a structure such as: we demonstrated that PeacoQC and FlowCut fail under realistic conditions X, Y, and Z; upon analysis, we determined this is attributable to specific aspects of how those algorithms operate; accordingly, we designed our algorithm to overcome these limitations. No such explanation is currently provided, nor is there a clear justification for why the particular algorithmic choices were made.

### Reviewer 2, Comment 1

> In the abstract, the authors claim “FlowMOP is the first automated approach capable of debris cleaning”. However, there are methods like GateNet, 10.1002/cyto.a.20531, and UNITO that can do automatic gating and probably can be used to gate out debris. There is even a review for this purpose (10.3389/fimmu.2015.00380).

### Reviewer 2, Comment 6

> The introduction did not introduce available related tools.

### Reviewer 2, Comment 8

> The authors claimed, “Conversely, this paper seeks to compare performance not only through traditional human-expert defined standards, but also the development of bespoke data generated explicitly for pre-processing validation.” In fact, both simulation data and read data are commonly used for new method validation.

### Response

We now frame FlowMOP as a training-free preprocessing workflow that combines automated time, debris, and doublet removal. We revised the Introduction to distinguish automated population-gating tools, time-quality-control tools, and FlowMOP's intended role as an integrated preprocessing workflow. We removed the claim that FlowMOP is the first automated approach capable of debris cleaning. The revised <span style="color:#c00000"><s>Introduction discusses</s></span><span style="color:#0066cc">Abstract describes alternative preprocessing algorithms such as</span> FlowCut and <span style="color:#c00000"><s>PeacoQC</s></span><span style="color:#0066cc">PeacoQC, while the Introduction discusses them</span> as time-dependent quality-control approaches and <span style="color:#c00000"><s>GateNet</s></span><span style="color:#0066cc">distinguishes GateNet, UNITO,</span> and <span style="color:#c00000"><s>UNITO</s></span><span style="color:#0066cc">FATE</span> as broader <span style="color:#c00000"><s>automated population-gating</s></span><span style="color:#0066cc">population-identification or representation-learning</span> approaches.

We have also revised the description of the three FlowMOP modules. Time gating is based on positive-population fluorescence summaries across acquisition order, debris gating derives an FSC-A threshold from cross-parameter positive-population structure, and doublet gating uses FSC-A/FSC-H and SSC-A/SSC-H ratio structure. These algorithmic choices are now distinguished from computational implementation choices.

In response to Reviewer 2's observation that simulated and real data are both commonly used in method validation, we now describe the synthetic samples specifically as event-labelled technical controls developed to provide objective ground truth for <span style="color:#c00000"><s>flow-cytometry</s></span><span style="color:#0066cc">flow cytometry</span> preprocessing tasks.

### Changes made

- We removed the claim that FlowMOP is the first automated approach capable of debris cleaning and added the following clarification:

  > “Several tools automate cytometry population identification, including <span style="color:#0066cc">deep-learning approaches [7,8],</span> the model-based flowClust method <span style="color:#c00000"><s>[13],</s></span><span style="color:#0066cc">[9],</span> the neural-network method GateNet <span style="color:#c00000"><s>[14],</s></span><span style="color:#0066cc">[10],</span> and the bivariate-segmentation framework UNITO <span style="color:#c00000"><s>[15];</s></span><span style="color:#0066cc">[11], while FATE uses representation learning to produce generalized flow cytometry embeddings [12];</span> automated gating approaches are reviewed elsewhere <span style="color:#c00000"><s>[16].</s></span><span style="color:#0066cc">[13]. Conversely,</span> FlowMOP is a Python-based, training-free preprocessing workflow that combines automated time-gating, debris removal, and doublet exclusion in a single headless tool. FlowCut and PeacoQC provide the most direct comparison for its time-gating component.”

- We reframed the synthetic-data contribution as follows:

  > “The synthetic datasets provide event-level labels for estimating sensitivity and specificity in preprocessing tasks where real ground truth is otherwise difficult to define.”

- We added a concise description of the three algorithmic components to the Figure 1 legend:

  > “A) FlowMOP selects valid parameters using positive-peak detection, generates time-binned fluorescence summaries, applies two smoothing resolutions, and performs robust outlier rejection across acquisition order. B) FlowMOP applies the same valid-parameter check, derives a candidate FSC-A threshold from each eligible parameter’s positive events, and uses the median candidate as the final FSC-A gate. C) FlowMOP identifies doublets from inflection points in FSC-A/FSC-H and SSC-A/SSC-H ratio histograms, with a fixed-ratio fallback.”

**Location:** Introduction; Results, “Algorithmic design,” “Time Gating,” “Debris Gating,” and “Doublet Gating”; Figure 1 legend.

## Combined Comment 2 — Computational Benchmarking and Within-Sample Parallelisability

**Raised by:** Associate Editor; Reviewer 1, Comment 1; Reviewer 2, Comment 7.

### Reviewer 1, Comment 1 — Technical Infrastructure and Performance Claims

> The adoption of a Dask-based Python framework is a highly commendable architectural choice. Flow cytometry analysis has historically under-utilized distributed compute resources, and providing a modern, scalable alternative to R-based packages is a significant contribution to the “Big Data” era of spectral cytometry. However, the manuscript currently lacks empirical data to support the claims of improved runtime or memory efficiency, while other contemporary algorithms explicitly highlight their scalability for large datasets.
>
> Recommendation: To make a compelling case for FlowMOP’s adoption, the authors could include benchmarking figures (e.g., execution time and peak memory usage) comparing FlowMOP against PeacoQC and FlowCut across datasets of increasing file count and/or size (~10^3 to ~10^7 events or some [Event x Parameter] matrix varying each). Another point of comparison may be in the distributed compute scenario vs. more traditional local compute resources.
>
> Example Performance Benchmarking Table:
>
> | Dataset Size (Events) | Metric | FlowMOP (Python/Dask) | PeacoQC (R) | FlowCut (R) |
> | --- | --- | --- | --- | --- |
> | 10^4 | Execution Time (s) | | | |
> | | Peak RAM (MB) | | | |
> | 10^5 | Execution Time (s) | | | |
> | | Peak RAM (MB) | | | |
> | ...etc. | | | | |

### Associate Editor — Algorithmic vs. Implementation Advantages

> Reviewer 1 raises several points regarding the use of the Dask framework, and I would like to extend that line of inquiry. Specifically, I would ask the authors to clarify whether there is something inherent to the FlowMop algorithm that makes it particularly well-suited to parallelization via Dask — as distinct from the implementation itself. This distinction matters considerably. Given current development tools, a competent programmer with AI tooling could likely port the existing FlowCut or PeacoQC code — neither of which is especially lengthy — to a Dask-based parallel implementation in a matter of hours. If that is the case, it raises the question of whether a substantial portion of FlowMop's reported advantage derives from the implementation rather than from any intrinsic algorithmic innovation, which would significantly diminish the novelty of the contribution. I would encourage the authors to focus their comparative analysis on algorithmic advantages, particularly in cases where existing algorithms present genuine structural barriers to parallelization.

### Reviewer 2, Comment 7

> The authors should introduce “Dask” before using the “Dask-based framework” in the introduction. Please keep in mind, most readers probably have no experience using Python at all. I use Python a lot myself. Still, I am not familiar with this “Dask” thing. Also, the Dask-based design is about calculation efficiency, instead of accuracy. Readers will be more interested in which algorithm is used for automatic gating, instead of how to speed up calculation. The authors did not really introduce the algorithm for gating in the introduction. But they use a whole paragraph on the Dask-based design.

### Response

We evaluated computational performance using matched 36-channel FCS inputs containing 10,000, 100,000, 300,000, 1,000,000, and 2,000,000 events. FlowMOP, PeacoQC, and FlowCut were run on the same clone-based inputs. Optional plotting, reporting, and output generation were disabled where supported, and FlowMOP's debris and doublet modules were disabled so that the shared time-gating task was timed fairly. Each condition included one warm-up and three measured repeats. Runtime and peak resident memory are reported as mean ± SD in Table 1. At 2,000,000 events, FlowMOP was approximately 2.5-fold faster than PeacoQC and 3.5-fold faster than FlowCut while using approximately 32% and 45% of their respective peak RAM.

The benchmark used local non-distributed execution. Testing active Dask execution showed that its overhead did not improve this workload, so Dask was inactive for the reported benchmark and removed from the manuscript's novelty framing.

Among the three evaluated decision structures, only FlowMOP is inherently suited to independent within-sample channel parallelisation. Once shared acquisition bins have been defined, each eligible channel independently establishes its fluorescence reference, produces a fixed-shape time-bin summary, and applies smoothing and MAD filtering. Cross-channel coordination occurs only when the resulting bin flags are combined in the final parameter vote. This creates coarse, regular channel-level tasks with a single final reduction.

PeacoQC must instead reconcile peak configurations across bin-channel combinations before its MAD, Isolation Tree, and consecutive-bin decisions. FlowCut combines segment- and channel-level statistics through adjacent-segment and file-wide comparisons, contiguous-region decisions, and an optional flagged-file rerun. Both methods can process separate files in parallel, and some internal suboperations could be parallelised, but their complete within-sample decision pipelines contain intermediate dependencies that prevent the same independent channel-wise reduction without changing their decision rules. Reimplementation in Python or another scheduling framework would not remove those structural dependencies.

FlowMOP's within-sample parallelisability is therefore an inherent property of its decision structure rather than its programming language. The reported runtime benchmark used local non-distributed execution, so it measures the performance of the tested implementations rather than an empirical distributed-computing speedup.

### Changes made

- We removed Dask from the manuscript's novelty and accuracy framing and specified the benchmark configuration as follows:

  > “For fair timing of the shared time-gating task, FlowMOP was run using local non-distributed execution, with debris and doublet removal disabled and annotated output FCS writing disabled. PeacoQC and FlowCut were run with optional plotting, reporting, and output generation disabled where supported.”

- We state FlowMOP's contribution and identify the most direct time-gating comparators:

  > “FlowMOP is a Python-based, training-free preprocessing workflow that combines automated time-gating, debris removal, and doublet exclusion in a single headless tool. FlowCut and PeacoQC provide the most direct comparison for its time-gating component.”

- We retained a concise operational description in the Methods:

  > “FlowMOP accepts `.csv`, `.fcs`, and Parquet files. It reduces event-level measurements into time-bin and channel summaries, applies smoothing and robust outlier detection, and combines the resulting flags through parameter voting.”

- We placed the architectural comparison and its relationship to the computational benchmark in the Discussion:

  > “These measurements were obtained using local non-distributed execution and therefore demonstrate the performance of the tested implementations, not a distributed-computing speedup.”

  > “Among the three evaluated decision structures, only FlowMOP exposes an inherently channel-parallel within-sample computation. After shared acquisition bins are defined, each eligible fluorescence channel independently establishes its reference, produces a fixed-shape time-bin summary, and applies smoothing and MAD filtering; cross-channel coordination occurs only in the final parameter vote. PeacoQC instead performs data-dependent peak identification and reconciliation across bin-channel combinations, while FlowCut applies adjacent-segment, contiguous-region, file-wide, and conditional rerun decisions <span style="color:#c00000"><s>[5,9].</s></span><span style="color:#0066cc">[5,6].</span> Their suboperations and separate files can be parallelised, but their complete within-sample decision pipelines cannot be expressed as the same independent channel-wise reduction without changing their decision rules. Porting either implementation to another language or scheduling framework would therefore not remove these structural dependencies. This structural distinction, together with FlowMOP’s earlier data reduction, fewer full-data passes, and smaller intermediate state, is consistent with its lower runtime and memory use in our benchmarks, although the benchmark does not independently establish causation.”

- We added matched runtime and peak-memory testing across five event counts:

  > “Computational scalability was evaluated using clone-based real-FCS scaling. A representative FCS file was subsampled/replicated to matched event counts of 10,000, 100,000, 300,000, 1,000,000, and 2,000,000 events while preserving the original 36-channel structure. FlowMOP, PeacoQC, and FlowCut were run on the same generated inputs for each size.”

- We reported the principal computational result as follows:

  > “The computational benchmark demonstrates that FlowMOP provides speed gains with lower peak RAM usage at larger event counts. This is most evident from 300,000 events onward, where FlowMOP had both the fastest mean runtime and lowest peak memory use. At 2,000,000 events, FlowMOP was approximately 2.5-fold faster than PeacoQC and 3.5-fold faster than FlowCut, while using approximately 32% of PeacoQC’s peak RAM and 45% of FlowCut’s peak RAM.”

**Location:** Methods, <span style="color:#0066cc">“Computational scalability benchmark”; Results,</span> “Algorithmic design” and “Computational scalability”;<span style="color:#c00000"><s>Results, “Computational scalability”;</s></span> Table 1; Discussion, “Synthetic Sample Time Gating.”

## Combined Comment 3 — Downstream Biological Validation, Dataset Breadth, and the Cost of Errors

**Raised by:** Associate Editor; Reviewer 1, Comments 3 and 4; Reviewer 2, general assessment and Comment 10.

### Associate Editor — Downstream Biological Impact

> Alternatively — or in addition — the authors could evaluate the downstream biological impact of each gating strategy. By substituting FlowMop, FlowCut, and PeacoQC gates for the time and debris gating steps while retaining the human gating strategy for all other steps, one could directly assess whether differences in automated gating performance have any meaningful effect on the biological conclusions drawn from the data. Ultimately, the relevant question for end users is not whether an algorithm removes a few percent more or fewer debris events, but whether doing so materially affects downstream analysis.

### Reviewer 1, Comment 3 — Biological Dataset Breadth

> Methodological Robustness: Utilizing a single expert per tissue type limits the statistical power of the ranking methodology. It prevents the reader from seeing the intra-tissue consensus—how consistently multiple experts on the same tissue agree on the rank-order of methods used on a specific sample. What it currently highlights is how a single tissue expert has a unique perspective of the correct gating for a given tissue and inter-operator gating variability; not algorithmic gating superiority. Other improvements could be adding different processing conditions (e.g. fixation) or more variation in tissue used (e.g. include human PBMC or other species' tissues).

### Reviewer 1, Comment 4 — Discussion of Error “Costs” and Sensitivity Trade-offs

> The manuscript notes that PeacoQC exhibits higher sensitivity but lower specificity compared to FlowMOP. This trade-off is central to clinical discovery and warrants a deeper discussion.
>
> Discussion Point: The authors should explicitly address the biological “cost” of these errors. For instance, is it more detrimental to the study's integrity to leave in a small transient artifact or to accidentally remove a rare, clinically significant biological population?

### Reviewer 2 — Difficult Biological Samples and Tumour Validation

> The authors developed An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool. This can be a useful tool for cytometry. However, the dataset used in the paper are relatively “easy” cases. I recommend the authors to challenge the FlowMOP with more difficult but common samples like digested tumor sample. Performance against human expert is needed.

### Reviewer 2, Comment 10

> The study did not validate the FlowMOP on tumor datasets, which is a common sample type for flow cytometry but difficult to perform debris removal. I highly suggest the authors to include a tumor dataset, even if the performance is not optimal, the readers can know whether the method can be applied in their sample or not.

### Response

We <span style="color:#c00000"><s>agree that the most important validation question is whether automated preprocessing preserves biologically meaningful downstream results. We have</s></span><span style="color:#0066cc">elected to conduct further biological validation, and</span> therefore <span style="color:#c00000"><s>brought the downstream biological-validation evidence together here, rather than dispersing it across the responses concerning expert preference, tumour samples,</s></span><span style="color:#0066cc">added PBMC samples from eight donors and three human liver-tumour samples. These analyses quantify PBMC population recovery and biological composition</span> and <span style="color:#c00000"><s>the experimental relevance of the synthetic benchmark.</s></span><span style="color:#0066cc">tumour-population recovery.</span>

We <span style="color:#c00000"><s>also clarify</s></span><span style="color:#0066cc">agree with Reviewer 1</span> that the original <span style="color:#c00000"><s>study was not restricted to murine samples. It already included three distinct human sample contexts: PBMCs in</s></span><span style="color:#0066cc">single-expert-per-tissue design cannot establish within-tissue expert consensus or support claims of algorithmic superiority. We therefore present it as an expert-preference evaluation, retain</span> the <span style="color:#c00000"><s>synthetic time-gating benchmark, cultured human T cells</s></span><span style="color:#0066cc">complete analysis</span> in the <span style="color:#c00000"><s>expert-ranking comparison,</s></span><span style="color:#0066cc">Supplementary Information,</span> and <span style="color:#c00000"><s>human liver tissue</s></span><span style="color:#0066cc">report its principal comparative findings</span> in the <span style="color:#c00000"><s>expert-ranking comparison. The revised study builds on this existing human-sample breadth by adding a direct downstream comparison between FlowMOP and matched manual expert preprocessing</s></span><span style="color:#0066cc">main Results, as detailed</span> in <span style="color:#c00000"><s>complex human tumour-liver samples.</s></span><span style="color:#0066cc">Combined Comment 6. To broaden the biological validation, we added the PBMC and tumour datasets described here; we did not add a separate fixation or cross-species experiment.</span>

<span style="color:#c00000"><s>To address Reviewer 2's request directly,</s></span><span style="color:#0066cc">For the PBMC analysis,</span> we <span style="color:#0066cc">compared the time-gating methods while holding the downstream expert-defined singlet, debris, Live-cell, and lineage coordinates fixed. We also</span> evaluated <span style="color:#c00000"><s>FlowMOP</s></span><span style="color:#0066cc">the debris and doublet modules separately against matched expert inputs</span> in <span style="color:#c00000"><s>three complex human tumour-liver samples</s></span><span style="color:#0066cc">which only the relevant cleaning step differed,</span> and compared <span style="color:#c00000"><s>its</s></span><span style="color:#0066cc">the</span> complete <span style="color:#c00000"><s>preprocessing output</s></span><span style="color:#0066cc">Time + Debris + Doublet workflow</span> with<span style="color:#c00000"><s>matched manual expert gating. We assessed downstream biological consequences using four prespecified endpoints rather than relying solely on gate overlap. This analysis therefore addresses</s></span> the <span style="color:#c00000"><s>request for testing in tumour samples, comparison with a human expert, and evaluation of whether automated cleaning changes downstream biological conclusions.</s></span><span style="color:#0066cc">expert-combined workflow.</span>

<span style="color:#c00000"><s>The principal</s></span><span style="color:#0066cc">Live CD45+ was reported as a standalone reference endpoint. Applying it as an</span> additional <span style="color:#c00000"><s>biological validation uses human PBMC data and prespecified downstream population endpoints. This analysis evaluates whether cleaning preserves</s></span><span style="color:#0066cc">parent gate excluded many low-scatter events before</span> the <span style="color:#c00000"><s>B-cell-to-T-cell ratio</s></span><span style="color:#0066cc">lineage endpoints were quantified</span> and <span style="color:#c00000"><s>relevant constituent populations, providing a direct biological readout rather than another subjective gate-ranking exercise. PLACEHOLDER:</s></span><span style="color:#0066cc">therefore obscured</span> the <span style="color:#c00000"><s>final PBMC cohort description, population definitions, results, statistical comparisons,</s></span><span style="color:#0066cc">effects of debris removal. To expose those effects directly, we did not include CD45+ as the parent of the lineage populations. B cells were defined as Live CD3−CD19+ events, T cells as Live CD3+CD19− events,</span> and <span style="color:#c00000"><s>Figure 5 references will be inserted here when</s></span><span style="color:#0066cc">NKT cells as Live CD19−CD3+CD56+ events. We report counts and frequencies normalized to their corresponding matched Raw values (Raw = 1); before normalization, B-, T-, and NKT-cell frequencies were calculated relative to Live cells. All analyses used paired PBMC samples from</span> the <span style="color:#c00000"><s>analysis is complete.</s></span><span style="color:#0066cc">same eight donors. Direct workflow comparisons used two-sided paired *t*-tests, while Raw comparisons used one-sample *t*-tests against 1 on the Raw-normalized values, equivalent to paired comparisons with the matched Raw values; tests were Holm-adjusted separately within endpoint and outcome metric.</span>

<span style="color:#c00000"><s>We compared</s></span><span style="color:#0066cc">For time gating, no evaluated B-, T-, or NKT-cell frequency differed significantly among</span> the <span style="color:#c00000"><s>Raw data with sequential manual preprocessing and the complete</s></span><span style="color:#0066cc">methods (all adjusted p values were 0.420 or greater). Relative to Expert Manual,</span> FlowMOP <span style="color:#c00000"><s>output, in which independently calculated time, debris, and doublet exclusions are combined as a union of excluded events. We then quantified four prespecified downstream endpoints: the T:B-cell ratio, Live CD45+ cell count, B-cell frequency,</s></span><span style="color:#0066cc">retained 11.1–12.7 percentage points more B, T,</span> and <span style="color:#c00000"><s>T-cell frequency. T</s></span><span style="color:#0066cc">NKT</span> cells <span style="color:#0066cc">(all adjusted p values</span> were <span style="color:#c00000"><s>defined as CD3+CD19− and B cells as CD3−CD19+. Each endpoint was normalized within sample to its matched Raw value,</s></span><span style="color:#0066cc">0.003 or less), FlowCut retained 12.3–13.8 points more (all adjusted p values were 0.006 or less),</span> and <span style="color:#c00000"><s>all three unadjusted, two-sided paired t-tests</s></span><span style="color:#0066cc">PeacoQC retained 11.7–13.4 points fewer (all adjusted p values</span> were <span style="color:#c00000"><s>performed.</s></span><span style="color:#0066cc">0.035 or less; Figure 5).</span>

<span style="color:#c00000"><s>No difference between Manual</s></span><span style="color:#0066cc">Figure 6 reports debris-only, doublet-only,</span> and <span style="color:#0066cc">combined preprocessing in three subcolumns per population.</span> FlowMOP <span style="color:#c00000"><s>was detected for the T:B-cell ratio (p = 0.636), Live CD45+ cell</s></span><span style="color:#0066cc">generally removed fewer events overall than Expert Manual during debris cleaning. No debris</span> count <span style="color:#c00000"><s>(p = 0.545), B-cell</s></span><span style="color:#0066cc">endpoint differed between the methods, and no population</span> frequency <span style="color:#c00000"><s>(p = 0.855), or</s></span><span style="color:#0066cc">differed except</span> T-cell <span style="color:#c00000"><s>frequency (p = 0.985). Relative to Raw, both methods reduced B-cell frequency (Manual p = 0.025;</s></span><span style="color:#0066cc">frequency, which was lower after</span> FlowMOP <span style="color:#0066cc">cleaning (Raw-normalized mean 1.013 versus 1.069; adjusted</span> p = <span style="color:#c00000"><s>0.020) and T-cell</s></span><span style="color:#0066cc">0.037). Doublet cleaning produced no significant count or</span> frequency <span style="color:#c00000"><s>(Manual p = 0.023; FlowMOP p = 0.006).</s></span><span style="color:#0066cc">difference between</span> FlowMOP <span style="color:#c00000"><s>also reduced the Live CD45+ cell</s></span><span style="color:#0066cc">and Expert Manual. When Time, Debris, and Doublet preprocessing were combined, no</span> count <span style="color:#c00000"><s>relative to Raw (p = 0.024), whereas the Manual-versus-Raw comparison</s></span><span style="color:#0066cc">endpoint differed; B-cell frequency</span> was <span style="color:#c00000"><s>not significant (p = 0.069). No T:B-cell ratio comparison</s></span><span style="color:#0066cc">the only frequency that differed and</span> was <span style="color:#c00000"><s>significant (Manual versus Raw, p = 0.715;</s></span><span style="color:#0066cc">lower after</span> FlowMOP <span style="color:#0066cc">cleaning (0.877</span> versus <span style="color:#c00000"><s>Raw,</s></span><span style="color:#0066cc">1.022; adjusted</span> p = <span style="color:#c00000"><s>0.906; Manual versus FlowMOP, p = 0.636).</s></span><span style="color:#0066cc">0.049).</span> These results <span style="color:#c00000"><s>support concordant downstream conclusions in these three complex samples. The analysis compares the complete preprocessing workflows</s></span><span style="color:#0066cc">indicate broad agreement between human-</span> and <span style="color:#c00000"><s>their downstream consequences; it does not provide event-level tumour-debris ground truth or isolate debris-classification accuracy.</s></span><span style="color:#0066cc">FlowMOP-mediated pregating. Supplementary</span> Figure S8 <span style="color:#c00000"><s>additionally contrasts the sequential manual strategy</s></span><span style="color:#0066cc">contains representative debris and doublet preprocessing projections,</span> with <span style="color:#c00000"><s>FlowMOP's parallel exclusion logic.</s></span><span style="color:#0066cc">gate outlines shown only where the corresponding preprocessing gate was applied.</span>

<span style="color:#c00000"><s>These biological analyses complement, but do not convert,</s></span><span style="color:#0066cc">In the tumour analysis,</span> the <span style="color:#c00000"><s>earlier forced expert rankings into an objective benchmark. The expert-ranking exercise remains an exploratory comparison of</s></span><span style="color:#0066cc">B- and T-cell endpoints are population recovery</span> relative <span style="color:#c00000"><s>preferences</s></span><span style="color:#0066cc">to Raw, not frequency or composition within the cleaned output, because their source counts use the original total-event denominator. No statistically detectable differences were observed between Manual</span> and <span style="color:#c00000"><s>is addressed separately</s></span><span style="color:#0066cc">FlowMOP for the evaluated recovery endpoints</span> in <span style="color:#c00000"><s>Combined Comment 6.</s></span><span style="color:#0066cc">these three samples (Figure 7). Given n = 3, these nonsignificant results do not establish equivalence.</span>

<span style="color:#c00000"><s>Neither higher</s></span><span style="color:#0066cc">The biological data alone cannot establish whether the lower retention by Expert Manual and PeacoQC reflects more sensitive artifact</span> removal <span style="color:#c00000"><s>nor closer agreement</s></span><span style="color:#0066cc">or whether the higher retention by FlowMOP and FlowCut reflects more specific preservation of valid events. Considering that PeacoQC demonstrated poorer specificity than FlowMOP and FlowCut in the synthetic datasets, together</span> with <span style="color:#c00000"><s>a single</s></span><span style="color:#0066cc">the relatively blunt nature of</span> human <span style="color:#c00000"><s>gate is inherently preferable. Under-cleaning can retain acquisition artifacts</s></span><span style="color:#0066cc">manual time gating, we believe</span> that <span style="color:#c00000"><s>create, inflate, or obscure apparent</s></span><span style="color:#0066cc">the higher retention is more likely to reflect specific preservation of valid events than insufficient artifact removal.</span>

<span style="color:#0066cc">The trade-off between sensitivity and specificity reflects competing</span> biological <span style="color:#c00000"><s>populations. Conversely, over-cleaning can remove rare</s></span><span style="color:#0066cc">risks rather than a purely statistical optimization. A more permissive gate may protect genuine</span> or <span style="color:#c00000"><s>transient</s></span><span style="color:#0066cc">rare populations while allowing artifacts to remain; a more stringent gate may remove artifacts more completely while also deleting valid</span> biological events. <span style="color:#0066cc">No balance is universally correct because the consequences of each error depend on the intended downstream analysis.</span> We therefore <span style="color:#c00000"><s>interpret</s></span><span style="color:#0066cc">report both</span> sensitivity and specificity <span style="color:#c00000"><s>as complementary measures with competing biological costs rather than treating either direction of</s></span><span style="color:#0066cc">and interpret them alongside their effects on population recovery and composition. Importantly, FlowMOP had the strongest overall combined sensitivity-specificity profile in the source-labelled synthetic time-gating benchmarks: its gains were not achieved simply by exchanging one</span> error <span style="color:#c00000"><s>as universally preferable.</s></span><span style="color:#0066cc">type for the other.</span>

<span style="color:#0066cc">These biological analyses better illustrate FlowMOP's capabilities and reinforce the findings demonstrated in the synthetic datasets. Figure 7A additionally shows representative Manual and FlowMOP Time, Debris, and Doublet preprocessing plots.</span>

### Changes made

- We added a <span style="color:#0066cc">biological validation using</span> PBMC <span style="color:#c00000"><s>biological-validation section based on prespecified downstream population endpoints; the final analysis</s></span><span style="color:#0066cc">samples from eight donors, with matched-Raw frequencies and counts for Live CD45+, B, T,</span> and <span style="color:#0066cc">NKT cells.</span> Figure 5 <span style="color:#c00000"><s>content remain a PLACEHOLDER.</s></span><span style="color:#0066cc">reports time-cleaning representatives (A), frequencies (B), and counts (C). Figure 6 reports the combined Time + Debris + Doublet representative (A) and the debris-only, doublet-only, and combined frequency (B) and count (C) comparisons. Supplementary Figure S8 contains the debris and doublet preprocessing projections; gate outlines appear only where the corresponding preprocessing gate was applied.</span>

<span style="color:#0066cc">- We found no direct count difference between FlowMOP and Expert Manual in the debris, doublet, or combined comparisons. No doublet frequency differed between methods; the only direct frequency differences in Figure 6 were debris-only T-cell frequency (adjusted p = 0.037) and combined B-cell frequency (adjusted p = 0.049).</span>

- We added a tumour validation comprising three tumour samples, with paired Raw, Manual, and FlowMOP comparisons of<span style="color:#c00000"><s>the T:B-cell ratio,</s></span> Live CD45+ <span style="color:#c00000"><s>cell count, B-cell frequency,</s></span><span style="color:#0066cc">cell-count recovery and B-</span> and T-cell <span style="color:#c00000"><s>frequency.</s></span><span style="color:#0066cc">population recovery.</span> Figure <span style="color:#c00000"><s>6</s></span><span style="color:#0066cc">7</span> displays the matched gating plots and <span style="color:#c00000"><s>the Live CD45+ cell-count, B-cell-frequency, and T-cell-frequency</s></span><span style="color:#0066cc">downstream</span> comparisons. <span style="color:#0066cc">No statistically detectable differences were observed between Manual and FlowMOP for these recovery endpoints, but n = 3 does not support an equivalence claim.</span>

- We added<span style="color:#c00000"><s>a</s></span> representative <span style="color:#c00000"><s>schematic showing sequential manual gates versus FlowMOP's parallel exclusions</s></span><span style="color:#0066cc">Manual</span> and <span style="color:#c00000"><s>union/intersection logic</s></span><span style="color:#0066cc">FlowMOP Time, Debris, and Doublet preprocessing plots</span> (Figure <span style="color:#c00000"><s>S8).</s></span><span style="color:#0066cc">7A).</span>

- We <span style="color:#0066cc">acknowledged that the single-expert-per-tissue rankings cannot estimate within-tissue consensus, retained the complete ranking analysis in the Supplementary Information, and</span> added <span style="color:#0066cc">its principal comparative findings to the main Results. We broadened</span> the <span style="color:#c00000"><s>following discussion of competing error costs:</s></span><span style="color:#0066cc">validation with human PBMC and tumour datasets but did not add a separate fixation or cross-species experiment.</span>

<span style="color:#c00000"><s>> “The biological cost</s></span><span style="color:#0066cc">- We added a direct discussion</span> of <span style="color:#c00000"><s>preprocessing errors is difficult to measure directly, and neither under-cleaning nor over-cleaning is preferable. Under-cleaning may allow acquisition-time artifacts with abnormal staining patterns to confound downstream results, including by creating, inflating, or obscuring apparent rare populations. Conversely, over-cleaning could remove rare or transient biological populations.”</s></span><span style="color:#0066cc">competing error costs:</span>

  <span style="color:#0066cc">> “The trade-off between sensitivity and specificity reflects competing biological risks rather than a purely statistical optimization. A more permissive gate may protect genuine or rare populations while allowing artifacts to remain; a more stringent gate may remove artifacts more completely while also deleting valid biological events. No balance is universally correct because the consequences of each error depend on the intended downstream analysis. In this context, FlowMOP's synthetic time-gating results are notable because its performance gains were not achieved simply by exchanging one error type for the other. FlowMOP had the strongest overall combined sensitivity-specificity profile across the tested conditions.”</span>

**Location:** Methods, <span style="color:#0066cc">“Human PBMC biological-validation sample preparation,” “Biological-validation analysis,” and</span> “Tumour biological-validation analysis”; Results, <span style="color:#0066cc">“Biological validation” and</span> “Tumour biological validation”; Discussion, “Biological validation” and “Other remarks”; Figures <span style="color:#c00000"><s>5 and 6;</s></span><span style="color:#0066cc">5–7;</span> Figure S8.

## Combined Comment 4 — Gating Mechanism, Smoothing, and Parameter Fairness

**Raised by:** Associate Editor; Reviewer 1, Comment 5; Reviewer 2, Comment 20.

### Associate Editor — Algorithmic Motivation and Theoretical Justification

> More troublingly, the authors themselves appear uncertain about why their algorithm performs better in certain cases. In the Discussion, they write that a performance difference “may be attributed to FlowMop's 'smoothing' implementation.” This speculation is unsatisfying — can the authors not test this directly by modifying the smoothing implementation and evaluating whether the results change accordingly? While presenting cherry-picked examples in which one algorithm outperforms another has illustrative value, a rigorous justification of the theoretical underpinnings of the algorithmic design is essential. Without it, readers cannot reasonably assess which sample types and experimental conditions are expected to favor FlowMop over existing alternatives.

### Reviewer 1, Comment 5 — Parametric Sensitivity Analysis

> To strengthen the claim of “superiority,” it is essential to demonstrate that the results are not merely a function of suboptimal default settings for the competing algorithms.
>
> Recommendation: A supplemental figure showing a sensitivity analysis—where the parameters of PeacoQC and FlowCut are tuned across a range—would provide a more “apples-to-apples” comparison and confirm that FlowMOP’s performance gains are algorithmic rather than parametric.

### Reviewer 2, Comment 20

> The authors say, “Finally, FlowMOP also novelly employs the use of smoothing at multiple resolutions to ensure multiple types of abherrations are detected”. What are the “multiple types of abherrations”? please clarify.

### Response

We now seek to address the mechanisms underlying FlowMOP's superior performance under the primary fixed comparison settings in the tested Segment, Bimix, and Trimix scenarios. Our hypotheses arose from the different signals used by the methods and from the pattern observed in the original benchmark. FlowMOP uses acquisition order to define time bins but bases its removal decision on fluorescence summaries of globally defined positive populations; Time itself is excluded from the quality variables. FlowCut, by contrast, can respond to local acquisition-density structure. This distinction was particularly relevant because the Segment files retained stronger source-linked Time-density structure and showed the clearest FlowCut performance loss, whereas Time was approximately normalized in the constructed Bimix and Trimix files. We therefore hypothesised that FlowCut's relative loss arose, at least in part, from sensitivity to local acquisition rate rather than to the source-linked fluorescence defect itself.

To test this hypothesis, we conducted a matched Time-only mechanism analysis. This involved changing only the Time channel while preserving fluorescence, scatter, source labels, and event order across 30 source-labelled Bimix, Trimix, and Segment files. We found that FlowMOP and PeacoQC were both unchanged under source-linked and random Time warping, whereas FlowCut's removal behaviour changed when local Time density was altered. The clearest loss occurred in Segment inputs under source-linked Time warping.

PeacoQC's documentation identifies acquisition-bin size as a trade-off between the accuracy of within-bin density estimation, the number of bins available for evaluating signal stability, and the number of events affected when a bin is removed [5]. The Bimix and Trimix benchmarks contain short, interspersed source-defined intervals. FlowMOP's stronger performance is therefore consistent with its temporal summarisation being better matched to these brief intervals.

In the regenerated primary benchmark, FlowMOP had higher sensitivity than FlowCut in both Segment settings and in the 5000-event Bimix and Trimix benchmarks. At 5000 events, FlowMOP also had higher specificity than both competitors across Segment, Bimix, and Trimix. In the smaller mixed-source benchmarks, FlowMOP retained higher specificity than PeacoQC without a significant sensitivity disadvantage.

Consequently, the earlier smoothing attribution has been removed. The matched Time-only benchmark directly supports the distinction between FlowMOP's fluorescence-population-summary decision and FlowCut's response to acquisition-density changes when fluorescence is unchanged. PeacoQC's documented bin-size trade-off provides the relevant context for its fixed-setting comparison with FlowMOP.

We thank the reviewers for prompting this direct examination of smoothing. We tested a range of smoothing settings across all synthetic time-gating inputs while holding all other FlowMOP settings fixed (Table S4). No smoothing provided the best specificity, but at the cost of sensitivity. Inspection showed that the additional target-source events removed under smoothing lay in distribution shoulders, where target and non-target events were difficult to distinguish reliably. We therefore retained smoothing as the conservative default: it avoids treating ambiguous shoulder events as confidently valid and provides higher sensitivity, while accepting the corresponding specificity trade-off. Among the smoothed settings, <span style="color:#c00000"><s>`0.01,0.05`</s></span><span style="color:#0066cc">0.01,0.05</span> provided the strongest balance and was selected as the default. We reran FlowMOP across all Figure 2 inputs and regenerated Figure 2 using these outputs.

All algorithms were compared using documented recommended, default, or automatically selected settings, including fixed FlowMOP parameters, to reflect typical unsupervised use. We did not perform extensive parameter tuning for FlowCut or PeacoQC because the original method descriptions do not provide dataset-specific guidance for how such tuning should be performed. We therefore restrict our conclusions to the fixed comparison settings and the behaviours directly tested by the matched Time-only benchmark.

The two smoothing resolutions are now described as complementary spline fits applied to the same time-bin fluorescence-summary series before MAD filtering. A bin can be detected through either smoothing pass; the manuscript no longer relies on the imprecise phrase “multiple types of aberrations.”

### Changes made

- We removed the earlier smoothing attribution and added the following matched comparison:

  > “To test the effect of flow-rate disturbances aligned with or independent of source-linked fluorescence changes, we altered only the Time channel while leaving fluorescence, scatter, source labels, and event order unchanged (Fig. S4). FlowMOP and PeacoQC were unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shifted after Time-only perturbation.”

- We clarified the function of the two smoothing resolutions as follows:

  > “Two spline smoothing values, one small and one larger (current default <span style="color:#c00000"><s>`0.01,0.05`),</s></span><span style="color:#0066cc">0.01,0.05),</span> are applied to the returned time-bin series before median absolute deviation (MAD) filtering.”

  > “The two smoothing resolutions target both shorter and more sustained deviations, while parameter voting limits removal driven by isolated noisy channels in higher-dimensional panels.”

- We expanded the smoothing analysis across all 173 primary synthetic time-gating inputs. Supplementary Table S4 reports the tested range and identifies <span style="color:#c00000"><s>`0.01,0.05`</s></span><span style="color:#0066cc">0.01,0.05</span> as the strongest-performing smoothed setting on the balanced comparison.

- We reran FlowMOP using the selected <span style="color:#c00000"><s>`0.01,0.05`</s></span><span style="color:#0066cc">0.01,0.05</span> setting and regenerated the Figure 2 violin distributions, p-values, and significance brackets from these outputs.

- We clarified the scope of the fixed-setting comparison:

  > “The primary comparison used recommended or automatically selected settings, including fixed FlowMOP parameters, to reflect typical unsupervised use.”

- We added the complete mechanism-benchmark methods and results as Figure S4.

**Location:** Methods, “Time-only acquisition-rate mechanism benchmark” and “MAD-smoothing ablation and default selection”; Results, “Time Gating” and “Synthetic Time Gating Benchmark”; Figure 2; Figure S4;<span style="color:#c00000"><s>Supplementary</s></span> Table S4; Discussion, “Synthetic Sample Time Gating.”

## Combined Comment 5 — Parameter Voting and the Ten-Parameter Threshold

**Raised by:** Reviewer 2, Comments 18 and 21.

### Reviewer 2, Comment 18

> The authors say ”when >10 parameters are present, bins are rejected if they are flagged by two or more parameters”. Why “10” is used as threshold here?

### Reviewer 2, Comment 21

> The authors say, “parameter voting to ensure that higher dimensional sample sets are not overly degraded.” It would be nice to show how bad the overly degradation will be is parameter voting is not applied. It is a little a bit concern that events with aberrant signal in one fluorescence are not detected, even in a big panel.

### Response

The revised manuscript identifies the ten-parameter cutoff as an empirical operational threshold rather than a mathematically derived boundary.

The rationale for requiring two channel flags in panels with more than ten eligible parameters is conservative. Biological datasets have a non-zero probability of isolated channel-specific outliers, and the opportunity for such outliers increases with panel dimensionality. Requiring corroboration from a second parameter reduces the risk that a single noisy channel removes otherwise valid biological events. The corresponding trade-off is that an abnormality confined to one fluorescence channel may be retained in a high-dimensional panel. We did not add a separate voting-ablation analysis and have avoided presenting the rule as universally optimal.

### Changes made

- We added the following rationale and limitation:

  > “For panels with more than 10 parameters, FlowMOP requires two or more parameters to flag a bin before rejection. This empirical safeguard reduces false-positive removal caused by isolated noisy channels in high-dimensional panels, although an aberration confined to one channel may consequently be retained (Figure 1A).”

- We did not add a voting-ablation experiment and do not present the threshold as universally optimal.

**Location:** Results, “Time Gating.”

## Combined Comment 6 — Expert Rankings, Terminology, Visualisation, and Bayesian Modelling

**Raised by:** Associate Editor; Reviewer 1, Comment 3; Reviewer 2, Comment 16.

### Associate Editor — Expert Preference Rankings and Statistical Interpretation

> My other major concern pertains to the substantial space — both in the main manuscript and supplemental materials — devoted to preferential rankings by four experts across multiple sample types and algorithms, analyzed using a Bayesian statistical framework. Setting aside the question of whether all experts can reasonably be considered equally expert across all sample types, and the limited statistical power afforded by four data points, I find the interpretation of these rankings fundamentally ambiguous.
>
> As expected, experts generally preferred human-gated data over automated tools — but this finding is difficult to interpret in isolation. Forcing experts to rank outputs necessarily produces an ordering, but that ordering conflates two very different scenarios: one in which all automated results are unacceptable and the other in which all automated results are perfectly adequate, with human gating simply preferred at the margin. Similarly, even if FlowMop was ranked above the other automated tools, the reader has no basis for judging whether this difference reflects a meaningful improvement in analytical utility or merely a marginal statistical preference with no practical consequence.
>
> I would suggest that the authors replace or supplement the ranking analysis with a simpler, more interpretable quality scale — for example: 1 = excellent gating; 2 = adequate and suitable for analysis; 3 = inadequate for analysis but not terrible; 4 = unacceptable. This approach would convey absolute quality rather than relative preference and would be more informative to readers.

### Reviewer 1, Comment 3 — Refinement of Human Subjective Rankings

> While the subjective rankings provide an interesting window into expert perspectives on automated gating, the presentation and terminology in this section require significant refinement:
>
> Nomenclature: In computer science, the term “benchmarking” typically implies an objective, repeatable measure of performance in specific scenarios. In the context of subjective human preference, “Comparison” or “Evaluation” is a more accurate/appropriate term. This sentence, “FlowMOP’s outputs in these samples were benchmarked against a set of human experts”, should more precisely say that FlowMOP's outputs were “compared against expert gating”.
>
> Methodological Robustness: Utilizing a single expert per tissue type limits the statistical power of the ranking methodology. It prevents the reader from seeing the intra-tissue consensus—how consistently multiple experts on the same tissue agree on the rank-order of methods used on a specific sample. What it currently highlights is how a single tissue expert has a unique perspective of the correct gating for a given tissue and inter-operator gating variability; not algorithmic gating superiority. Other improvements could be adding different processing conditions (e.g. fixation) or more variation in tissue used (e.g. include human PBMC or other species' tissues).
>
> Data Visualization: The ranking scale (7 = Best, 1 = Worst) is possibly counter-intuitive and should be reversed to follow standard ordinal ranking conventions. The row axis title in the grid figure should be updated to “Gate Provided By” to clearly disambiguate between the creator of the gate (e.g., “Expert 1”) and the individual performing the ranking (“<tissue> Expert”).
>
> The scatter plot in Panel B adds limited value; replacing it with an “Average Score” column at the end of the grid would be cleaner. This statement is oddly contorted to be true, but is misconstruing the data: “However, it is of note that in 4/5 and 7/9 datasets in the debris and doublets tasks respectively, FlowMOP was not the least preferred, indicating superiority to at least one human benchmarker (Fig. 6A, 7A)”. Another interpretation is that FlowMOP had the lowest average score in Fig. 7A/B and had the fewest tissue cases where experts deemed it as the best result or even better-than-average traditional gating.

### Reviewer 2, Comment 16

> There is only a “Bayesian Modelling” section under the “Non-synthetic samples” section in the Method. The Bayesian Modelling section described the method to rank performance of human experts, FlowMOP, FlowCut and PeacoQC. It is not about how “Non-synthetic samples” are generated at all. Also, the purpose of the Bayesian Modelling is well described in the beginning of this subsection, which makes it difficult to follow.

### Response

Forced rankings measure relative preference <span style="color:#c00000"><s>and cannot determine whether a gate is absolutely acceptable for analysis.</s></span><span style="color:#0066cc">rather than objective event-level accuracy.</span> We therefore replaced “benchmarking” with “expert comparison” or “expert evaluation,” removed the contorted “not least preferred” interpretation, and now state directly <span style="color:#c00000"><s>where</s></span><span style="color:#0066cc">how</span> FlowMOP ranked <span style="color:#c00000"><s>below</s></span><span style="color:#0066cc">relative to</span> the human <span style="color:#0066cc">and automated</span> gates.

The <span style="color:#c00000"><s>ranking</s></span><span style="color:#0066cc">evaluation produced distinct</span> results <span style="color:#0066cc">across the three cleaning modules. For time gating, FlowMOP had the best mean rank among the automated methods and was preferred to FlowCut (BF = 5.39, P = 84.3%) and PeacoQC (BF = 12.10, P = 92.4%). Human gates generally ranked higher for debris and doublet removal, although FlowMOP was competitive in several tissue-specific comparisons: it ranked first for debris removal in mouse blood, third for debris removal in human liver and mouse skin, and second for doublet removal in human liver (Figs. S5-S7). We now report these principal findings and statistics in the main Results; the complete dataset-level rankings and mean ranks</span> are <span style="color:#c00000"><s>presented as exploratory evidence of expert preference rather than objective ground-truth evidence of gating quality or algorithmic superiority.</s></span><span style="color:#0066cc">shown in Figures S5-S7, with statistical comparisons reported in the Supplementary Results.</span>

The <span style="color:#c00000"><s>expert-ranking exercise used an earlier</s></span><span style="color:#0066cc">revised Discussion considers these results alongside the other validation approaches. The preference for</span> FlowMOP <span style="color:#c00000"><s>configuration. Subsequent parameter optimisation substantially improved FlowMOP's</s></span><span style="color:#0066cc">over the automated time-gating comparators agrees with its combined sensitivity-specificity</span> performance in the <span style="color:#c00000"><s>objective benchmarks, and the current implementation would therefore be expected to perform</s></span><span style="color:#0066cc">synthetic benchmarks. The</span> more <span style="color:#c00000"><s>strongly. However, because the expert evaluation was not repeated, the earlier</s></span><span style="color:#0066cc">variable debris and doublet</span> rankings <span style="color:#c00000"><s>do not represent the performance of the current implementation. We have consequently moved them</s></span><span style="color:#0066cc">are discussed in relation</span> to<span style="color:#c00000"><s>the Supplementary Information and further de-emphasised</s></span> their <span style="color:#c00000"><s>interpretation.</s></span><span style="color:#0066cc">dependence on sample-specific scatter distributions. Taken together, the results indicate that FlowMOP-generated gates can be comparable in acceptability to human gating across many use cases.</span>

The <span style="color:#c00000"><s>design used</s></span><span style="color:#0066cc">evaluation was designed as</span> a <span style="color:#c00000"><s>single relevant expert to rank</s></span><span style="color:#0066cc">relative-ranking exercise, with</span> each <span style="color:#c00000"><s>tissue type and therefore cannot estimate within-tissue consensus across multiple raters.</s></span><span style="color:#0066cc">dataset assessed by an expert familiar with that sample type.</span> We did not retrospectively <span style="color:#c00000"><s>collect a new</s></span><span style="color:#0066cc">replace this with an</span> absolute-quality scale because <span style="color:#c00000"><s>that</s></span><span style="color:#0066cc">doing so</span> would require <span style="color:#c00000"><s>re-running</s></span><span style="color:#0066cc">repeating</span> the<span style="color:#c00000"><s>expert</s></span> assessment under a <span style="color:#c00000"><s>newly defined</s></span><span style="color:#0066cc">different</span> protocol. <span style="color:#c00000"><s>The revised manuscript states this limitation and interprets</s></span><span style="color:#0066cc">Instead, we retain</span> the <span style="color:#c00000"><s>rankings</s></span><span style="color:#0066cc">ranking analysis and report its outcomes directly,</span> alongside the <span style="color:#0066cc">source-labelled</span> synthetic <span style="color:#c00000"><s>technical-control results that contain event-level ground truth.</s></span><span style="color:#0066cc">controls and the new PBMC and tumour analyses.</span> The reviewer's request for broader biological datasets, including human PBMCs, and the Associate Editor's request for downstream biological endpoints are addressed together in Combined Comment 3.

The visual summaries now include an Average Score column within each ranking grid rather than a separate panel, and the row axis is labelled “Gate Provided By.” The displayed scale now follows the standard convention that rank 1 is best. This display correction did not change the underlying ranking order or Bayesian analysis.

The Bayesian model is retained because the observed outcome is an ordinal ranking. The Plackett–Luce model retains the complete ordering and represents uncertainty in relative preference without treating differences between adjacent ranks as equal-interval continuous measurements. We have clarified that it estimates relative preference only. The subsection is presented as the statistical analysis of the expert-ranking data rather than as a method for generating non-synthetic samples.

### Changes made

- We defined the scope of the expert-ranking analysis as follows:

  > “The expert-ranking analysis was used to summarize relative preferences among gates generated by FlowMOP, comparator algorithms, and human operators; it was not treated as an absolute measure of gating <span style="color:#c00000"><s>adequacy.”</s></span><span style="color:#0066cc">adequacy [16].”</span>

- We revised the <span style="color:#0066cc">Supplementary</span> Results terminology and interpretation as follows:

  > “FlowMOP’s outputs in these samples were compared with expert-provided gates using a forced-ranking preference task. These rankings measure relative expert preference and should not be interpreted as an absolute measure of gating adequacy.”

<span style="color:#c00000"><s>- We added the following limitations:</s></span>

<span style="color:#c00000"><s>  > “One relevant expert ranked each tissue type, only four experts contributed gates, and the ordering does not distinguish a marginal preference from a judgement that a gate is unsuitable for analysis. Gate style could also reveal which outputs were algorithmic. We therefore interpret the ranking results as exploratory relative preferences, not as absolute quality scores or proof of algorithmic superiority or inferiority.”</s></span>

- We retained the Plackett–Luce analysis and clarified that it models ordinal preference rankings rather than absolute gating quality.

- We moved the response concerning dataset breadth and downstream biological validation to Combined Comment 3 and cross-reference it here rather than treating the expert rankings as biological validation.

- <span style="color:#c00000"><s>Because the rankings were generated using an earlier FlowMOP configuration, we moved</s></span><span style="color:#0066cc">We retained</span> the <span style="color:#0066cc">full</span> expert-ranking <span style="color:#c00000"><s>exercise to</s></span><span style="color:#0066cc">analysis in</span> the Supplementary Information and <span style="color:#c00000"><s>now describe it as an exploratory historical evaluation rather than a primary assessment of</s></span><span style="color:#0066cc">added its principal comparative findings and statistics to the main Results, with their implications discussed alongside</span> the <span style="color:#c00000"><s>current selected implementation.</s></span><span style="color:#0066cc">synthetic and biological-validation findings.</span>

- We regenerated Supplementary Figures <span style="color:#c00000"><s>S5–S7 as editable SVGs,</s></span><span style="color:#0066cc">S5–S7,</span> changed the row-axis title to “Gate Provided By,” displayed rank 1 as best, and replaced the separate summary panel with an Average Score column in each grid.

- We revised the Figure S5 caption as follows:

  > “Figure S5. Expert preference rankings for time gates provided by four human experts, FlowMOP (black border), FlowCut, and PeacoQC across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. Abbreviations: DRG, dorsal root ganglion; CNS, central nervous system.”

- We clarified the scoring direction in the <span style="color:#0066cc">Supplementary</span> Results <span style="color:#0066cc">and figure captions</span> as follows:

  > “Rank 1 indicates the greatest preference.”

**Location:** <span style="color:#0066cc">Results, “Expert preference evaluation”; Discussion; Conclusion;</span> Supplementary <span style="color:#c00000"><s>Information,</s></span><span style="color:#0066cc">data,</span> “Supplementary expert preference evaluation,” including “Supplementary methods” and “Supplementary results”; Supplementary Figures S5–S7.

## Combined Comment 7 — Dataset Scale and Complexity

**Raised by:** Associate Editor.

### Associate Editor — Smaller Point

> In the Introduction, the authors state that sophisticated analysis software has contributed to datasets of increasing size and complexity. It is not clear to me how analysis software makes debris or time gating more complex, particularly when pre-processing is predominantly scatter-based. Do the authors perhaps mean that derived parameters from imaging cytometry increase pre-processing complexity? Similarly, the claim that “datasets exceed human capabilities” presumably refers to the sheer number of files rather than the complexity of individual datasets — this should be stated explicitly. It is also not apparent from the manuscript how the tools described here depend on data complexity per se.

### Response

The revised Introduction distinguishes study scale from the intrinsic complexity of an individual preprocessing gate. It now refers explicitly to increasing numbers of files, events, and measured parameters and explains that this scale makes repeated manual preprocessing time-consuming and often impractical. It no longer implies that analysis software itself necessarily makes a debris or time gate more complex.

### Changes made

- We clarified the distinction between study scale and individual-gate complexity as follows:

  > “Modern <span style="color:#c00000"><s>flow-cytometry</s></span><span style="color:#0066cc">flow cytometry</span> studies can contain increasing numbers of files, events, and measured parameters <span style="color:#c00000"><s>[2].</s></span><span style="color:#0066cc">[1].</span> The large scale of these data makes repeated manual preprocessing time-consuming and often impractical, potentially amplifying variability between operators. Reproducible automated preprocessing is therefore valuable for large studies.”

**Location:** Introduction.

## Combined Comment 8 — Acquisition Settings and Newer Scatter Configurations

**Raised by:** Reviewer 2, Comments 13 and 28.

### Reviewer 2, Comment 13

> The voltage set of the cytometer significantly impact the resolution of debris. If the voltage is relatively low with bad resolution, can FlowMOP perform the automatic task well? Also, please describe the voltage setting in the data generation section in the Method.

### Reviewer 2, Comment 28

> The Xenith flow cytometry form Thermo Fisher provides 405 FSC/SSC, 488 FSC/SSC, and polar 488 FSC/SSC. This provides more flexible approach to gate target population. How can the FlowMOP adapt to this kind of flow data? It would be nice to discuss this in the Discussion.

### Response

The <span style="color:#0066cc">benchmarking relied on real, existing datasets, each acquired using experiment-specific, biology-driven antibody panels and instrument settings. Consequently,</span> acquisition voltage/gain settings <span style="color:#c00000"><s>used</s></span><span style="color:#0066cc">differed among datasets and were not controlled variables</span> in this <span style="color:#c00000"><s>study remain a PLACEHOLDER. We considered evaluating FlowMOP across a range of settings, but</s></span><span style="color:#0066cc">study. As with the antigen–fluorochrome combinations,</span> these <span style="color:#0066cc">settings</span> are <span style="color:#c00000"><s>highly dependent on</s></span><span style="color:#0066cc">therefore not material to</span> the <span style="color:#c00000"><s>instrument, detector, panel, sample type,</s></span><span style="color:#0066cc">preprocessing comparisons reported here,</span> and<span style="color:#c00000"><s>experimental conditions;</s></span> a <span style="color:#c00000"><s>voltage series on one cytometer</s></span><span style="color:#0066cc">single voltage/gain specification</span> would<span style="color:#c00000"><s>therefore</s></span> not <span style="color:#c00000"><s>provide generalisable thresholds for other settings.</s></span><span style="color:#0066cc">describe the datasets used.</span> We <span style="color:#c00000"><s>therefore did not add</s></span><span style="color:#0066cc">have clarified</span> this <span style="color:#c00000"><s>experiment</s></span><span style="color:#0066cc">scope in the Methods.</span>

<span style="color:#0066cc">Acquisition settings nevertheless determine whether the scatter signals needed by the debris</span> and <span style="color:#c00000"><s>instead added a combined acquisition/scatter limitation.</s></span><span style="color:#0066cc">doublet modules are adequately resolved.</span> FlowMOP requires appropriate acquisition voltage/gain settings; if relevant signals are poorly resolved or saturated, the lost information cannot be recovered and reliable cleaning cannot be guaranteed. FlowMOP currently expects users to identify the scatter channels used by the workflow; it has not been validated systematically across 405-nm, 488-nm, and polar 488-nm FSC/SSC configurations on instruments with multiple scatter measurements. Future versions could assess multiple scatter-channel pairs and select or combine the pair with the clearest debris/doublet separation.

### Changes made

- <span style="color:#c00000"><s>The</s></span><span style="color:#0066cc">We clarified the dataset-specific</span> acquisition <span style="color:#c00000"><s>voltage/gain information remains a PLACEHOLDER.</s></span><span style="color:#0066cc">settings in the Methods:</span>

  <span style="color:#c00000"><s>- We did not add a voltage/gain-series experiment because</s></span><span style="color:#0066cc">> “The benchmarking uses real, existing datasets, each acquired using experiment-specific, biology-driven antibody panels and instrument settings. Consequently,</span> the <span style="color:#c00000"><s>tested</s></span><span style="color:#0066cc">antigen–fluorochrome combinations and acquisition voltage/gain</span> settings <span style="color:#c00000"><s>would be specific to the instrument, detector, panel, sample type,</s></span><span style="color:#0066cc">differ among datasets</span> and <span style="color:#c00000"><s>experimental conditions rather than broadly generalisable. Instead, we added</s></span><span style="color:#0066cc">were not variables under evaluation.”</span>

<span style="color:#0066cc">- We retained</span> the following combined acquisition/scatter limitation:

  > “FlowMOP requires appropriate acquisition voltage/gain settings; if relevant signals are poorly resolved or saturated, the lost information cannot be recovered and reliable cleaning cannot be guaranteed. FlowMOP currently expects users to identify the scatter channels used by the workflow; it has not been validated systematically across 405-nm, 488-nm, and polar 488-nm FSC/SSC configurations on instruments with multiple scatter measurements. Future versions could assess multiple scatter-channel pairs and select or combine the pair with the clearest debris/doublet separation.”

**Location:** <span style="color:#0066cc">Methods, “Synthetic time source-sample preparation”;</span> Discussion, “Other <span style="color:#c00000"><s>remarks”; acquisition voltage/gain information: PLACEHOLDER.</s></span><span style="color:#0066cc">remarks.”</span>

## Combined Comment 9 — Temporal Artifacts and Synthetic Time-Sample Design

**Raised by:** Associate Editor; Reviewer 2, Comments 3, 4, and 14.

### Associate Editor — Smaller Point

> Regarding the bimix and trimix data used to simulate realistic fluorescence fluctuations: I am uncertain how these mixtures reproduce the kinds of flow rate variations that occur during actual experiments. If I understand correctly, the simulated fluctuations were subtle enough that human reviewers could not reliably detect them. I would ask the authors to clarify whether this represents a clinically or experimentally common problem — in my experience, flow rate deviations typically manifest as clear departures at the beginning or end of an acquisition, with mid-acquisition instabilities being relatively uncommon.

### Reviewer 2, Comment 3

> The authors should explain the concept of “temporal artifacts” in the abstract, given this is the core issue to be addressed in the study.

### Reviewer 2, Comment 4

> There are many terms in the abstract “(concatenated and mixed time perturbations, high-debris mixtures, and CFSE/CTV co-labeled doublets)”. I am not sure whether it is a good idea to put them here without explanation, considering unexplained term might add confusion to the readers. Anyway, the authors should introduce these terms in the introduction, which is missing.

### Reviewer 2, Comment 14

> The “Generation of Synthetic Time Samples” section is difficult to understand. This section is important for readers to understand the validation results. I recommend the authors use a illustration to show these 3 synthetic manners. Also, please explain why these three manners are designed in this way? So that the readers can follow the text.

### Response

The revised text explains the practical relevance of the Bimix and Trimix designs. It defines a “microblockage” operationally as a short, self-resolving mid-acquisition disturbance that produces a localized fluorescence shift; the term does not imply that a physical obstruction was directly observed. Existing work supports the substance of this phenomenon even though it uses broader terminology. flowAI documented flow-rate surges interspersed throughout acquisitions and their association with signal-intensity variation <span style="color:#c00000"><s>[10].</s></span><span style="color:#0066cc">[4].</span> PeacoQC describes temporary acquisition shifts caused by slow sample uptake or a clog and notes that such problems can be difficult to detect manually [5], while FlowCut describes manual identification and removal of transient acquisition problems as time-consuming and subjective <span style="color:#c00000"><s>[9].</s></span><span style="color:#0066cc">[6].</span>

Because the affected intervals can be short and their events interspersed with otherwise plausible events, they are difficult to recognize reliably by eye and impractical to exclude using a series of manual gates. We use “microblockage” for this subtle, self-resolving acquisition failure mode that a time-quality-control method should be able to detect. Estimating its prevalence across experiments would require a separate study.

The Bimix and Trimix samples model the observable fluorescence consequence in this operational definition rather than reproducing or proving a physical obstruction. This distinction is important. The synthetic samples can appear acceptable by eye, but their retained source labels reveal short intervals containing events from an intentionally perturbed fluorescence source. Under the benchmark definition, those events should be excluded even when visual inspection alone would not identify a defensible manual gate. The apparent visual normality of these files is therefore the reason an event-labelled benchmark is <span style="color:#c00000"><s>needed, not evidence that the modeled problem is unimportant.</s></span><span style="color:#0066cc">needed.</span>

The downstream biological-validation analyses are presented together in Combined Comment 3. That evidence provides an independent biological context for the technical benchmark; it is cross-referenced here rather than repeated. Together, the synthetic and biological analyses motivate an evaluation that includes both readily visible sustained artifacts and visually subtle transient microblockages.

We also defined temporal artifacts, simplified the terminology, and added Figure S1. We clarified that the primary synthetic samples model fluorescence changes across acquisition order without flow-rate disturbances, because rate changes without corresponding fluorescence changes should not prompt event exclusion. Flow-rate effects were tested separately, either aligned with source-linked fluorescence changes or independently of them. FlowMOP was unchanged in both conditions, whereas FlowCut's removal behavior shifted (Figure S4).

### Changes made

- In the Abstract, we incorporated the definition into the methodological summary:

  > “Methodologically, FlowMOP identifies temporal artifacts—acquisition-dependent deviations in event quality or fluorescence signal—via parameter-wise peak checks, bin-level fluorescence summaries across acquisition time, and robust outlier rejection.”

- In the Methods, we added the following clarification:

  <span style="color:#0066cc">> “Following acquisition, events from these real, source-labelled FCS files were computationally recombined to construct three time-gating benchmark designs. Their construction is illustrated in Supplementary Figure S1, and the dataset compositions are reported in Supplementary Table S1A.”</span>

<span style="color:#0066cc">  The wet-lab PBMC protocol is now presented separately under “Synthetic time source-sample preparation,” followed by “Computational construction of Segment, Bimix, and Trimix datasets.” This distinguishes preparation of the real source samples from post-acquisition benchmark construction.</span>

  > Here, we use “microblockage” operationally to denote a short, self-resolving mid-acquisition disturbance that produces a localized fluorescence shift; the term does not imply that a physical obstruction was directly observed. Segmented samples model sustained changes, whereas Bimix and Trimix model the observable fluorescence consequence in this operational definition by introducing short source-defined fluorescence shifts during acquisition. They do not recreate or establish the physical mechanism itself. The Bimix and Trimix files can appear acceptable on visual inspection because the altered intervals are short and interspersed with otherwise plausible events; however, the retained source labels identify the intentionally perturbed events that should be excluded under the benchmark definition.

  > “Flow-rate disturbances without corresponding fluorescence changes should not prompt event exclusion; these samples therefore model fluorescence changes across acquisition order without introducing flow-rate disturbances. Flow-rate effects were tested separately by altering Time either in alignment with source-linked fluorescence changes or independently of them.”

- In the Results, we added the following comparison:

  > “To test the effect of flow-rate disturbances aligned with or independent of source-linked fluorescence changes, we altered only the Time channel while leaving fluorescence, scatter, source labels, and event order unchanged (Fig. S4). FlowMOP and PeacoQC were unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shifted after Time-only perturbation.”

- In the Discussion, we now explain why the visually subtle synthetic cases are experimentally relevant and require an event-labelled benchmark:

  > Transient acquisition disturbances are not confined to the beginning or end of a run: flow-rate surges interspersed throughout acquisition and associated signal-intensity variation have been documented <span style="color:#c00000"><s>[10].</s></span><span style="color:#0066cc">[4].</span> PeacoQC notes that temporary acquisition problems can be difficult to detect manually [5], and FlowCut describes manual identification and removal of transient acquisition problems as time-consuming and subjective <span style="color:#c00000"><s>[9].</s></span><span style="color:#0066cc">[6].</span> We use “microblockage” operationally for a short, self-resolving instance of this broader phenomenon that produces a localized fluorescence shift, without asserting that a physical obstruction was directly observed.

  > “Although these synthetic samples can appear acceptable on visual inspection, the source labels show that the short altered intervals contain events from an intentionally perturbed fluorescence source and therefore should be excluded under the benchmark definition. Visual subtlety is thus a central feature of the benchmark: it demonstrates why apparent normality by eye is not sufficient ground truth.”

- We cross-reference the consolidated downstream biological-validation response in Combined Comment 3 rather than repeating its analyses here.

- We added Figure S1 with the following caption:

  > “Figure S1: Construction of Segment, Bimix, and Trimix synthetic time samples. No flow-rate disturbance was introduced.”

**Location:** Abstract; Introduction; Methods, <span style="color:#c00000"><s>“Generation</s></span><span style="color:#0066cc">“Synthetic time source-sample preparation,” “Computational construction</span> of <span style="color:#c00000"><s>Synthetic Time Samples”</s></span><span style="color:#0066cc">Segment, Bimix, and Trimix datasets,”</span> and “Time-only acquisition-rate mechanism benchmark”; Results, “Synthetic Time Gating Benchmark”; Discussion, “Synthetic Sample Time Gating”; Figures S1 and <span style="color:#c00000"><s>S4.</s></span><span style="color:#0066cc">S4; Table S1A.</span> The biological-validation changes are detailed in Combined Comment 3.

## Combined Comment 10 — Debris Methodology and Validation

**Raised by:** Reviewer 1, Comment 2; Reviewer 2, Comments 15, 22, and 24.

### Reviewer 1, Comment 2 — Debris Gating Methodology

> The inclusion of an automated debris-gating method is a welcome addition, particularly as this is often omitted in other QC tools. However, the current reliance on FSC-A as the primary determinant for debris may explain the lower rankings in expert consensus (Figure 6A).
>
> Critique: While the rationale is that debris is generally “low-size,” this does not account for larger morphological debris, such as large-cell clumps (e.g. tumor) or non-biological aggregates, which can occupy higher FSC/SSC ranges. Metadata-based margin removal techniques may augment this debris filtering.
>
> Recommendation: The authors should provide a more robust rationale for leaning exclusively on FSC-A or, ideally, discuss how the integration of SSC-A or Pulse Width might improve debris discrimination beyond ultra-low-size events.

### Reviewer 2, Comment 15

> The authors say “For assessment of FlowMOP gating performance, ‘high debris’ and ‘low debris’ samples were synthetically combined.” How are they combined? Please remember, well-described Method guarantees the transparency of the study.

### Reviewer 2, Comment 22

> The FlowMOP applies an FSC-A based threshold to exclude debris. I have concern about this FSC-A based idea. Commonly, we use FSC vs. SSC scatter to gate debris, which is clear in samples like PBMC. However, the resolution of a simple FSC-A histogram will not be as good as that in a FSC vs. SSC scatter, because debris and cells will overlap in FSC-A axis. In the Figure 3, the authors show representative results which is based on simple samples, which is possible to use the FSC-A based threshold method to gate debris. However, in difficult cases, which is not very rare, this FSC-A based threshold method may perform very bad.

### Reviewer 2, Comment 24

> I would like to see the debris remove rate calculated against the human expert.

### Response

The revised Methods describes the debris-removal procedure more precisely. FlowMOP ultimately applies a one-dimensional FSC-A threshold, but that threshold is not obtained from a single unconditioned FSC-A histogram. The algorithm identifies eligible fluorescence parameters, examines the FSC-A structure of each parameter's positive population, derives a candidate FSC-A threshold from each eligible channel, and applies the median candidate threshold to the sample. Thus, information across eligible fluorescence channels informs the final FSC-A gate, while the applied decision remains a conservative FSC-A threshold.

The current module is not a universal debris classifier. Debris and intact cells can overlap on FSC-A, and large clumps, tumour-associated material, non-biological aggregates, or other high-scatter contaminants may not be removed reliably. We specifically considered whether SSC-A should be incorporated. SSC-A can be informative for large or internally complex debris, but large and internally complex events can also represent desirable populations that should be retained. Their interpretation varies with tissue, staining panel, cytometer configuration, acquisition settings, and the biological populations of interest. Unlike the comparatively transferable relationship between very low FSC-A and small debris, there is no broadly safe high-SSC-A rule for excluding large events.

In the present technical-control datasets, the labelled small-debris and desirable source populations overlapped substantially along SSC-A, so a fixed SSC-A threshold offered limited additional separation for the small-event removal targeted by FlowMOP. We therefore did not incorporate SSC-A into the current debris decision. Pulse-width measurements may similarly help identify some aggregates or clumps, but their availability and interpretation are instrument- and acquisition-dependent. Incorporating SSC-A or pulse width responsibly would require a wide configurable parameter range or sample-specific multivariate decision rules validated for the intended dataset, panel, instrument, and cell populations. <span style="color:#0066cc">Metadata-based margin removal is complementary rather than a substitute for debris classification: it can identify events at acquisition limits, but it does not identify all low-FSC debris or large aggregates. FlowMOP currently includes only an FSC-A maximum-value precleaning check and does not implement a generalized margin-event filter.</span> The revised Discussion therefore defines the current scope as conservative low-FSC debris removal and presents SSC-A, pulse-width measurements, margin-event metadata, and sample-specific multivariate models as future extensions rather than universal default filters.

<span style="color:#c00000"><s>Preservation of</s></span><span style="color:#0066cc">Figure 3 compares FlowMOP with four expert debris gates using the source-labelled controls. Figure 6 complements this comparison with total debris-gate removal and</span> downstream <span style="color:#0066cc">biological preservation in paired</span> PBMC <span style="color:#c00000"><s>population structure and</s></span><span style="color:#0066cc">samples from eight donors. Across all events, FlowMOP excluded 3.3–56.7% of</span> the <span style="color:#c00000"><s>difficult-sample tumour analysis</s></span><span style="color:#0066cc">matched debris input (mean 18.3%), compared with 6.7–43.8% for Expert Manual (mean 24.8%), and therefore removed debris-gated events in every paired input. These percentages summarize total debris-gate removal and</span> are <span style="color:#c00000"><s>addressed as biological-validation questions</s></span><span style="color:#0066cc">distinct from the downstream population values plotted</span> in <span style="color:#c00000"><s>Combined Comment 3. We cross-reference those analyses here rather than repeating them as part</s></span><span style="color:#0066cc">Figure 6B,C. FlowMOP retained mean counts equivalent to 91.4–94.9%</span> of <span style="color:#0066cc">matched Raw across the Live CD45+ reference, B, T, and NKT endpoints, compared with 96.3–98.8% for Expert Manual; no count endpoint differed directly between</span> the <span style="color:#c00000"><s>debris-method description.</s></span><span style="color:#0066cc">methods. No population frequency differed from Expert Manual except T-cell frequency, which was lower after FlowMOP cleaning (Raw-normalized mean 1.013 versus 1.069; Holm-adjusted p = 0.037). Figure 6 reports the downstream plots, and Supplementary Figure S8 shows the representative debris and doublet preprocessing projections. The source-labelled synthetic debris benchmark separately provides the objective assessment of targeted debris removal.</span>

The synthetic preparation is now described explicitly. Approximately equal event numbers were sampled from the high-debris and low-debris sources and concatenated into matched mixtures while retaining source labels. These source labels provide the objective basis for the reported post-cleaning proportions.

The requested human comparison is<span style="color:#c00000"><s>already</s></span> represented by the percentage-based analysis in Figure 3. FlowMOP's enrichment of the labelled low-debris component is compared directly with four expert gates, with mean percentages, variability, and paired statistical comparisons. Because the input event count differs between samples, percentage-based retention and enrichment are more directly comparable than raw removed-event counts.

<span style="color:#0066cc">FlowMOP determines its debris gate independently for each sample, whereas manual debris gating is commonly performed groupwise by applying one gate across related samples. We therefore also asked the experts to perform both groupwise and individual-sample gating. No difference was detected between these strategies for any expert (Fig. 3D, unadjusted paired t-tests, p > 0.05), indicating that the comparison with FlowMOP was not driven by this difference in gating strategy.</span>

### Changes made

- We described the synthetic debris mixture as follows:

  > “For assessment of FlowMOP debris-gating performance, high-debris and low-debris samples were combined by sampling approximately equal event numbers from each source and concatenating them into matched synthetic mixtures while retaining source labels for ground-truth quantification.”

- We clarified the scope and derivation of the FSC-A gate as follows:

  > “FlowMOP’s debris module targets small, low-FSC debris. The final gate is applied on FSC-A, but its threshold is informed by FSC-A distributions across eligible fluorescence-positive populations rather than the overall FSC-A histogram alone.”

  > “SSC-A may improve recognition of larger or internally complex debris, while pulse-width measurements may assist with aggregates. Accordingly, FlowMOP does not currently incorporate SSC-A or pulse-width measurements into its debris decision because broadly applicable thresholds for these features are difficult to establish across tissues, panels, instruments, and acquisition settings.”

  > “The median FSC-A threshold across all parameters is taken as the final FSC-A gate to be applied to the sample (Figure 1B).”

- We expanded the Discussion of SSC-A and pulse-width integration:

  > “SSC-A may improve recognition of larger or internally complex debris, while pulse-width measurements may assist with aggregates. Accordingly, FlowMOP does not currently incorporate these features because broadly applicable decision rules are difficult to establish across tissues, panels, instruments, and acquisition settings.”

<span style="color:#0066cc">- We clarified that generalized metadata-based margin removal is not currently implemented beyond the FSC-A maximum-value precleaning check and is a potential complementary extension rather than a complete debris classifier.</span>

<span style="color:#0066cc">- We added the PBMC biological-validation result showing that FlowMOP removed debris-gated events in every paired input, with variable removal across samples. The overall removal percentages are distinguished from the downstream population values plotted in Figure 6B,C. No evaluated debris count endpoint differed directly from Expert Manual; T-cell frequency was the only population frequency that differed (adjusted p = 0.037). We interpret this biological preservation analysis alongside the objective source-labelled debris benchmark.</span>

- We reported the existing expert comparison as follows:

  > “FlowMOP did not differ significantly from any human evaluator except Expert 4, for whom FlowMOP removed more labelled high-debris events (Bonferroni-adjusted paired t-test, p = <span style="color:#c00000"><s>0.04)</s></span><span style="color:#0066cc">0.049)</span> (Fig. 3C).”

<span style="color:#0066cc">- We clarified that FlowMOP estimates debris gates independently for each sample and reported the existing groupwise-versus-individual-sample control, for which no difference was detected for any expert (Fig. 3D).</span>

**Location:** Methods, “Synthetic Debris Sample Preparation and <span style="color:#c00000"><s>Generation”;</s></span><span style="color:#0066cc">Generation,” “Human PBMC biological-validation sample preparation,” and “Biological-validation analysis”;</span> Results, “Debris <span style="color:#c00000"><s>Gating” and</s></span><span style="color:#0066cc">Gating,”</span> “Synthetic Debris Gating <span style="color:#c00000"><s>Benchmark”;</s></span><span style="color:#0066cc">Benchmark,” and “Biological validation”;</span> Discussion, “Synthetic Sample Debris and Doublet <span style="color:#c00000"><s>Gating”;</s></span><span style="color:#0066cc">Gating” and “Biological validation”; Figures 3 and 6; Supplementary</span> Figure <span style="color:#c00000"><s>3.</s></span><span style="color:#0066cc">S8.</span>

## Combined Comment 11 — Aggregates, Doublets, Saturation, and Validation

**Raised by:** Reviewer 2, Comments 5, 25, and 26.

### Reviewer 2, Comment 5

> Preprocessing sometimes include aggregates removal. Is it possible for FlowMOP to handle this task?

### Reviewer 2, Comment 25

> The FlowMOP creates a histogram of the FSC-A/FSC-H ratio to gate doublets. I also have a concern about this method. One unmentioned hypothesis of this algorithm is that all cells fall within the measurement range of FSC. However, it is very common that some cells exceed the FCS limit and collapsed in the edge of a FSC-A vs FSC-H scatter plot. For example, big myeloids in PBMC, if the target population is lymphocytes. Such population is expected to lead to weird histogram of the FSC-A/FSC-H ratio. This can be difficult to be addressed by the FlowMOP and are not mentioned in the study.

### Reviewer 2, Comment 26

> The authors use CTV-CFSE double positive cells to represent doublets. However, there are probably CTV- CTV doublets and CFSE - CFSE doublets, which are missed in the performance evaluation. I would like to see the doublet remove rate calculated against the human expert.

### Response

FlowMOP's doublet module targets events whose FSC-A/FSC-H and, where available, SSC-A/SSC-H pulse ratios distinguish them from singlets. It can remove some higher-order coincident events when those events produce separable ratio structure, as observed for triplet-like events in the technical controls, but it cannot identify every aggregate or clump from these ratios alone.

The general requirement for appropriate acquisition settings is addressed in Combined Comment 8. For the doublet module specifically, FSC-A/FSC-H and SSC-A/SSC-H ratios must remain informative. Saturated or edge-collapsed scatter measurements may therefore require acquisition review, manual intervention, or alternative pulse-shape features. We did not add a new saturation-handling algorithm.

The reviewer is also correct that CTV-CFSE double-positive events represent<span style="color:#c00000"><s>only</s></span> heterologous labelled <span style="color:#c00000"><s>doublets.</s></span><span style="color:#0066cc">doublets, with rare dye-transfer events as a possible exception.</span> CTV-CTV and CFSE-CFSE doublets cannot be distinguished from their respective single-labelled populations using the dye labels alone. The technical-control design therefore gives a sensitive measure of retained known heterologous doublets but cannot establish the complete false-positive and false-negative rate for all doublet classes.

Figure 4 already compares the percentage of CTV-CFSE double-positive events removed by FlowMOP with the corresponding percentages removed by human experts. We now describe this endpoint and its same-label limitation explicitly rather than implying complete doublet ground truth.

<span style="color:#0066cc">FlowMOP also estimates its doublet gates independently for each sample rather than applying a shared gate across a group. The experts therefore performed both groupwise and individual-sample doublet gating. These strategies did not differ for three of the four experts; Expert 3 removed fewer doublets with individual-sample gating (Fig. 4C, paired t-test, p = 0.009). We now explain this methodological distinction and its practical implications in the Results and Discussion.</span>

<span style="color:#0066cc">We additionally compared FlowMOP with Expert Manual doublet cleaning in paired PBMC samples from eight donors. FlowMOP produced slightly lower mean retained counts across the Live CD45+ reference, B-, T-, and NKT-cell endpoints, but no doublet count or frequency endpoint differed significantly between methods (Figure 6B,C). When Time, Debris, and Doublet preprocessing were combined, no count endpoint differed; B-cell frequency was the only frequency that differed between FlowMOP and Expert Manual (adjusted p = 0.049). Supplementary Figure S8 shows the representative doublet preprocessing projections.</span>
### Changes made

- We cross-reference the general acquisition-setting limitation in Combined Comment 8 and retain only the doublet-specific requirement that the relevant scatter ratios remain informative.

- We clarified the limits of the CTV/CFSE validation as follows:

  > <span style="color:#c00000"><s>“Events</s></span><span style="color:#0066cc">“Because the CTV- and CFSE-labelled populations were generated separately, events</span> positive for both <span style="color:#c00000"><s>CFSE and CTV provide an observable class of</s></span><span style="color:#0066cc">dyes were interpreted as</span> heterologous <span style="color:#c00000"><s>labelled doublets, subject to</s></span><span style="color:#0066cc">doublets;</span> rare dye-transfer <span style="color:#c00000"><s>events; same-label</s></span><span style="color:#0066cc">events are a possible exception. Same-label</span> CTV-CTV and CFSE-CFSE doublets are not identifiable from these labels alone.”

- We reported the comparison with human experts as follows:

  > “No statistically significant difference was detected between FlowMOP and any expert for this endpoint (Fig. 4B, paired t-test; unadjusted p > 0.05).”

<span style="color:#0066cc">- We clarified that FlowMOP estimates doublet gates independently for each sample and reported the existing comparison of groupwise and individual-sample expert gating (Fig. 4C).</span>

<span style="color:#0066cc">- We added the PBMC biological-validation comparison with Expert Manual. No doublet count or frequency endpoint differed significantly between methods; in the combined workflow, B-cell frequency was the only direct frequency difference (adjusted p = 0.049; Figure 6B,C). Supplementary Figure S8 shows the representative doublet preprocessing projections.</span>

- We retained the observation that FlowMOP removed a triplet-like population in the technical-control samples and restricted that finding to those samples.

**Location:** <span style="color:#0066cc">Methods, “Biological-validation analysis”;</span> Results, “Doublet <span style="color:#c00000"><s>Gating” and</s></span><span style="color:#0066cc">Gating,”</span> “Synthetic Doublet Gating <span style="color:#c00000"><s>Benchmark”;</s></span><span style="color:#0066cc">Benchmark,” and “Biological validation”;</span> Discussion, “Synthetic Sample Debris and Doublet <span style="color:#c00000"><s>Gating”</s></span><span style="color:#0066cc">Gating,” “Biological validation,”</span> and “Other remarks”; <span style="color:#0066cc">Figures 4 and 6; Supplementary</span> Figure <span style="color:#c00000"><s>4.</s></span><span style="color:#0066cc">S8.</span>

## Combined Comment 12 — CTV/CFSE Preparation Protocol

**Raised by:** Reviewer 2, Comment 27.

### Reviewer 2, Comment 27

> The protocol about how the CTV-CFSE samples are prepared is not clear. Are there wash steps in between? This is very important. It would be nice if the authors can provide detailed protocol as supplementary files.

### Response

We agree that the original description did not provide enough information to reproduce the CTV/CFSE preparation, particularly the wash and quenching steps. We have replaced the abbreviated description with a detailed protocol specifying the digestion conditions, filtration and RBC lysis, cell numbers, centrifugation and wash steps, dye stocks and volumes, labelling and quenching conditions, 1:1 recombination of the labelled populations, viability staining, and acquisition platform.

### Changes made

The revised Methods now state:

> “To generate samples with high proportions of doublets, C57BL/6 mouse spleens were injected with 1 mL digestion buffer comprising RPMI supplemented with 50 µg/mL collagenase P (Roche) and 10 µg/mL DNase I (Roche). Spleens were incubated for 20 minutes at room temperature (RT) in a further 1 mL of digestion buffer, mechanically dissociated, and incubated for a further 20 minutes at RT. Samples were passed through a 70-µm cell strainer with 10 mL FACS wash (PBS containing 2% heat-inactivated fetal bovine serum [FBS]), centrifuged (5 minutes, 500 × *g*, RT), and the supernatant discarded. Cell pellets were resuspended in 3 mL RBC lysis buffer and incubated for 3 minutes at RT, after which cells were washed twice with FACS wash and recovered by centrifugation.
>
> For dye labelling, 5 × 10^6 cells were transferred to each fresh 15-mL tube and stained with either CellTrace Violet (CTV; Invitrogen) or carboxyfluorescein succinimidyl ester (CFSE; eBioscience). Cells were centrifuged (5 minutes, 500 × *g*, 4°C), the supernatant was removed, and pellets were resuspended in 1 mL complete IMDM (cIMDM; Gibco IMDM supplemented with 10% heat-inactivated FBS, 100 U/mL penicillin, 100 µg/mL streptomycin, 2 mM L-glutamine, and 55 µM 2-mercaptoethanol). Two microlitres of 5 mM CTV or 10 mM CFSE were added to the side of the corresponding tube; samples were rapidly inverted and mixed, then incubated for 10 minutes at 37°C. Labelling was quenched by adding 5 mL ice-cold cIMDM and incubating for a further 5 minutes at RT. Cells were centrifuged and resuspended in cIMDM, after which 2 × 10^6 CTV-labelled cells and 2 × 10^6 CFSE-labelled cells were combined in a single sample and incubated for 30 minutes at 37°C and 5% CO2. Samples were centrifuged, the supernatant was removed, and cells were stained for 20 minutes at RT with Fixable Viability Dye eFluor 780 (Invitrogen; 1:1,000 in PBS). Cells were washed with PBS, centrifuged, resuspended in FACS wash, and acquired on a BD LSRII flow cytometer.”

**Location:** Methods, “Synthetic Doublet Sample Preparation and Generation”.

## Combined Comment 13 — Missing and Duplicated Methods Information

**Raised by:** Reviewer 2, Comments 11 and 12.

### Reviewer 2, Comment 11

> There is duplicated line in the Method “Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.”

### Reviewer 2, Comment 12

> In the “Preparation of Human PBMC Samples for Synthetic Time Benchmarking Samples” section, the authors mentioned there are many antibodies in the staining cocktail. However, they did not list which antibodies.

### Response

The duplicated cytometer sentence has been removed. The benchmarking uses real, existing datasets acquired with <span style="color:#c00000"><s>biology-specific</s></span><span style="color:#0066cc">experiment-specific, biology-driven</span> antibody <span style="color:#c00000"><s>panels.</s></span><span style="color:#0066cc">panels and instrument settings.</span> Consequently, the antigen–fluorochrome combinations differ among datasets and are not variables under evaluation; they are not material to the preprocessing <span style="color:#c00000"><s>questions</s></span><span style="color:#0066cc">comparisons</span> raised here. We have clarified this scope in the Methods rather than presenting a single fluorochrome list that would not describe the distinct panels used across the benchmark datasets.

### Changes made

- We removed the duplicated sentence, “Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.”

- We clarified why a single fluorochrome list is not applicable across the benchmark datasets:

  > “The benchmarking uses real, existing <span style="color:#c00000"><s>datasets</s></span><span style="color:#0066cc">datasets, each</span> acquired <span style="color:#c00000"><s>with biology-specific</s></span><span style="color:#0066cc">using experiment-specific, biology-driven</span> antibody <span style="color:#c00000"><s>panels.</s></span><span style="color:#0066cc">panels and instrument settings.</span> Consequently, the antigen–fluorochrome combinations <span style="color:#0066cc">and acquisition voltage/gain settings</span> differ among datasets and <span style="color:#c00000"><s>are</s></span><span style="color:#0066cc">were</span> not variables under <span style="color:#c00000"><s>evaluation; they are not material to the preprocessing questions addressed here.”</s></span><span style="color:#0066cc">evaluation.”</span>

**Location:** Methods, <span style="color:#c00000"><s>“Preparation of Human PBMC Samples for Synthetic Time Benchmarking Samples.”</s></span><span style="color:#0066cc">“Synthetic time source-sample preparation.”</span>

## Combined Comment 14 — Precleaning Event Removal and Threshold

**Raised by:** Reviewer 2, Comment 17.

### Reviewer 2, Comment 17

> In the “Precleaning” section of Results, how many events are being removed in these used datasets? This information can be provided as supplementary table/figure to justify the set of threshold (5%).

### Response

The current implementation uses a 1% threshold, not 5%; the inconsistent earlier value has been corrected. FlowMOP checks the number of events at the maximum FSC-A value and removes those maximum-valued events only when they exceed 1% of the sample.

<span style="color:#c00000"><s>The number and percentage of events removed by</s></span><span style="color:#0066cc">We calculated</span> this<span style="color:#c00000"><s>specific precleaning</s></span> step <span style="color:#c00000"><s>are</s></span><span style="color:#0066cc">directly from</span> the <span style="color:#c00000"><s>relevant quantities. A dataset-level summary of those</s></span><span style="color:#0066cc">passed_lod channel, treating</span> values <span style="color:#c00000"><s>is</s></span><span style="color:#0066cc">below 0.5 as removed. The rule did</span> not <span style="color:#c00000"><s>yet</s></span><span style="color:#0066cc">activate in PBMC samples from any of the eight donors. In the three tumour samples, it removed 3,472–9,206 events, corresponding to 2.39–3.05% of each input. The</span> complete <span style="color:#0066cc">sample-level counts</span> and <span style="color:#c00000"><s>remains to be added before submission; percentage-based retention results from later gating stages</s></span><span style="color:#0066cc">percentages</span> are <span style="color:#c00000"><s>not presented as a substitute for this precleaning summary.</s></span><span style="color:#0066cc">reported in Supplementary Table S3.</span>

The 1% cutoff is <span style="color:#c00000"><s>a pragmatic safeguard rather than a uniquely ground-truth-derived boundary. It is low enough to identify files with a non-trivial accumulation of events at the acquisition maximum while avoiding removal triggered by isolated maximum-valued events. The revised Results state this</s></span><span style="color:#0066cc">an arbitrary</span> operational <span style="color:#c00000"><s>rationale.</s></span><span style="color:#0066cc">safeguard, not an empirically optimized biological threshold. Its activation depends on acquisition scaling and voltage/gain settings, and the observed activation rates are dataset-specific and should not be generalized across instruments.</span>

### Changes made

- We corrected and defined the threshold as follows:

  > “FlowMOP first checks the input file for events at the limit of detection, defined here as events at the maximum FSC-A value for that sample. If the number of events at this maximum exceeds a threshold (default 1%), FlowMOP removes these maximum-valued events. Otherwise, it retains all values.”

- We added the following operational <span style="color:#c00000"><s>rationale:</s></span><span style="color:#0066cc">rationale and dataset-specific qualification:</span>

  > “The 1% cutoff is <span style="color:#c00000"><s>a pragmatic safeguard intended to identify non-trivial accumulation at</s></span><span style="color:#0066cc">an arbitrary operational safeguard, not an empirically optimized biological threshold. In</span> the <span style="color:#0066cc">biological-validation inputs, the rule did not activate in any of the eight PBMC samples, whereas it removed 3,472–9,206 events, or 2.39–3.05%, in the three tumour samples (Table S3). Activation depends on</span> acquisition <span style="color:#c00000"><s>maximum without responding to isolated maximum-valued events.”</s></span><span style="color:#0066cc">scaling and voltage/gain settings, and these observed activation rates are dataset-specific and should not be generalized across instruments.”</span>

- <span style="color:#c00000"><s>The dataset-level precleaning removal counts and percentages remain to be</s></span><span style="color:#0066cc">We</span> added <span style="color:#c00000"><s>before submission.</s></span><span style="color:#0066cc">Supplementary Table S3 with the input events, events removed, percentage removed, and threshold-activation status for every biological-validation input.</span>

**Location:** Results, <span style="color:#c00000"><s>“Precleaning.”</s></span><span style="color:#0066cc">“Precleaning”; Supplementary Table S3.</span>

## Combined Comment 15 — Figure 1 Axes and Annotations

**Raised by:** Reviewer 2, Comments 19 and 23.

### Reviewer 2, Comment 19

> The Figure 1A missing many axis titles. Also, there are tow subplots after time-binned median fluorescence. What are these two subplots are not clearly annotated.

### Reviewer 2, Comment 23

> The Figure 1B also misses many axis titles. The Figure 1C misses y-axis title.

### Response

Figure 1 remains a conceptual schematic, and its axes are not treated as quantitative measurements. The revised legend explains the two smoothing resolutions in Figure 1A, identifies the debris and doublet-ratio quantities in Figures 1B and 1C, and states that the schematic fluorescence-intensity and signal-strength axes use arbitrary units.

### Changes made

- We revised the Figure 1 legend as follows:

  > “Figure 1. Conceptual schematics depicting FlowMOP’s time-gating (A), debris-gating (B), and doublet-gating (C) methods; plotted fluorescence intensity and signal-strength axes are schematic and use arbitrary units. A) FlowMOP selects valid parameters using positive-peak detection, generates time-binned fluorescence summaries, applies two smoothing resolutions, and performs robust outlier rejection across acquisition order. B) FlowMOP applies the same valid-parameter check, derives a candidate FSC-A threshold from each eligible parameter’s positive events, and uses the median candidate as the final FSC-A gate. C) FlowMOP identifies doublets from inflection points in FSC-A/FSC-H and SSC-A/SSC-H ratio histograms, with a fixed-ratio fallback.”

**Location:** Figure 1 legend.

## Combined Comment 16 — Abstract, Introduction, and Overall Manuscript Structure

**Raised by:** Reviewer 2, general assessment and Comments 2 and 9.

### Reviewer 2 — General Assessment (Manuscript Structure)

> Also, the manuscript is not well structured. Careful edition by experienced researcher is needed to increase the readability of the manuscript.

### Reviewer 2, Comment 2

> The structure of abstract is kind of odd. The first paragraph should focus on background instead of providing a conclusion.

### Reviewer 2, Comment 9

> The introduction is missing an end paragraph for summary.

### Response

The manuscript has been considerably restructured and rewritten. The Abstract now begins with the general preprocessing problem and introduces FlowMOP directly. Temporal artifacts are then defined within the methodological summary before the validation design and principal findings are presented. We also added a concluding Introduction paragraph that summarizes FlowMOP's intended role and previews the time-gating comparison, debris and doublet validation, expert evaluation, and computational benchmark. Considering that the paper has been considerably restructured, we hope that the new format and writing address the reviewer's concerns.

### Changes made

- We reordered the Abstract so that it begins with the study context:

  > “Flow cytometry now generates high-parameter datasets whose scale and variability challenge manual preprocessing, leading to subjectivity and poor reproducibility. Here, we introduce FlowMOP, a Python-native framework that automates three major preprocessing steps—time-gating, debris removal, and doublet exclusion.”

- We added the following study summary to the end of the Introduction:

  > “Here, we introduce FlowMOP, an automated preprocessing tool for time-gating, debris removal, and doublet exclusion. We compare time-gating performance with PeacoQC and FlowCut, evaluate debris and doublet removal using synthetic ground-truth datasets and expert comparison, and benchmark computational scalability.”

- We reorganized and edited the manuscript so that the motivation, algorithmic design, technical benchmarks, biological validation, expert evaluation, and limitations are presented in a clearer progression.

**Location:** Abstract; final paragraph of the Introduction; structure and writing throughout the manuscript.

## Associate Editor's Closing Request

> Finally, I ask that the authors also provide thorough and complete responses to all comments raised by the other reviewers.

<span style="color:#c00000"><s>**Response:** Every comment from Reviewers 1 and 2 is covered in this response, and the coverage index below identifies the consolidated response containing each point. Completed revisions are described and quoted where available; author-supplied methodological details and the ongoing biological-validation analysis remain clearly marked as placeholders.</s></span>

<span style="color:#c00000"><s>**Location:** Not applicable.</s></span>

## Comment Coverage Index

### Associate Editor

| Associate Editor point | Response location |
| --- | --- |
| General assessment | Associate Editor's General Assessment |
| Algorithmic motivation and theoretical justification — existing alternatives and design rationale | Combined Comment 1 |
| Algorithmic motivation and theoretical justification — smoothing mechanism | Combined Comment 4 |
| Algorithmic versus implementation advantages | Combined Comment 2 |
| Expert preference rankings, downstream biological impact, and statistical interpretation | Combined Comments 3 and 6 |
| Smaller point — dataset scale and complexity | Combined Comment 7 |
| Smaller point — Bimix/Trimix and experimental relevance | Combined Comment 9 |
| Request for complete responses | Associate Editor's Closing Request |

### Reviewer 1

| Reviewer 1 comment | Response location |
| --- | --- |
| Comment 1 — Technical infrastructure and performance claims | Combined Comment 2 |
| Comment 2 — Debris gating methodology | Combined Comment 10 |
| Comment 3 — Refinement of human subjective rankings and dataset breadth | Combined Comments 3 and 6 |
| Comment 4 — Error costs and sensitivity trade-offs | Combined Comment 3 |
| Comment 5 — Parametric sensitivity analysis | Combined Comment 4 |

### Reviewer 2

| Reviewer 2 comment | Response location |
| --- | --- |
| General assessment | Combined Comments 3 and 16 |
| Comment 1 | Combined Comment 1 |
| Comment 2 | Combined Comment 16 |
| Comment 3 | Combined Comment 9 |
| Comment 4 | Combined Comment 9 |
| Comment 5 | Combined Comment 11 |
| Comment 6 | Combined Comment 1 |
| Comment 7 | Combined Comment 2 |
| Comment 8 | Combined Comment 1 |
| Comment 9 | Combined Comment 16 |
| Comment 10 | Combined Comment 3 |
| Comment 11 | Combined Comment 13 |
| Comment 12 | Combined Comment 13 |
| Comment 13 | Combined Comment 8 |
| Comment 14 | Combined Comment 9 |
| Comment 15 | Combined Comment 10 |
| Comment 16 | Combined Comment 6 |
| Comment 17 | Combined Comment 14 |
| Comment 18 | Combined Comment 5 |
| Comment 19 | Combined Comment 15 |
| Comment 20 | Combined Comment 4 |
| Comment 21 | Combined Comment 5 |
| Comment 22 | Combined Comment 10 |
| Comment 23 | Combined Comment 15 |
| Comment 24 | Combined Comment 10 |
| Comment 25 | Combined Comment 11 |
| Comment 26 | Combined Comment 11 |
| Comment 27 | Combined Comment 12 |
| Comment 28 | Combined Comment 8 |

## Closing Statement

The revisions and explicit limitations provide a clearer account of the manuscript's evidentiary basis and practical interpretation.
