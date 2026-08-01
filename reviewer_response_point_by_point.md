# Response to the Associate Editor and Reviewers

**Manuscript:** *FlowMOP: An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool*

We thank Dr. David Novo, the Associate Editor, and both reviewers for their careful and constructive assessment of our manuscript. Their comments have helped us clarify FlowMOP's algorithmic motivation, distinguish its algorithmic and implementation contributions, strengthen the validation and limitations, and improve the structure and accessibility of the manuscript.

The Associate Editor and reviewers raised several overlapping concerns. To avoid repeating the same response while ensuring that every point is addressed, closely related comments are reproduced verbatim and answered together below. A complete comment-coverage index is provided at the end of this response.

## Associate Editor's General Assessment

> Novel tools to expedite flow cytometry data analysis are always welcome, and I commend the authors for their effort in introducing these new methods. However, as currently submitted, the manuscript requires major revision. The reviewers have identified numerous areas requiring improvement, and I encourage the authors to address their comments carefully and comprehensively. In addition to the reviewers' critiques, I offer several observations of my own that I believe will strengthen the manuscript's suitability for publication.

**Response:** We thank the Associate Editor for recognizing the potential value of FlowMOP and for identifying the central issues that required attention. We have organized our revisions around the algorithmic motivation, comparative mechanism, computational benchmarking, interpretation of expert evaluations, methodological clarity, and the limits of the current approach. Detailed responses follow.

**Location:** Not applicable; detailed manuscript locations are provided under the specific responses below.

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

We agree that the original framing did not distinguish sufficiently between automated population-gating tools, time-quality-control tools, and FlowMOP's intended role as an integrated preprocessing workflow. We have removed the claim that FlowMOP is the first automated approach capable of debris cleaning. The revised Introduction discusses FlowCut and PeacoQC as time-dependent quality-control approaches and GateNet and UNITO as broader automated population-gating approaches. We now frame FlowMOP more narrowly as a training-free preprocessing workflow that combines time, debris, and doublet removal rather than as the first method capable of identifying debris-like populations.

We have also revised the motivation for the three FlowMOP modules. Time gating is based on positive-population fluorescence summaries across acquisition order, debris gating derives an FSC-A threshold from cross-parameter positive-population structure, and doublet gating uses FSC-A/FSC-H and SSC-A/SSC-H ratio structure. These algorithmic choices are now distinguished from computational implementation choices.

We agree with Reviewer 2 that simulated and real data are both commonly used in method validation. We have removed the oppositional and overly strong wording and now describe the synthetic samples specifically as event-labelled technical controls developed to provide objective ground truth for flow-cytometry preprocessing tasks.

### Changes made

- We removed the claim that FlowMOP is the first automated approach capable of debris cleaning and added the following clarification:

  > “Several tools automate cytometry population identification, including the model-based flowClust method [13], the neural-network method GateNet [14], and the bivariate-segmentation framework UNITO [15]; automated gating approaches are reviewed elsewhere [16]. These approaches can exclude debris-like populations, so FlowMOP is not presented as the first automated method capable of debris cleaning. Its narrower contribution is a Python-based, training-free preprocessing workflow that combines automated time-gating, debris removal, and doublet exclusion in a single headless tool. FlowCut and PeacoQC are the most direct comparators for its time-gating component because they primarily address time-dependent quality control.”

- We reframed the synthetic-data contribution as follows:

  > “The synthetic datasets used here are not intended to suggest that simulation-based validation is unique; rather, they provide event-level labels for estimating sensitivity and specificity in preprocessing tasks where real ground truth is otherwise difficult to define.”

- We added a concise description of the three algorithmic components to the Figure 1 legend:

  > “A) FlowMOP selects valid parameters using positive-peak detection, generates time-binned fluorescence summaries, applies two smoothing resolutions, and performs robust outlier rejection across acquisition order. B) FlowMOP applies the same valid-parameter check, derives a candidate FSC-A threshold from each eligible parameter’s positive events, and uses the median candidate as the final FSC-A gate. C) FlowMOP identifies doublets from inflection points in FSC-A/FSC-H and SSC-A/SSC-H ratio histograms, with a fixed-ratio fallback.”

**Location:** Introduction; Results, “Algorithmic design,” “Time Gating,” “Debris Gating,” and “Doublet Gating”; Figure 1 legend.

## Combined Comment 2 — Comparative Mechanism, Smoothing, and Parameter Fairness

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

We agree that the original attribution to smoothing was speculative. We removed that statement and replaced it with a matched mechanism analysis that preserves fluorescence, scatter, source labels, and event order while changing only the Time channel. Across 30 source-labelled Bimix, Trimix, and Segment files, FlowMOP was unchanged under both source-linked and random Time warping, whereas FlowCut's removal behaviour changed when local Time density was altered. The clearest loss occurred in Segment inputs under source-linked Time warping. This analysis supports a more specific interpretation: FlowMOP's time-gating decision is anchored to fluorescence-population summaries, whereas FlowCut can respond to acquisition-density changes even when fluorescence is unchanged.

We also clarified the comparison with PeacoQC separately. PeacoQC assesses the stability of local density-peak positions across acquisition bins. In mixed-source bins, finite sampling from multiple fluorescence distributions can make those local peak estimates unstable even when the events do not correspond cleanly to the source-labelled contaminating population. The revised Discussion presents this as a susceptibility of the benchmarked signal, not as evidence that PeacoQC is intrinsically incorrect.

We did not perform a comprehensive PeacoQC or FlowCut parameter-sensitivity sweep. We therefore do not claim that the mechanism benchmark establishes robustness to every possible competitor parameterization. The revised framing identifies the fixed comparison settings, removes or tempers general claims of superiority, and restricts the mechanistic conclusion to the behaviours directly tested by the matched Time-only benchmark.

The two smoothing resolutions are now described as complementary spline fits applied to the same time-bin fluorescence-summary series before MAD filtering. A bin can be detected through either smoothing pass; the manuscript no longer relies on the imprecise phrase “multiple types of aberrations.”

### Changes made

- We removed the speculative smoothing attribution and added the following matched comparison:

  > “To test the effect of flow-rate disturbances aligned with or independent of source-linked fluorescence changes, we altered only the Time channel while leaving fluorescence, scatter, source labels, and event order unchanged (Fig. S4). FlowMOP was unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shifted after Time-only perturbation.”

- We clarified the function of the two smoothing resolutions as follows:

  > “Two spline smoothing values, one small and one larger, are applied to the returned time-bin series before median absolute deviation (MAD) filtering.”

  > “The two smoothing resolutions target both shorter and more sustained deviations, while parameter voting limits removal driven by isolated noisy channels in higher-dimensional panels.”

- We added the following limitation concerning parameter sensitivity:

  > “All algorithms were compared using recommended or default settings, including fixed FlowMOP parameters, to reflect typical unsupervised use. We did not perform extensive parameter tuning for FlowCut or PeacoQC because the original method descriptions do not provide dataset-specific guidance for how such tuning should be performed. We therefore treat full cross-method parameter optimization as outside the scope of this validation.”

- We added the complete mechanism-benchmark methods and results as Figure S4.

**Location:** Methods, “Time-only acquisition-rate mechanism benchmark”; Results, “Time Gating” and “Synthetic Time Gating Benchmark”; Discussion, “Synthetic Sample Time Gating” and “Other remarks”; Figure S4.

## Combined Comment 3 — Parameter Voting and the Ten-Parameter Threshold

**Raised by:** Reviewer 2, Comments 18 and 21.

### Reviewer 2, Comment 18

> The authors say ”when >10 parameters are present, bins are rejected if they are flagged by two or more parameters”. Why “10” is used as threshold here?

### Reviewer 2, Comment 21

> The authors say, “parameter voting to ensure that higher dimensional sample sets are not overly degraded.” It would be nice to show how bad the overly degradation will be is parameter voting is not applied. It is a little a bit concern that events with aberrant signal in one fluorescence are not detected, even in a big panel.

### Response

We thank the reviewer for highlighting that the original text implied a stronger theoretical basis than was warranted. The ten-parameter cutoff is an empirical operational threshold, not a mathematically derived boundary, and the manuscript now states this explicitly.

The rationale for requiring two channel flags in panels with more than ten eligible parameters is conservative. Biological datasets have a non-zero probability of isolated channel-specific outliers, and the opportunity for such outliers increases with panel dimensionality. Requiring corroboration from a second parameter reduces the risk that a single noisy channel removes otherwise valid biological events. We acknowledge the corresponding trade-off: an abnormality confined to one fluorescence channel may be retained in a high-dimensional panel. We did not add a separate voting-ablation analysis and have avoided presenting the rule as universally optimal.

### Changes made

- We added the following rationale and limitation:

  > “For panels with more than 10 parameters, FlowMOP requires two or more parameters to flag a bin before rejection. This empirical safeguard reduces false-positive removal caused by isolated noisy channels in high-dimensional panels, although an aberration confined to one channel may consequently be retained (Figure 1A).”

- We did not add a voting-ablation experiment and do not present the threshold as universally optimal.

**Location:** Results, “Time Gating.”

## Combined Comment 4 — Dask, Algorithmic Contribution, Runtime, and Memory

**Raised by:** Associate Editor; Reviewer 1, Comment 1; Reviewer 2, Comment 7.

### Associate Editor — Algorithmic vs. Implementation Advantages

> Reviewer 1 raises several points regarding the use of the Dask framework, and I would like to extend that line of inquiry. Specifically, I would ask the authors to clarify whether there is something inherent to the FlowMop algorithm that makes it particularly well-suited to parallelization via Dask — as distinct from the implementation itself. This distinction matters considerably. Given current development tools, a competent programmer with AI tooling could likely port the existing FlowCut or PeacoQC code — neither of which is especially lengthy — to a Dask-based parallel implementation in a matter of hours. If that is the case, it raises the question of whether a substantial portion of FlowMop's reported advantage derives from the implementation rather than from any intrinsic algorithmic innovation, which would significantly diminish the novelty of the contribution. I would encourage the authors to focus their comparative analysis on algorithmic advantages, particularly in cases where existing algorithms present genuine structural barriers to parallelization.

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

### Reviewer 2, Comment 7

> The authors should introduce “Dask” before using the “Dask-based framework” in the introduction. Please keep in mind, most readers probably have no experience using Python at all. I use Python a lot myself. Still, I am not familiar with this “Dask” thing. Also, the Dask-based design is about calculation efficiency, instead of accuracy. Readers will be more interested in which algorithm is used for automatic gating, instead of how to speed up calculation. The authors did not really introduce the algorithm for gating in the introduction. But they use a whole paragraph on the Dask-based design.

### Response

We agree that the manuscript conflated implementation infrastructure with algorithmic contribution. We benchmarked FlowMOP without active Dask execution and found that Dask overhead did not improve the reported workload. Dask is therefore inactive for the reported benchmark, and we have removed Dask from the manuscript's novelty framing. We do not claim that Dask improves gating accuracy or that FlowMOP's algorithm is uniquely suited to Dask.

To address the empirical performance request, we added a clone-based real-FCS scalability benchmark using matched 36-channel inputs containing 10,000, 100,000, 300,000, 1,000,000, and 2,000,000 events. FlowMOP, PeacoQC, and FlowCut were run on the same inputs. Optional plotting, reporting, and output generation were disabled where supported, and FlowMOP's debris and doublet modules were disabled so that the shared time-gating task was timed fairly. Each condition included one warm-up and three measured repeats. Runtime and peak resident memory are reported as mean ± SD in Table 1.

The revised manuscript separates these computational results from algorithmic validation. The runtime and memory results support computational scalability only; the source-labelled and mechanism benchmarks address gating behaviour.

### Changes made

- We removed Dask from the manuscript's novelty and accuracy framing and specified the benchmark configuration as follows:

  > “For fair timing of the shared time-gating task, FlowMOP was run using local non-distributed execution, with debris and doublet removal disabled and annotated output FCS writing disabled. PeacoQC and FlowCut were run with optional plotting, reporting, and output generation disabled where supported.”

- We added matched runtime and peak-memory testing across five event counts:

  > “Computational scalability was evaluated using clone-based real-FCS scaling. A representative FCS file was subsampled/replicated to matched event counts of 10,000, 100,000, 300,000, 1,000,000, and 2,000,000 events while preserving the original 36-channel structure. FlowMOP, PeacoQC, and FlowCut were run on the same generated inputs for each size.”

- We reported the principal computational result as follows:

  > “The computational benchmark demonstrates that FlowMOP provides speed gains with lower peak RAM usage at larger event counts. This is most evident from 300,000 events onward, where FlowMOP had both the fastest mean runtime and lowest peak memory use.”

**Location:** Methods, “Computational scalability benchmark”; Results, “Computational scalability” and Table 1; Discussion, “Synthetic Sample Time Gating.”

## Combined Comment 5 — Expert Rankings, Terminology, Visualisation, and Bayesian Modelling

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

We agree that forced rankings measure relative preference and cannot determine whether a gate is absolutely acceptable for analysis. We have therefore replaced “benchmarking” with “expert comparison” or “expert evaluation,” removed the contorted “not least preferred” interpretation, and now state directly where FlowMOP ranked below the human gates. The ranking results are presented as exploratory evidence of expert preference rather than objective ground-truth evidence of gating quality or algorithmic superiority.

We also acknowledge that the design used a single relevant expert to rank each tissue type and therefore cannot estimate within-tissue consensus across multiple raters. We did not retrospectively collect a new absolute-quality scale because that would require re-running the expert assessment under a newly defined protocol. Instead, we explicitly describe this limitation and interpret the rankings alongside the synthetic technical-control results that contain event-level ground truth.

The visual summaries now include an Average Score column within each ranking grid rather than a separate panel, and the row axis is labelled “Gate Provided By.” The displayed scale now follows the standard convention that rank 1 is best. This display correction did not change the underlying ranking order or Bayesian analysis.

The Bayesian model is retained because the observed outcome is an ordinal ranking. The Plackett–Luce model retains the complete ordering and represents uncertainty in relative preference without treating differences between adjacent ranks as equal-interval continuous measurements. We have clarified that it estimates relative preference only. The subsection is presented as the statistical analysis of the expert-ranking data rather than as a method for generating non-synthetic samples.

### Changes made

- We defined the scope of the expert-ranking analysis as follows:

  > “The expert-ranking analysis was used to summarize relative preferences among gates generated by FlowMOP, comparator algorithms, and human operators; it was not treated as an absolute measure of gating adequacy.”

- We revised the Results terminology and interpretation as follows:

  > “FlowMOP’s outputs in these samples were compared with expert-provided gates using a forced-ranking preference task. These rankings measure relative expert preference and should not be interpreted as an absolute measure of gating adequacy.”

- We added the following limitations:

  > “One relevant expert ranked each tissue type, only four experts contributed gates, and the ordering does not distinguish a marginal preference from a judgement that a gate is unsuitable for analysis. Gate style could also reveal which outputs were algorithmic. We therefore interpret the ranking results as exploratory relative preferences, not as absolute quality scores or proof of algorithmic superiority or inferiority.”

- We retained the Plackett–Luce analysis and clarified that it models ordinal preference rankings rather than absolute gating quality.

- We regenerated Figures 5–7 as editable SVGs, changed the row-axis title to “Gate Provided By,” displayed rank 1 as best, and replaced the separate summary panel with an Average Score column in each grid.

- We revised the Figure 5 caption as follows:

  > “Figure 5. Expert preference rankings for time gates provided by four human experts, FlowMOP (black border), FlowCut, and PeacoQC across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. Abbreviations: DRG, dorsal root ganglion; CNS, central nervous system.”

- We clarified the scoring direction in the Results as follows:

  > “Rank 1 indicates the greatest preference.”

**Location:** Methods, “Statistical Analysis of Expert Preference Rankings”; Results, “Expert Preference Evaluation” and Figures 5–7; Discussion, “Biological Datasets: Expert Preference Evaluation.”

## Combined Comment 6 — Downstream Biological Impact and the Cost of Errors

**Raised by:** Associate Editor; Reviewer 1, Comment 4.

### Associate Editor — Expert Preference Rankings and Statistical Interpretation

> Alternatively — or in addition — the authors could evaluate the downstream biological impact of each gating strategy. By substituting FlowMop, FlowCut, and PeacoQC gates for the time and debris gating steps while retaining the human gating strategy for all other steps, one could directly assess whether differences in automated gating performance have any meaningful effect on the biological conclusions drawn from the data. Ultimately, the relevant question for end users is not whether an algorithm removes a few percent more or fewer debris events, but whether doing so materially affects downstream analysis.

### Reviewer 1, Comment 4 — Discussion of Error “Costs” and Sensitivity Trade-offs

> The manuscript notes that PeacoQC exhibits higher sensitivity but lower specificity compared to FlowMOP. This trade-off is central to clinical discovery and warrants a deeper discussion.
>
> Discussion Point: The authors should explicitly address the biological “cost” of these errors. For instance, is it more detrimental to the study's integrity to leave in a small transient artifact or to accidentally remove a rare, clinically significant biological population?

### Response

We agree that neither higher removal nor closer agreement with a single human gate is inherently preferable. Under-cleaning can retain acquisition artifacts that create, inflate, or obscure apparent biological populations. Conversely, over-cleaning can remove rare or transient biological events. The revised Discussion therefore interprets sensitivity and specificity as competing error costs rather than treating one direction of error as universally preferable.

The requested downstream biological-impact analysis remains a PLACEHOLDER.

### Changes made

- We added the following discussion of error costs:

  > “The biological cost of preprocessing errors is difficult to measure directly, and neither under-cleaning nor over-cleaning is preferable. Under-cleaning may allow acquisition-time artifacts with abnormal staining patterns to confound downstream results, including by creating, inflating, or obscuring apparent rare populations. Conversely, over-cleaning could remove rare or transient biological populations.”

- We added the following limitation concerning downstream effects:

  > “Whether the observed gating differences materially alter downstream biological conclusions remains an important question for future work.”

- The requested downstream biological-impact analysis remains a PLACEHOLDER.

**Location:** Discussion, “Biological Datasets: Expert Preference Evaluation” and “Other remarks”; downstream biological-impact analysis: PLACEHOLDER.

## Combined Comment 7 — Dataset Scale, Complexity, and Tumor Validation

**Raised by:** Associate Editor; Reviewer 2, general assessment and Comment 10.

### Associate Editor — Smaller Point

> In the Introduction, the authors state that sophisticated analysis software has contributed to datasets of increasing size and complexity. It is not clear to me how analysis software makes debris or time gating more complex, particularly when pre-processing is predominantly scatter-based. Do the authors perhaps mean that derived parameters from imaging cytometry increase pre-processing complexity? Similarly, the claim that “datasets exceed human capabilities” presumably refers to the sheer number of files rather than the complexity of individual datasets — this should be stated explicitly. It is also not apparent from the manuscript how the tools described here depend on data complexity per se.

### Reviewer 2 — General Assessment

> The authors developed An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool. This can be a useful tool for cytometry. However, the dataset used in the paper are relatively “easy” cases. I recommend the authors to challenge the FlowMOP with more difficult but common samples like digested tumor sample. Performance against human expert is needed. Also, the manuscript is not well structured. Careful edition by experienced researcher is needed to increase the readability of the manuscript.

### Reviewer 2, Comment 10

> The study did not validate the FlowMOP on tumor datasets, which is a common sample type for flow cytometry but difficult to perform debris removal. I highly suggest the authors to include a tumor dataset, even if the performance is not optimal, the readers can know whether the method can be applied in their sample or not.

### Response

We agree that the original wording conflated dataset scale with the intrinsic complexity of an individual preprocessing gate. The revised framing uses “beyond practical manual capacity” to refer to the number of files, total event volume, and panel dimensionality encountered in large studies. We no longer imply that analysis software itself necessarily makes a debris or time gate more complex.

We also agree that tumor digests are an important and difficult validation setting. A sufficiently controlled tumor dataset with an appropriate event-level reference was not available for this revision. We have therefore limited the scope of our claims rather than implying validation in this context. The Discussion now identifies tumor digests, necrotic material, large debris, aggregates, and heterogeneous scatter profiles as important future validation cases that may challenge the present conservative FSC-A-centred debris strategy.

### Changes made

- We clarified the distinction between study scale and individual-gate complexity as follows:

  > “Modern flow-cytometry studies can contain increasing numbers of files, events, and measured parameters [2]. Although this scale does not necessarily make time or debris gating more complex within an individual file, it can make repeated manual preprocessing impractical and can amplify variability between operators.”

- We added the following tumor-digest and complex-debris limitation:

  > “Tumor digests and other high-debris tissues remain important future validation contexts because necrotic debris, aggregates, and heterogeneous scatter profiles may challenge a conservative FSC-A-centered debris strategy.”

- We did not add a tumor dataset and do not claim validation on tumor samples.

**Location:** Introduction; Discussion, “Synthetic Sample Debris and Doublet Gating.”

## Combined Comment 8 — Acquisition Settings and Newer Scatter Configurations

**Raised by:** Reviewer 2, Comments 13 and 28.

### Reviewer 2, Comment 13

> The voltage set of the cytometer significantly impact the resolution of debris. If the voltage is relatively low with bad resolution, can FlowMOP perform the automatic task well? Also, please describe the voltage setting in the data generation section in the Method.

### Reviewer 2, Comment 28

> The Xenith flow cytometry form Thermo Fisher provides 405 FSC/SSC, 488 FSC/SSC, and polar 488 FSC/SSC. This provides more flexible approach to gate target population. How can the FlowMOP adapt to this kind of flow data? It would be nice to discuss this in the Discussion.

### Response

The acquisition voltage/gain settings remain a PLACEHOLDER.

We agree that instruments offering multiple scatter measurements introduce both opportunities and configuration choices not covered by the present validation. FlowMOP currently expects the user to identify the scatter channels used by the workflow; it has not been validated systematically across 405-nm, 488-nm, and polar scatter configurations. We therefore present adaptation to these instruments as a future validation and channel-selection task rather than claiming that the current results transfer automatically to every scatter configuration.

### Changes made

- The acquisition voltage/gain information remains a PLACEHOLDER.

- We added the following scope statement:

  > “Instruments providing multiple scatter measurements, including 405-nm, 488-nm, and polar 488-nm FSC/SSC configurations, were not evaluated here. Future versions could assess multiple scatter-channel pairs and select or combine the pair with the clearest debris/doublet separation.”

**Location:** Discussion, “Other remarks”; acquisition voltage/gain information: PLACEHOLDER.

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

Thank you. We defined temporal artifacts, simplified the terminology, and added Figure S1. We clarified that these samples model fluorescence changes across acquisition order without flow-rate disturbances, because rate changes without corresponding fluorescence changes should not prompt event exclusion. Flow-rate effects were tested separately, either aligned with source-linked fluorescence changes or independently of them. FlowMOP was unchanged in both conditions, whereas FlowCut's removal behavior shifted (Figure S4).

### Changes made

- In the Abstract, we added the following definition:

  > “Temporal artifacts are acquisition-time-dependent deviations in event quality or fluorescence signal caused by blockages, bubbles, flow instability, or instrument instability.”

- In the Methods, we added the following clarification:

  > “Segmented samples model sustained changes, whereas Bimix and Trimix provide controlled tests of subtler mid-acquisition changes. Flow-rate disturbances without corresponding fluorescence changes should not prompt event exclusion; these samples therefore model fluorescence changes across acquisition order without introducing flow-rate disturbances. Flow-rate effects were tested separately by altering Time either in alignment with source-linked fluorescence changes or independently of them.”

- In the Results, we added the following comparison:

  > “To test the effect of flow-rate disturbances aligned with or independent of source-linked fluorescence changes, we altered only the Time channel while leaving fluorescence, scatter, source labels, and event order unchanged (Fig. S4). FlowMOP was unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shifted after Time-only perturbation.”

- We added Figure S1 with the following caption:

  > “Figure S1: Construction of Segment, Bimix, and Trimix synthetic time samples. No flow-rate disturbance was introduced. Source labels were retained for scoring only and excluded from algorithm input.”

**Location:** Abstract; Introduction; Methods, “Generation of Synthetic Time Samples” and “Time-only acquisition-rate mechanism benchmark”; Results, “Synthetic Time Gating Benchmark”; Figures S1 and S4.

## Combined Comment 10 — Abstract and Introduction Structure

**Raised by:** Reviewer 2, Comments 2 and 9.

### Reviewer 2, Comment 2

> The structure of abstract is kind of odd. The first paragraph should focus on background instead of providing a conclusion.

### Reviewer 2, Comment 9

> The introduction is missing an end paragraph for summary.

### Response

We agree. The Abstract has been restructured to begin with the preprocessing problem and its biological and computational context before introducing FlowMOP, the validation design, and the principal findings. We also added a concluding Introduction paragraph that summarizes FlowMOP's intended role and previews the time-gating comparison, debris and doublet validation, expert evaluation, and computational benchmark.

### Changes made

- We reordered the Abstract so that it begins with the study context:

  > “Flow cytometry now generates high-parameter datasets whose scale and variability challenge manual preprocessing, leading to subjectivity and poor reproducibility. Temporal artifacts are acquisition-time-dependent deviations in event quality or fluorescence signal caused by blockages, bubbles, flow instability, or instrument instability. This study introduces FlowMOP, a Python-native framework that automates three major preprocessing steps—time-gating, debris removal, and doublet exclusion.”

- We added the following study summary to the end of the Introduction:

  > “Here, we introduce FlowMOP, an automated preprocessing tool for time-gating, debris removal, and doublet exclusion. We compare time-gating performance with PeacoQC and FlowCut, evaluate debris and doublet removal using synthetic ground-truth datasets and expert comparison, and benchmark computational scalability.”

**Location:** Abstract; final paragraph of the Introduction.

## Combined Comment 11 — Debris Methodology and Validation

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

We agree that the method needed a more precise description. FlowMOP ultimately applies a one-dimensional FSC-A threshold, but that threshold is not obtained from a single unconditioned FSC-A histogram. The algorithm identifies eligible fluorescence parameters, examines the FSC-A structure of each parameter's positive population, derives a candidate FSC-A threshold from each eligible channel, and applies the median candidate threshold to the sample. Thus, information across eligible fluorescence channels informs the final FSC-A gate, while the applied decision remains a conservative FSC-A threshold.

We nevertheless agree that this does not make the current module a universal debris classifier. Debris and intact cells can overlap on FSC-A, and large clumps, tumor-associated material, non-biological aggregates, or other high-scatter contaminants may not be removed reliably. The revised Discussion defines the intended scope as conservative low-FSC debris removal and identifies SSC-A, pulse width, margin-event information, and sample-specific models as possible future extensions.

The synthetic preparation is now described explicitly. Approximately equal event numbers were sampled from the high-debris and low-debris sources and concatenated into matched mixtures while retaining source labels. These source labels provide the objective basis for the reported post-cleaning proportions.

The requested human comparison is already represented by the percentage-based analysis in Figure 3. FlowMOP's enrichment of the labelled low-debris component is compared directly with four expert gates, with mean percentages, variability, and paired statistical comparisons. Because the input event count differs between samples, percentage-based retention and enrichment are more directly comparable than raw removed-event counts.

### Changes made

- We described the synthetic debris mixture as follows:

  > “For assessment of FlowMOP debris-gating performance, high-debris and low-debris samples were combined by sampling approximately equal event numbers from each source and concatenating them into matched synthetic mixtures while retaining source labels for ground-truth quantification.”

- We clarified the scope and derivation of the FSC-A gate as follows:

  > “To debris gate, FlowMOP applies a conservative FSC-A-based threshold intended primarily to remove low-FSC debris. FSC-A was selected because low-forward-scatter material is a comparatively universal debris signal across sample types, whereas SSC-A patterns are more tissue- and instrument-dependent. This module is not intended to classify all possible debris morphologies.”

  > “The median FSC-A threshold across all parameters is taken as the final FSC-A gate to be applied to the sample (Figure 1B).”

- We reported the existing expert comparison as follows:

  > “FlowMOP did not differ significantly from any human evaluator except Expert 4, for whom FlowMOP removed more labelled high-debris events (Bonferroni-adjusted paired t-test, p = 0.04) (Fig. 3C).”

- We added the following limitation:

  > “Tumor digests and other high-debris tissues remain important future validation contexts because necrotic debris, aggregates, and heterogeneous scatter profiles may challenge a conservative FSC-A-centered debris strategy.”

**Location:** Methods, “Synthetic Debris Sample Preparation and Generation”; Results, “Debris Gating” and “Synthetic Debris Gating Benchmark”; Discussion, “Synthetic Sample Debris and Doublet Gating”; Figure 3.

## Combined Comment 12 — Aggregates, Doublets, Saturation, and Validation

**Raised by:** Reviewer 2, Comments 5, 25, and 26.

### Reviewer 2, Comment 5

> Preprocessing sometimes include aggregates removal. Is it possible for FlowMOP to handle this task?

### Reviewer 2, Comment 25

> The FlowMOP creates a histogram of the FSC-A/FSC-H ratio to gate doublets. I also have a concern about this method. One unmentioned hypothesis of this algorithm is that all cells fall within the measurement range of FSC. However, it is very common that some cells exceed the FCS limit and collapsed in the edge of a FSC-A vs FSC-H scatter plot. For example, big myeloids in PBMC, if the target population is lymphocytes. Such population is expected to lead to weird histogram of the FSC-A/FSC-H ratio. This can be difficult to be addressed by the FlowMOP and are not mentioned in the study.

### Reviewer 2, Comment 26

> The authors use CTV-CFSE double positive cells to represent doublets. However, there are probably CTV- CTV doublets and CFSE - CFSE doublets, which are missed in the performance evaluation. I would like to see the doublet remove rate calculated against the human expert.

### Response

FlowMOP is not intended to provide a general aggregate-removal solution. Its doublet module targets events whose FSC-A/FSC-H and, where available, SSC-A/SSC-H pulse ratios distinguish them from singlets. It can remove some higher-order coincident events when those events produce separable ratio structure, as observed for triplet-like events in the technical controls, but it cannot identify every aggregate or clump from these ratios alone.

We agree that informative pulse ratios require the relevant scatter measurements to remain within the instrument's measurement range. If cells are saturated or collapsed at an FSC boundary, the lost pulse-shape information cannot be reconstructed downstream. The Methods now state this assumption and identify saturated or edge-collapsed scatter, large myeloid populations, and incorrectly configured acquisition parameters as conditions requiring acquisition review, manual intervention, or alternative pulse-shape features. We did not add a new saturation-handling algorithm.

The reviewer is also correct that CTV-CFSE double-positive events represent only heterologous labelled doublets. CTV-CTV and CFSE-CFSE doublets cannot be distinguished from their respective single-labelled populations using the dye labels alone. The technical-control design therefore gives a sensitive measure of retained known heterologous doublets but cannot establish the complete false-positive and false-negative rate for all doublet classes.

Figure 4 already compares the percentage of CTV-CFSE double-positive events removed by FlowMOP with the corresponding percentages removed by human experts. We now describe this endpoint and its same-label limitation explicitly rather than implying complete doublet ground truth.

### Changes made

- We added the following measurement-range limitation:

  > “FlowMOP's doublet module assumes that FSC-A/FSC-H and SSC-A/SSC-H ratios remain informative and that acquisition voltages and scatter parameters have been set appropriately. If relevant scatter channels are saturated, edge-collapsed, or incorrectly configured at acquisition, the lost pulse-shape information cannot be recovered by FlowMOP or by other downstream preprocessing algorithms; such samples require acquisition review, manual intervention, or alternative pulse-shape features where available.”

- We clarified the limits of the CTV/CFSE validation as follows:

  > “Events positive for both CFSE and CTV provide an observable class of heterologous labelled doublets, subject to rare dye-transfer events; same-label CTV-CTV and CFSE-CFSE doublets are not identifiable from these labels alone.”

- We reported the comparison with human experts as follows:

  > “No statistically significant difference was detected between FlowMOP and any expert for this endpoint (Fig. 4B, paired t-test; unadjusted p > 0.05).”

- We retained the limited observation that FlowMOP removed a triplet-like population in the technical-control samples, but we do not claim universal aggregate removal.

**Location:** Results, “Doublet Gating” and “Synthetic Doublet Gating Benchmark”; Discussion, “Synthetic Sample Debris and Doublet Gating” and “Other remarks”; Figure 4.

## Combined Comment 13 — CTV/CFSE Preparation Protocol

**Raised by:** Reviewer 2, Comment 27.

### Reviewer 2, Comment 27

> The protocol about how the CTV-CFSE samples are prepared is not clear. Are there wash steps in between? This is very important. It would be nice if the authors can provide detailed protocol as supplementary files.

### Response

PLACEHOLDER

### Changes made

PLACEHOLDER

**Location:** PLACEHOLDER

## Combined Comment 14 — Missing and Duplicated Methods Information

**Raised by:** Reviewer 2, Comments 11 and 12.

### Reviewer 2, Comment 11

> There is duplicated line in the Method “Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.”

### Reviewer 2, Comment 12

> In the “Preparation of Human PBMC Samples for Synthetic Time Benchmarking Samples” section, the authors mentioned there are many antibodies in the staining cocktail. However, they did not list which antibodies.

### Response

We thank the reviewer for identifying these omissions. The duplicated cytometer sentence has been removed.

**PBMC antibody identities and fluorochromes:** PLACEHOLDER

### Changes made

- We removed the duplicated sentence, “Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.”

- The complete PBMC antibody panel remains a PLACEHOLDER.

**Location:** Methods, “Preparation of Human PBMC Samples for Synthetic Time Benchmarking Samples”; complete PBMC antibody panel: PLACEHOLDER.

## Combined Comment 15 — Precleaning Event Removal and Threshold

**Raised by:** Reviewer 2, Comment 17.

### Reviewer 2, Comment 17

> In the “Precleaning” section of Results, how many events are being removed in these used datasets? This information can be provided as supplementary table/figure to justify the set of threshold (5%).

### Response

We thank the reviewer for raising this point. The current implementation uses a 1% threshold, not 5%; the earlier 5% wording was stale and has been corrected. FlowMOP checks the number of events at the maximum FSC-A value and removes those maximum-valued events only when they exceed 1% of the sample.

We agree that the removal rate is the relevant quantity for comparison. The samples contain different total event numbers, so raw removed-event counts are direct functions of file size and are not comparable across datasets without their denominators. We therefore report percentage-based removal and retention measures, which permit comparison across files and settings with different event counts. We respectfully decline to add a large supplementary table of absolute counts because it would not add information beyond the input size and reported rate.

The 1% cutoff is a pragmatic safeguard rather than a uniquely ground-truth-derived boundary. It is low enough to identify files with a non-trivial accumulation of events at the acquisition maximum while avoiding removal triggered by isolated maximum-valued events. The revised Methods state this operational rationale.

### Changes made

- We corrected and defined the threshold as follows:

  > “FlowMOP first checks the input file for events at the limit of detection, defined here as events at the maximum FSC-A value for that sample. If the number of events at this maximum exceeds a threshold (default 1%), FlowMOP removes these maximum-valued events. Otherwise, it retains all values.”

- We added the following operational rationale:

  > “The 1% cutoff is a pragmatic safeguard intended to identify non-trivial accumulation at the acquisition maximum without responding to isolated maximum-valued events.”

- We retained percentage-based reporting because the input files contain different total event numbers and respectfully declined to add a separate raw-count table.

**Location:** Results, “Precleaning.”

## Combined Comment 16 — Figure 1 Axes and Annotations

**Raised by:** Reviewer 2, Comments 19 and 23.

### Reviewer 2, Comment 19

> The Figure 1A missing many axis titles. Also, there are tow subplots after time-binned median fluorescence. What are these two subplots are not clearly annotated.

### Reviewer 2, Comment 23

> The Figure 1B also misses many axis titles. The Figure 1C misses y-axis title.

### Response

We agree that the conceptual nature of Figure 1 did not remove the need for clear descriptions. The Figure 1 legend now explains the two smoothing resolutions in Figure 1A, identifies the debris and doublet-ratio quantities in Figures 1B and 1C, and states that schematic fluorescence-intensity and signal-strength axes use arbitrary units.

### Changes made

- We revised the Figure 1 legend as follows:

  > “Figure 1. Conceptual schematics depicting FlowMOP’s time-gating (A), debris-gating (B), and doublet-gating (C) methods; plotted fluorescence intensity and signal-strength axes are schematic and use arbitrary units. A) FlowMOP selects valid parameters using positive-peak detection, generates time-binned fluorescence summaries, applies two smoothing resolutions, and performs robust outlier rejection across acquisition order. B) FlowMOP applies the same valid-parameter check, derives a candidate FSC-A threshold from each eligible parameter’s positive events, and uses the median candidate as the final FSC-A gate. C) FlowMOP identifies doublets from inflection points in FSC-A/FSC-H and SSC-A/SSC-H ratio histograms, with a fixed-ratio fallback.”

**Location:** Figure 1 legend.

## Associate Editor's Closing Request

> Finally, I ask that the authors also provide thorough and complete responses to all comments raised by the other reviewers.

**Response:** We agree. Every comment from Reviewers 1 and 2 is reproduced and addressed in this response. The coverage index below identifies the consolidated response containing each comment.

**Location:** Not applicable.

## Comment Coverage Index

### Associate Editor

| Associate Editor point | Response location |
| --- | --- |
| General assessment | Associate Editor's General Assessment |
| Algorithmic motivation and theoretical justification — existing alternatives and design rationale | Combined Comment 1 |
| Algorithmic motivation and theoretical justification — smoothing speculation | Combined Comment 2 |
| Algorithmic versus implementation advantages | Combined Comment 4 |
| Expert preference rankings and statistical interpretation | Combined Comments 5 and 6 |
| Smaller point — dataset scale and complexity | Combined Comment 7 |
| Smaller point — Bimix/Trimix and experimental relevance | Combined Comment 9 |
| Request for complete responses | Associate Editor's Closing Request |

### Reviewer 1

| Reviewer 1 comment | Response location |
| --- | --- |
| Comment 1 — Technical infrastructure and performance claims | Combined Comment 4 |
| Comment 2 — Debris gating methodology | Combined Comment 11 |
| Comment 3 — Refinement of human subjective rankings | Combined Comment 5 |
| Comment 4 — Error costs and sensitivity trade-offs | Combined Comment 6 |
| Comment 5 — Parametric sensitivity analysis | Combined Comment 2 |

### Reviewer 2

| Reviewer 2 comment | Response location |
| --- | --- |
| General assessment | Combined Comment 7 |
| Comment 1 | Combined Comment 1 |
| Comment 2 | Combined Comment 10 |
| Comment 3 | Combined Comment 9 |
| Comment 4 | Combined Comment 9 |
| Comment 5 | Combined Comment 12 |
| Comment 6 | Combined Comment 1 |
| Comment 7 | Combined Comment 4 |
| Comment 8 | Combined Comment 1 |
| Comment 9 | Combined Comment 10 |
| Comment 10 | Combined Comment 7 |
| Comment 11 | Combined Comment 14 |
| Comment 12 | Combined Comment 14 |
| Comment 13 | Combined Comment 8 |
| Comment 14 | Combined Comment 9 |
| Comment 15 | Combined Comment 11 |
| Comment 16 | Combined Comment 5 |
| Comment 17 | Combined Comment 15 |
| Comment 18 | Combined Comment 3 |
| Comment 19 | Combined Comment 16 |
| Comment 20 | Combined Comment 2 |
| Comment 21 | Combined Comment 3 |
| Comment 22 | Combined Comment 11 |
| Comment 23 | Combined Comment 16 |
| Comment 24 | Combined Comment 11 |
| Comment 25 | Combined Comment 12 |
| Comment 26 | Combined Comment 12 |
| Comment 27 | Combined Comment 13 |
| Comment 28 | Combined Comment 8 |

## Closing Statement

We again thank the Associate Editor and reviewers for their detailed and constructive comments. We believe that the revisions and the more explicit limitations substantially improve the manuscript's clarity, evidentiary basis, and practical interpretation.
