# FlowMOP: An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool

Running Title: Automated Python-based Sample Cleanup

**Authors:**

Tony Xu [1], N. A. Roberts [1], Felix Marsh-Wakefield [2,3], Rebecca A. Jaeger [1], Sarah Croft [1], Abhimanu Pandey [1], Dalton Leibold [4], Angela L. Ferguson [2,3], Lily Rodgers [2,3], Umaimainthan Palendira [3], Geoffrey W. McCaughan [2,5,6], Ben Quah [7], Robin Vlieger* [8], Anne Brüstle* [1]

**Affiliations:**

1: Department of Immunology and Infectious Disease, John Curtin School of Medical Research, the Australian National University, Canberra, Australian Capital Territory, Australia 
2: Liver Injury & Cancer Program, Cancer Innovations Centre, Centenary Institute, The University of Sydney, Sydney, New South Wales, Australia
3: Human Immunology Laboratory, School of Medical Sciences, Faculty of Medicine and Health, The University of Sydney, Sydney NSW, Australia	
4: Division of Ecology and Evolution, Research School of Biology, the Australian National University, Canberra, Australian Capital Territory, Australia 
5: A.W. Morrow Gastroenterology and Liver Centre, Royal Prince Alfred Hospital, Sydney NSW, Australia
6: Sydney Medical School , Faculty of Medicine and Health, The University of Sydney, Sydney NSW, Australia
7: Division of Genome Sciences and Cancer, John Curtin School of Medical Research, the Australian National University, Canberra, Australian Capital Territory, Australia
8: School of Medicine and Psychology, Australian National University, Canberra, Australian Capital Territory, Australia

## Acknowledgements

We thank Givanna Putri for her very helpful feedback and suggestions throughout the manuscript’s preparation. This work was also supported by computational resources provided by the Australian Government through the National Computational Infrastructure (NCI) under the ANU Merit Allocation Scheme.

**Funding information:**

TX is supported by the Australian Government Research Training Program PhD Scholarship. There are no other relevant sources of funding.

## Abstract

Flow cytometry now generates high-parameter datasets whose scale and variability challenge manual preprocessing, leading to subjectivity and poor reproducibility. Here, we introduce FlowMOP, a Python-native framework that automates three major preprocessing steps—time-gating, debris removal, and doublet exclusion. FlowMOP was developed to combine these preprocessing steps in a single workflow, whereas alternative preprocessing algorithms such as FlowCut and PeacoQC primarily address time-dependent quality control, and broader automatic gating frameworks are aimed at population identification rather than standardized preprocessing cleanup.

Methodologically, FlowMOP identifies temporal artifacts—acquisition-dependent deviations in event quality or fluorescence signal—via parameter-wise peak checks, bin-level fluorescence summaries across acquisition time, and robust outlier rejection. Debris is excluded by adaptive FSC-A thresholding derived from cross-parameter peak structure. Finally, doublets are removed using dynamic inflection detection on FSC-A/FSC-H and SSC-A/SSC-H ratio histograms. The implementation uses memory-conscious array operations; computational scaling was evaluated to 2 million events in a 36-channel file.

Against event-labelled synthetic technical controls, FlowMOP showed a favourable balance of sensitivity and specificity relative to comparator time-gating methods and effectively removed labelled debris- and doublet-enriched populations.

In a matched analysis of PBMC samples from eight donors, FlowMOP time gating preserved significantly more B, T, and NKT cells than expert preprocessing without altering their relative frequencies. Across the complete preprocessing pathway, no direct count difference between FlowMOP and expert cleaning was detected; B-cell frequency was modestly lower after FlowMOP, while the other evaluated frequencies did not differ. In three human liver-tumour samples with substantial debris, FlowMOP and manual preprocessing produced concordant downstream population measurements. Together with its synthetic accuracy and computational performance, these findings support FlowMOP as a fast, reproducible, fully automated preprocessing workflow for flow cytometry. FlowMOP can be accessed at https://github.com/1ordinateur/FlowMOP.

## Introduction

Modern flow cytometry studies can contain increasing numbers of files, events, and measured parameters [1]. The large scale of these data makes repeated manual preprocessing time-consuming and often impractical, potentially amplifying variability between operators. Reproducible automated preprocessing is therefore valuable for large studies.

Manual preprocessing of flow cytometry data, whilst still not standardised, typically involves three gating components. I) Time gating: operators perform time-gating to remove events potentially acquired erroneously due to transient or persistent artifacts in the instrument or sample, including air bubbles, blockages, or laser malfunctions. II) Debris gating: operators remove debris, which is generated by both the sample preparation process and inherent in the sample itself. Debris events are generally identified by reference to measured events with low size (determined by FSC-A (Forward SCatter – Area)) and internal complexity (determined by SSC-A (Side SCatter – Area)) [2]. III) Doublet gating: Events classified as "doublets"—two or more cells erroneously detected as one—are removed as their fluorescence measurements lack reliability. This is often done with reference to the ratio of the signal duration, and signal intensity. Doublets feature a signal peak strength comparable to a single cell, but for twice the duration. Hence, analysts may filter these events out by comparing an event’s total FSC Area relative to its FSC height (peak signal intensity) or by comparing the FSC signal height against the FSC signal width [3]. These manual preprocessing steps are time-consuming, inherently subjective, and susceptible to inconsistency across operators.

The most conspicuous time artifacts often occur at the beginning or end of acquisition, but flow-rate surges interspersed throughout acquisition and associated signal-intensity variation have also been documented [4]. Here, we use “microblockage” operationally to denote a short, self-resolving mid-acquisition disturbance that produces a localized fluorescence shift; the term does not imply that a physical obstruction was directly observed. Temporary acquisition problems can be difficult to detect manually [5], and manual identification and removal of transient problems can be time-consuming and subjective [6]. Because the affected intervals may be short and interspersed with otherwise plausible events, they can be impractical to exclude reproducibly using multiple manual gates.

Several tools automate cytometry population identification, including deep-learning approaches [7,8], the model-based flowClust method [9], the neural-network method GateNet [10], and the bivariate-segmentation framework UNITO [11], while FATE uses representation learning to produce generalized flow cytometry embeddings [12]; automated gating approaches are reviewed elsewhere [13]. Conversely, FlowMOP is a Python-based, training-free preprocessing workflow that combines automated time-gating, debris removal, and doublet exclusion in a single headless tool. FlowCut and PeacoQC provide the most direct comparison for its time-gating component.

The Python implementation specifically facilitates integration with contemporary machine-learning workflows, which predominantly utilize Python-based frameworks.

To date, cytometry preprocessing algorithms have been evaluated either by comparison to human-defined gold standards (as in FlowCut and PeacoQC) or through mathematical metrics (consider cyCombine’s approach to batch correction [14]). The synthetic datasets provide event-level labels for estimating sensitivity and specificity in preprocessing tasks where real ground truth is otherwise difficult to define.

Here, we introduce FlowMOP, an automated preprocessing tool for time-gating, debris removal, and doublet exclusion. We compare time-gating performance with PeacoQC and FlowCut, evaluate debris and doublet removal using synthetic ground-truth datasets and expert comparison, and benchmark computational scalability.

## Methods

For further details concerning datasets, see Table S1.

### Synthetic time benchmark

#### Synthetic time source-sample preparation

PBMC source samples used to generate the synthetic time benchmark were collected from healthy donors under ethics protocols ACT Health 2019.ETH.00081 and ANU HREC 2020/047. Blood was collected into ACD-A tubes and PBMCs isolated using Lymphoprep density gradient (Stemcell Technologies) and SepMate tubes (Stemcell Technologies) per manufacturer’s instructions prior to cryopreservation. After thawing, PBMCs were stained with ViaDye Red (Cytek Biosciences) and blocked with TruStain FcX (BioLegend) before antibody staining. Either 0.5, 1, or 3 × 10^6 cells were stained with a cocktail containing antibodies to several highly abundant antigens at less-than-saturating concentrations, yielding samples with different levels of fluorophore signal intensity for these common antigens. Other antibodies within the panel were present at saturating concentrations to yield similar fluorescence intensity across different cell seeding densities. Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer. The benchmarking uses real, existing datasets, each acquired using experiment-specific, biology-driven antibody panels and instrument settings. Consequently, the antigen–fluorochrome combinations and acquisition voltage/gain settings differ among datasets and were not variables under evaluation.

#### Computational construction of Segment, Bimix, and Trimix datasets

Following acquisition, events from these real, source-labelled FCS files were computationally recombined to construct three time-gating benchmark designs. In the ‘Segmented’ design, events from one or more source samples were appended to an existing sample. In the ‘Bimix’ design, events from two differently stained source samples were combined in randomly selected proportions (e.g. 40:60 or 75:25) using mixing-bin sizes of either 5000 or 2000 events. The ‘Trimix’ design similarly combined events from three differently stained source samples using 5000- or 2000-event mixing bins. Their construction is illustrated in Supplementary Figure S1, and the dataset compositions are reported in Supplementary Table S1. Here, we use “microblockage” operationally to denote a short, self-resolving mid-acquisition disturbance that produces a localized fluorescence shift; the term does not imply that a physical obstruction was directly observed. Segmented samples model sustained changes, whereas Bimix and Trimix model the observable fluorescence consequence in this operational definition by introducing short source-defined fluorescence shifts during acquisition. They do not recreate or establish the physical mechanism itself. The Bimix and Trimix files can appear acceptable on visual inspection because the altered intervals are short and interspersed with otherwise plausible events; however, the retained source labels identify the intentionally perturbed events that should be excluded under the benchmark definition. This provides event-level ground truth for a class of artifact that is difficult to recognize and impractical to gate manually [5,6]. Flow-rate disturbances without corresponding fluorescence changes should not prompt event exclusion; these samples therefore model fluorescence changes across acquisition order without introducing flow-rate disturbances. Flow-rate effects were tested separately by altering Time either in alignment with source-linked fluorescence changes or independently of them.

#### Time-only acquisition-rate mechanism benchmark

To distinguish acquisition-rate sensitivity from source-linked fluorescence and composition structure, we performed a matched mechanism benchmark using source-labelled smallcut synthetic-combo FCS files. Thirty high-count inputs were selected: ten Bimix, ten Trimix, and ten Segment files. For each file, 500,000 acquisition-order-preserving events were used without changing fluorescence, scatter, source labels, or event order. Bimix and Trimix files used the first 500,000 events; Segment files used a contiguous 500,000-event window centered on the source transition so that both segment sources were represented.

Each input generated three matched variants: raw, source-time-warped, and random-time-warped. In the source-time-warped variant, local Time increments were multiplied according to source identity using acquisition-interval multipliers spanning 1.0x to 20.0x. In the random-time-warped variant, the same multiplier range was assigned to contiguous 25,000-event chunks independently of source identity. After warping, the total Time range was rescaled to match the raw input so that the benchmark tested local acquisition-rate structure rather than total acquisition duration. FlowMOP, FlowCut, and PeacoQC were run on each matched input. Performance was evaluated relative to the matched raw file using sensitivity and specificity.

#### MAD-smoothing ablation and default selection

To test spline smoothing directly, FlowMOP was rerun across all 173 primary synthetic time-gating inputs with all non-smoothing settings fixed. Nineteen short/long smoothing-factor pairs, including a no-smoothing control, were compared by equally weighting the six benchmark groups. The current default (0.01,0.05) was selected as the marginally highest-scoring smoothed setting on this balanced comparison (Table S4). Figure 2 was regenerated using this setting. All completed quantitative FlowMOP results retained in the main figures, including the biological-validation figures, use the current selected configuration for the relevant module.

### Synthetic debris and doublet benchmark preparation

#### Mice

For generation of synthetic samples for FlowMOP validation (debris and doublet removal), splenocytes from 13-16-week old C57Bl/6N or C57Bl/6J mice were used. All animal experimentation was performed under ethics protocol 2024/379 at the Australian Phenomics Facility at the Australian National University, Canberra.

#### Synthetic Debris Sample Preparation and Generation

Mouse spleens were mechanically dissociated prior to lysis of red blood cells (RBC) with RBC lysis buffer (MilliQ H2O containing 150 mM NH4Cl, 10 mM KHCO3 and 1 mM EDTA). Splenocytes were then incubated with Fc block (BD), then stained with an antibody panel to delineate several major immune cell subsets (CD19 APC, CD3 PE, CD8 PE-Cy7, CD11b BB515, CD4 BV605, Fixable Viability Dye efluor780). To generate ‘high debris’ samples, cells were resuspended in MilliQ water and incubated for 2 minutes, prior to addition of 10X PBS to restore osmolarity. ‘Low debris’ samples were kept in isotonic solution throughout. Samples were acquired using a BD LSRII cytometer.

For assessment of FlowMOP debris-gating performance, high-debris and low-debris samples were combined by sampling approximately equal event numbers from each source and concatenating them into matched synthetic mixtures while retaining source labels for ground-truth quantification.

#### Synthetic Doublet Sample Preparation and Generation

To generate samples with high proportions of doublets, C57BL/6 mouse spleens were injected with 1 mL digestion buffer comprising RPMI supplemented with 50 µg/mL collagenase P (Roche) and 10 µg/mL DNase I (Roche). Spleens were incubated for 20 minutes at room temperature (RT) in a further 1 mL of digestion buffer, mechanically dissociated, and incubated for a further 20 minutes at RT. Samples were passed through a 70-µm cell strainer with 10 mL FACS wash (PBS containing 2% heat-inactivated fetal bovine serum [FBS]), centrifuged (5 minutes, 500 × *g*, RT), and the supernatant discarded. Cell pellets were resuspended in 3 mL RBC lysis buffer and incubated for 3 minutes at RT, after which cells were washed twice with FACS wash and recovered by centrifugation.

For dye labelling, 5 × 10^6 cells were transferred to each fresh 15-mL tube and stained with either CellTrace Violet (CTV; Invitrogen) or carboxyfluorescein succinimidyl ester (CFSE; eBioscience). Cells were centrifuged (5 minutes, 500 × *g*, 4°C), the supernatant was removed, and pellets were resuspended in 1 mL complete IMDM (cIMDM; Gibco IMDM supplemented with 10% heat-inactivated FBS, 100 U/mL penicillin, 100 µg/mL streptomycin, 2 mM L-glutamine, and 55 µM 2-mercaptoethanol). Two microlitres of 5 mM CTV or 10 mM CFSE were added to the side of the corresponding tube; samples were rapidly inverted and mixed, then incubated for 10 minutes at 37°C. Labelling was quenched by adding 5 mL ice-cold cIMDM and incubating for a further 5 minutes at RT. Cells were centrifuged and resuspended in cIMDM, after which 2 × 10^6 CTV-labelled cells and 2 × 10^6 CFSE-labelled cells were combined in a single sample and incubated for 30 minutes at 37°C and 5% CO2. Samples were centrifuged, the supernatant was removed, and cells were stained for 20 minutes at RT with Fixable Viability Dye eFluor 780 (Invitrogen; 1:1,000 in PBS). Cells were washed with PBS, centrifuged, resuspended in FACS wash, and acquired on a BD LSRII flow cytometer.

### Non-synthetic samples

#### Human PBMC biological-validation sample preparation

PBMC samples used for biological validation were prepared and acquired in the same manner as the PBMC source samples used to generate the synthetic time benchmark, except that they were stained with a different antibody panel designed to identify the populations evaluated in the downstream analysis.

#### Human liver-tumour samples

Human liver samples used in the non-synthetic validation datasets were collected under ethics approval from the Sydney Local Health District Ethics Review Committee (X19-0488 and 2019/ETH13790).

### Biological-validation analysis

The biological-validation analysis used PBMC samples from eight donors. Endpoint counts and frequencies were calculated within sample for each cleaning comparison. Counts and frequencies were each normalized to their corresponding matched ungated (Raw) input, such that Raw equalled 1 for both metrics: for time, the ungated input retained the expert-defined singlet and debris masks but applied no time mask; for debris, it retained the expert-defined time and doublet masks but applied no debris mask; for doublet, it retained the expert-defined time and debris masks but applied no doublet mask; and for the combined Time + Debris + Doublet comparison, no time, debris, or doublet preprocessing mask was applied. Live CD45+ was retained as a standalone reference endpoint and was expressed as a percentage of Live cells. Applying CD45+ as an additional parent gate excluded many low-scatter events before the lineage endpoints were quantified and therefore obscured the effects of debris removal. To expose those effects directly, we did not include CD45+ as a parent of the B-, T-, or NKT-cell endpoints. The expert-defined lineage coordinates were instead evaluated within Live cells: B cells were Live CD3−CD19+ events (Q1), T cells were Live CD3+CD19− events (Q3), and NKT cells were Live CD19−CD3+CD56+ events. Before Raw normalization, B-, T-, and NKT-cell frequencies were expressed as percentages of Live cells. Figure 5B (frequencies) and Figure 5C (counts) show the two major B- and T-cell populations and the less abundant NKT-cell population. Figure 6B (frequencies) and Figure 6C (counts) report the debris-only, doublet-only, and combined Time + Debris + Doublet comparisons; Supplementary Figure S8 retains the corresponding debris and doublet representative comparisons. Direct workflow comparisons used two-sided paired *t*-tests. Comparisons with Raw used two-sided one-sample *t*-tests against 1 on the within-sample Raw-normalized values, which is equivalent to a paired comparison with the matched Raw value. Holm adjustment was applied across the ten pairwise tests within each endpoint and metric for time gating and separately across the three tests within each endpoint, metric, and cleaning group for debris, doublet, and combined preprocessing. All analyses used n = 8.

### Tumour biological-validation analysis

Three human liver-tumour FCS files were analysed in their Raw state and after either manual or FlowMOP preprocessing. Manual preprocessing used sequential time, cells/debris, and single-cell gates. FlowMOP calculated time, debris, and doublet exclusions independently; the union of excluded events was removed, equivalently retaining the intersection of the three passed-event masks. The limit-of-detection mask was not included.

T and B cells were identified from CD3 and CD19 expression. T cells were defined as CD3+CD19− (Q1) and B cells as CD3−CD19+ (Q3). Three prespecified endpoints were calculated: the number of Live CD45+ cells and T- and B-cell frequencies as percentages of the original total event count. Each endpoint was normalized within sample to its matched Raw value (Raw = 100%). Raw, Manual, and FlowMOP values were compared using all three unadjusted, two-sided paired t-tests.

### Computational scalability benchmark

Computational scalability was evaluated using clone-based real-FCS scaling. A representative FCS file was subsampled/replicated to matched event counts of 10,000, 100,000, 300,000, 1,000,000, and 2,000,000 events while preserving the original 36-channel structure. FlowMOP, PeacoQC, and FlowCut were run on the same generated inputs for each size.

For fair timing of the shared time-gating task, FlowMOP was run using local non-distributed execution, with debris and doublet removal disabled and annotated output FCS writing disabled. PeacoQC and FlowCut were run with optional plotting, reporting, and output generation disabled where supported. Each condition was run once as a warm-up and then three measured times. Runtime and peak resident memory were recorded using `/usr/bin/time -v`; means and standard deviations were calculated across the three measured repeats.

### Coding Assistance

Parts of the code were generated with the aid of ChatGPT Codex and Claude Code. All code generated by LLMs was manually verified before implementation.

## RESULTS

![Embedded image 1](FlowMOP_submission_media/image1.png)

Figure 1. Conceptual schematics depicting FlowMOP’s time-gating (A), debris-gating (B), and doublet-gating (C) methods; plotted fluorescence intensity and signal-strength axes are schematic and use arbitrary units. A) FlowMOP selects valid parameters using positive-peak detection, generates time-binned fluorescence summaries, applies two smoothing resolutions, and performs robust outlier rejection across acquisition order. B) FlowMOP applies the same valid-parameter check, derives a candidate FSC-A threshold from each eligible parameter’s positive events, and uses the median candidate as the final FSC-A gate. C) FlowMOP identifies doublets from inflection points in FSC-A/FSC-H and SSC-A/SSC-H ratio histograms, with a fixed-ratio fallback.

### Algorithmic design

An overview of FlowMOP’s architecture is contained in Fig. 1, detailing approaches for its preprocessing, time-gating, debris removal, and doublet removal methods. This cleaned data can be applied to downstream analysis.

FlowMOP accepts `.csv`, `.fcs`, and Parquet files. It reduces event-level measurements into time-bin and channel summaries, applies smoothing and robust outlier detection, and combines the resulting flags through parameter voting.

#### Precleaning

FlowMOP first checks the input file for events at the limit of detection, defined here as events at the maximum FSC-A value for that sample. If the number of events at this maximum exceeds a threshold (default 1%), FlowMOP removes these maximum-valued events; otherwise, it retains all values. The 1% cutoff is an arbitrary operational safeguard, not an empirically optimized biological threshold. In the biological-validation inputs, the rule did not activate in any of the eight PBMC samples, whereas it removed 3,472–9,206 events, or 2.39–3.05%, in the three tumour samples (Table S3). Activation depends on acquisition scaling and voltage/gain settings, and therefore these observed activation rates are acquisition setting dependent. In ideal situations, no such activation of thresholding would be required. 

#### Time Gating

To time gate, FlowMOP builds upon the assumptions posited in PeacoQC and FlowCut regarding fluorescence fluctuations. That is, independent of flow rate variations, sections of acquired sample with aberrant positive fluorescence averages are the target portions to be removed. To achieve this, FlowMOP checks each parameter, excluding parameters with a unimodal distribution. ‘Unimodal distribution’ is presently defined as parameters with only one identifiable peak. Subsequently, for each fluorescence parameter that satisfies this criterion, FlowMOP excludes the first peak (selecting all subsequent peaks) and measures the average fluorescence value for each time bin. FlowMOP then can operate either in the ‘Positives’ mode, or ‘Geomean’ mode. In ‘Positives’ mode, all events before the first inflection point are discarded. All results shown presently operate in ‘Positives’ mode. In the ‘Geomean’ mode, all events are considered. Subsequently, on a per-parameter basis, the sample is transformed into bins grouped by time (the default being bin having minimum of 150 events, up to a maximum of 500 bins).

The median fluorescence of each bin’s cells is returned. Two spline smoothing values, one small and one larger (current default 0.01,0.05), are applied to the returned time-bin series before median absolute deviation (MAD) filtering. The smoothing factor scales the spline fit used for the binned fluorescence summary. Bins falling outside the MAD threshold in either smoothing pass are flagged for removal. Time-bins across all parameters are then combined, with time bins rejected if they have been flagged in any parameter. For panels with more than 10 parameters, FlowMOP requires two or more parameters to flag a bin before rejection. This empirical safeguard reduces false-positive removal caused by isolated noisy channels in high-dimensional panels, although an aberration confined to one channel may consequently be retained (Figure 1A).

FlowMOP summarizes each time bin using the median fluorescence of the selected positive population. If Geomean mode is selected, FlowMOP instead uses the geometric mean of all events in the bin. The two smoothing resolutions target both shorter and more sustained deviations, while parameter voting limits removal driven by isolated noisy channels in higher-dimensional panels.

#### Debris Gating

FlowMOP’s debris module targets small, low-FSC debris. The final gate is applied on FSC-A, but its threshold is informed by FSC-A distributions across eligible fluorescence-positive populations rather than the overall FSC-A histogram alone. SSC-A may improve recognition of larger or internally complex debris, while pulse-width measurements may assist with aggregates. Accordingly, FlowMOP does not currently incorporate SSC-A or pulse-width measurements into its debris decision because broadly applicable thresholds for these features are difficult to establish across tissues, panels, instruments, and acquisition settings. FlowMOP’s debris exclusion conducts a similar unimodality check on each fluorescence parameter, and the first peak is then excluded as the Time-gating feature. Thereafter, FlowMOP detects the global FSC-A peak as a reference point. For every parameter’s positive events, FlowMOP checks first for an FSC-A peak similar to the reference peak (default 30% of the reference peak’s value). If there is such a peak, it checks if the second FSC-A peak is the global maximum FSC-A peak. If that parameter’s positive cell’s second FSC-A is the global maxima, FlowMOP returns the FSC-A threshold as the minima between those two FSC-A peaks. If the second FSC-A is not the maximum, it returns the global interpeak minimum between the reference peak and maximal peak. Conversely, if there is no reference peak present in that parameter’s positive population, it selects the left-boundary of that parameter’s first peak. The median FSC-A threshold across all parameters is taken as the final FSC-A gate to be applied to the sample (Figure 1B).

#### Doublet Gating

To doublet gate, FlowMOP dynamically excludes sample doublets. To do this, FlowMOP creates a histogram of the FSC-A/FSC-H ratio. If there are multiple peaks all with a ratio of 1 or more, then it chooses the inflection point between those peaks, and excludes all events larger than that value. If there are insufficient peaks, it simply returns all events that have an FSC-A/FSC-H ratio smaller than a threshold (default 5). The process is repeated for the 
SSC-A/SSC-H variable. Consequently, FlowMOP is able to distill the implicit ratiometric information that current density based methodologies may overlook. FlowMOP's doublet module assumes that FSC-A/FSC-H and SSC-A/SSC-H ratios remain informative and that acquisition voltages and scatter parameters have been set appropriately. If relevant scatter channels are saturated, edge-collapsed, or incorrectly configured at acquisition, the lost pulse-shape information cannot be recovered by FlowMOP or by other downstream preprocessing algorithms; such samples require acquisition review, manual intervention, or alternative pulse-shape features where available.

### Algorithmic Validation

#### Synthetic Sample Benchmarking

The ability of FlowMOP to successfully detect time, debris, and doublet-perturbed data was first tested against the respective task’s synthetic datasets, namely the synthetically combined staining time samples, the high-debris + low debris samples, and the CTV / CFSE doublet samples.

For time-gating samples, sensitivity and specificity were reported for each benchmarked method using source labels as the reference. Target source(s) were defined as the source(s) with the largest filename-encoded mixture proportion; tied largest proportions were treated as co-targets. Sensitivity was defined as retained target-source events divided by all retained events. Specificity was defined as removed non-target-source events divided by all removed events.

![Figure 2](figs_data/figure_2.svg)

Figure 2. A) Representative flow cytometry plots showing CD3 fluorescence against time. The original synthetically generated sample is shown in column 1, with the resulting output following FlowMOP, PeacoQC, and FlowCut processing shown in the subsequent columns. The first row depicts a representative ‘segmentation’-based synthetic file. The second row shows a representative two-sample mixture with a 5000-event bin size. Frequency percentages shown are the percentage of cells left post-cleaning relative to the original synthetic sample (rounded to the nearest percentage point). B) Violin plots showing sensitivity and specificity, grouped by sample type (Segment, Bimix, and Trimix) and cleaning method (FlowMOP, PeacoQC, and FlowCut). Internal dashed lines indicate quartiles. The first and second rows represent mixing-bin sizes of 5000 events (n = 90: Segment, 33; Bimix, 33; Trimix, 24) and 2000 events (n = 83: Segment, 32; Bimix, 28; Trimix, 23), respectively. Brackets show significant Bonferroni-adjusted paired *t*-tests.

#### Synthetic Time Gating Benchmark

Several computational approaches address time-dependent quality control, including flowAI [4], PeacoQC [5], flowClean [15], and FlowCut [6]. FlowMOP combines a time-gating component with debris and doublet modules in one Python workflow. Here, its time-gating performance is evaluated against PeacoQC and FlowCut.

The Segment, Bimix, and Trimix samples were designed as complementary time-artifact stress tests with event-level source information for objective scoring (Fig. 2A). Sensitivity was defined as the proportion of retained events derived from the target source or sources, and specificity as the proportion of removed events derived from the non-target source or sources. Direct comparison with manual gating was not attempted because the short, interspersed intervals in the Bimix and Trimix samples are impractical to remove reproducibly by eye.

At the 5000-event bin size, FlowMOP had higher sensitivity than FlowCut in Segment (p < 0.001), Bimix (p < 0.001), and Trimix (p = 0.03) (Fig. 2B). PeacoQC also had higher sensitivity than FlowCut in Bimix (p = 0.03). FlowMOP had higher specificity than both PeacoQC and FlowCut in Segment (p = 0.03 and p = 0.009), Bimix (p < 0.001 and p = 0.001), and Trimix (p < 0.001 and p = 0.001), respectively.

At the 2000-event bin size, FlowMOP and PeacoQC had higher sensitivity than FlowCut in Segment (p < 0.001 and p = 0.007, respectively), while FlowMOP had higher specificity than FlowCut (p = 0.03). In Bimix, PeacoQC had lower sensitivity than FlowCut (p = 0.02) and lower specificity than both FlowMOP (p < 0.001) and FlowCut (p = 0.007). In Trimix, sensitivity did not differ significantly among methods, while PeacoQC had lower specificity than FlowMOP (p = 0.009) and FlowCut (p = 0.003). All comparisons were paired t-tests with Bonferroni correction, using each algorithm's fixed recommended or default settings.

To test the effect of flow-rate disturbances aligned with or independent of source-linked fluorescence changes, we altered only the Time channel while leaving fluorescence, scatter, source labels, and event order unchanged (Fig. S4). FlowMOP and PeacoQC were unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shifted after Time-only perturbation. Across all inputs, random Time warping reduced FlowCut's specificity by 11.45 percentage points relative to matched raw inputs. In Segment inputs, source-linked Time warping reduced FlowCut's sensitivity by 7.84 percentage points and specificity by 15.25 percentage points. These results support the interpretation that FlowCut responds to local acquisition-density structure even when fluorescence values are unchanged, whereas FlowMOP and PeacoQC are unaffected by rate-only variation in these inputs.

Dual-resolution smoothing changed the balance between sensitivity and specificity (Table S4). The no-smoothing control had the highest equal-weight balanced mean (0.7551), but its sensitivity was 1.92 percentage points lower than that of 0.01,0.05. Among smoothed settings, 0.01,0.05 had the highest balanced mean (0.7511), with specificity 2.70 percentage points lower than the no-smoothing control. We therefore selected 0.01,0.05 as the default and regenerated Figure 2 using this setting.

Representative plots for the 2000 bin Bimix method, and the 2000, and 5000 bin Trimix methods can be found in Supplementary Figure S2.

#### Computational scalability

Clone-based runtime and memory benchmarking showed that FlowMOP scaled favorably at larger event counts (Table 1). Values are mean ± SD across three measured repeats after one warm-up run. FlowMOP was fastest from 100,000 events onward and had the lowest peak RAM at all sizes except 100,000 events. At 1,000,000 events, FlowMOP completed in 10.69 ± 0.16 s, compared with 24.21 ± 0.23 s for PeacoQC and 30.34 ± 0.48 s for FlowCut, while using 658.0 ± 0.3 MB peak RAM compared with 1483.4 ± 86.8 MB and 1115.1 ± 0.3 MB. At 2,000,000 events, FlowMOP completed in 17.42 ± 0.88 s, compared with 43.18 ± 4.23 s and 60.61 ± 2.37 s, while using 839.5 ± 0.4 MB peak RAM compared with 2614.8 ± 0.1 MB and 1849.8 ± 0.3 MB.

**Table 1: Clone-based time-gating runtime and memory benchmark**

Mean runtime and peak RAM across three measured repeats are shown as mean ± SD. The best-performing runtime and RAM value for each event count is shown in bold.

| Events | FlowMOP time (s) | PeacoQC time (s) | FlowCut time (s) | FlowMOP RAM (MB) | PeacoQC RAM (MB) | FlowCut RAM (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 2.67 ± 0.50 | 4.40 ± 0.57 | **1.72 ± 0.02** | **237.9 ± 0.2** | 446.6 ± 0.2 | 261.8 ± 0.3 |
| 100,000 | **4.34 ± 0.11** | 13.74 ± 1.34 | 4.78 ± 0.33 | 351.6 ± 0.1 | 606.6 ± 26.1 | **326.4 ± 0.1** |
| 300,000 | **8.42 ± 0.85** | 15.08 ± 0.84 | 10.91 ± 0.49 | **462.3 ± 0.4** | 948.5 ± 5.9 | 505.4 ± 0.4 |
| 1,000,000 | **10.69 ± 0.16** | 24.21 ± 0.23 | 30.34 ± 0.48 | **658.0 ± 0.3** | 1483.4 ± 86.8 | 1115.1 ± 0.3 |
| 2,000,000 | **17.42 ± 0.88** | 43.18 ± 4.23 | 60.61 ± 2.37 | **839.5 ± 0.4** | 2614.8 ± 0.1 | 1849.8 ± 0.3 |

#### Synthetic Debris Gating Benchmark

![Figure 3](figs_data/figure_3.svg)

Figure 3. A) Representative flow cytometry plots showing FSC-A/SSC-A debris plots. The first plot represents a representative combined high + low debris sample, the second and third represent the high debris portion and the low debris portion of that sample respectively. The percentages below denote the proportion of the combined sample represented by the annotated plot. B) FSC-A/SSC-A flow plots of the Fig. 3A’s representative sample post-processing. Again, the first column shows the combined high + low debris sample, the second and third show the high debris and low debris samples separately respectively. The percentages in the first column denote the proportion of the sample remaining relative to the original combined sample. Percentages in the high debris and low debris columns denote the post-filtering proportion that each comprises (rounded to the nearest percentage). C) Mean low-debris proportion ± SD after FlowMOP and four human-expert gates. FlowMOP had a significantly higher low-debris proportion than Expert 4 (Bonferroni-adjusted paired *t*-test, p = 0.049). D) Mean low-debris proportion ± SD after human-expert debris gating using either a shared groupwise gate or gates drawn independently for each sample. Colours identify experts; solid and hatched bars denote groupwise and individual-sample gating, respectively. No difference was found between the two approaches (unadjusted paired *t*-tests, p > 0.05).

FlowMOP’s debris gating performance was tested by its ability to deplete the high-debris component in the combined high- and low-debris samples (Fig. 3A). FlowMOP reduced the high-debris source proportion from 50 ± 1.10% to 28.8 ± 0.1% (paired t-test, p < 0.001) (Fig. 3B). The final gate is applied on FSC-A, but its threshold is derived from candidate FSC-A distributions across eligible fluorescence-positive populations rather than from the overall FSC-A histogram alone. FlowMOP did not differ significantly from any human evaluator except Expert 4, for whom FlowMOP removed more labelled high-debris events (Bonferroni-adjusted paired t-test, p = 0.049) (Fig. 3C).

FlowMOP determines its debris gate independently for each sample, whereas manual debris gating is commonly performed groupwise by applying one gate across related samples. To evaluate whether this difference in strategy affected the comparison, human experts performed both groupwise and individual-sample gating. No difference was found between the two approaches for any expert (Fig. 3D, unadjusted paired t-test p > 0.05).

#### Synthetic Doublet Gating Benchmark

FlowMOP’s doublet removal performance was also examined through synthetic technical controls (Fig. 4A). Because the CTV- and CFSE-labelled populations were generated separately, events positive for both dyes were interpreted as heterologous doublets; rare dye-transfer events are a possible exception. Same-label CTV-CTV and CFSE-CFSE doublets are not identifiable from these labels alone. FlowMOP was then run on these samples, and the proportion of remaining CFSE/CTV double-positive cells was compared with the proportions remaining after human-expert gating (Fig. 4A).

![Figure 4](figs_data/figure_4_harmonized.svg)

Figure 4. A) Representative flow cytometry plots of synthetic doublet samples and gating. The first row shows all samples by CTV against CFSE, the second FSC-A/FSC-H, and the last by SSC-H/SSC-A. The columns show representative CTV-only stained, CFSE-only stained, mixed CTV-CFSE, human expert doublet-removed, and FlowMOP doublet removed samples. B) Bar graph showing the mean percentage ± SD of CTV-CFSE double positive events removed (relative to the original sample) following FlowMOP or human expert processing. C) Mean frequency ± SD of remaining CTV-CFSE double-positive events after human-expert doublet gating using either a shared groupwise gate or gates drawn independently for each sample (blue, groupwise; green, individual-sample). Only Expert 3 differed between the two approaches (unadjusted paired t-test, p = 0.009; all others p > 0.05).

FlowMOP significantly decreased the frequency of CTV-CFSE double-positive events from 7.84 ± 1.21% to 0.27 ± 0.11% (paired t-test; p = 0.001). No statistically significant difference was detected between FlowMOP and any expert for this endpoint (Fig. 4B, paired t-test; unadjusted p > 0.05). FlowMOP also determines doublet gates independently for each sample, rather than applying a shared groupwise gate. Human experts therefore performed both groupwise and individual-sample doublet gating. No statistical difference was detected between these approaches except for Expert 3, who consistently removed fewer doublets with individual-sample gating (Fig. 4C, paired t-test; p = 0.009; all other unadjusted p > 0.05).

### Biological validation

We wanted to investigate whether preprocessing changed biologically relevant cell populations. We therefore selected two major populations (B and T cells) and one less abundant population (NKT cells) in human PBMCs to examine this relationship.

No Raw-inclusive or between-method frequency contrast was significant (adjusted p ≥ 0.420; Fig. 5B). In contrast, count retention differed between methods (Fig. 5C). Relative to Expert Manual, FlowMOP retained 11.1, 11.9, and 12.7 percentage points more B, T, and NKT cells, respectively (adjusted p ≤ 0.003), and FlowCut retained 12.3, 13.4, and 13.8 percentage points more (adjusted p ≤ 0.006). PeacoQC retained 11.7–13.4 percentage points fewer cells than Expert Manual across these populations (adjusted p ≤ 0.035). Thus, FlowMOP and FlowCut preserved more cells without detectably changing major- or minor-population frequencies.

![Figure 5](figs_data/figure_5.svg)

Figure 5. Biological validation of time cleaning relative to matched ungated inputs. A) Representative time and downstream biological gates for Raw, Expert Manual, FlowMOP, PeacoQC, and FlowCut. Live CD45+ is shown as a reference endpoint; the B, T, and NKT gates do not inherit the CD45+ gate. B) Frequencies and C) counts for B, T, and NKT cells, each normalized within sample to its corresponding matched Raw value (Raw = 1). Before normalization, B-, T-, and NKT-cell frequencies were calculated relative to Live cells. Points represent eight paired inputs, connecting lines show within-sample changes, diamonds show means, and error bars show SD. Raw-comparison brackets are omitted for clarity. Displayed brackets show significant method-versus-method two-sided paired *t*-tests after Holm adjustment, with adjusted values labelled “p” for brevity (n = 8).

Figure 6 evaluates debris-only, doublet-only, and combined Time + Debris + Doublet preprocessing. Representative combined gates are shown in Figure 6A, while the corresponding frequencies and counts are reported in Figure 6B and C, respectively. FlowMOP debris cleaning removed events from every paired input, excluding 3.3–56.7% of the matched debris input (mean 18.3%), compared with 6.7–43.8% for Expert Manual (mean 24.8%). FlowMOP retained mean counts equivalent to 91.4–94.9% of matched Raw across the Live CD45+ reference, B, T, and NKT endpoints, compared with 96.3–98.8% for Expert Manual; however, no count endpoint differed significantly between the methods. No population frequency differed from Expert Manual except T-cell frequency, which was lower after FlowMOP cleaning (Raw-normalized mean 1.013 versus 1.069; adjusted p = 0.037; Fig. 6B).

Doublet cleaning did not differ directly from Expert Manual for any Live CD45+ reference, B-, T-, or NKT-cell count or frequency endpoint (Fig. 6B,C). Relative to Raw, both cleaned outputs reduced all four counts, reduced B-cell frequency, and increased T-cell frequency; neither Live CD45+ nor NKT-cell frequency differed from Raw. When all three preprocessing steps were applied together, no count endpoint differed between FlowMOP and Expert Manual. No population frequency differed except B-cell frequency, which was lower after FlowMOP cleaning (Raw-normalized mean 0.877 versus 1.022; adjusted p = 0.049; Fig. 6B). Together with the module-specific comparisons, these results indicate broad agreement between human- and FlowMOP-mediated pregating, with the debris-only T-cell frequency and combined B-cell frequency as the two direct differences across Figure 6.

![Figure 6](figs_data/figure_6.svg)

Figure 6. Biological validation of debris, doublet, and combined FlowMOP preprocessing relative to ungated inputs. A) Representative original Ungated input, Expert Manual, and FlowMOP outputs after combined Time + Debris + Doublet preprocessing, showing the Time × CD123 time-gating, FSC-A × SSC-A debris-gating, and FSC-H × FSC-W doublet-gating projections, followed by the Live CD45+ reference, shared CD19 × CD3 B/T, and NKT gates. B) Frequencies and C) counts for Live CD45+, B, T, and NKT cells. Within each population, the Debris, Doublet, and Combined subcolumns each show Raw, Expert Manual, and FlowMOP. Debris and Doublet use their matched ungated inputs, while Combined Raw is the original fully ungated input; every value is normalized within sample to that corresponding Raw value (Raw = 1). Live CD45+ is reported as a standalone reference and does not parent the B, T, or NKT endpoints, preventing CD45-based exclusion of low-scatter events from masking debris-cleaning effects. Before normalization, lineage frequencies were calculated relative to Live cells. Points represent eight paired inputs, connecting lines show within-sample changes, diamonds show means, and error bars show SD. Brackets and adjusted p values are displayed only for significant comparisons. Direct workflow comparisons used two-sided paired *t*-tests; Raw comparisons used one-sample *t*-tests against 1 on the Raw-normalized values. Tests were Holm-adjusted within each endpoint, metric, and cleaning group (n = 8).

### Tumour biological validation

We additionally examined FlowMOP in three high-debris tumour samples (Fig. 7). These samples provided a challenging setting in which the debris and non-debris distributions were not sharply demarcated. No statistically detectable differences were observed between Manual and FlowMOP for the evaluated recovery endpoints in these three samples. 

![Figure 7](figs_data/figure_7.svg)

Figure 7. FlowMOP preprocessing and downstream tumour-population measurements. A) Representative Manual and FlowMOP Time, Debris, and Doublet preprocessing plots for tumour Sample 1. B) CD3-versus-CD19 plots for tumour Samples 1, 2, and 3 in the Raw data and after Manual or FlowMOP preprocessing. T cells are CD3+CD19− (Q1), and B cells are CD3−CD19+ (Q3). C) Live CD45+ cell count, B-cell frequency, and T-cell frequency after normalization within sample to the matched Raw value (Raw = 100%). Small circles show individual samples, lines connect matched samples, and diamonds and error bars show mean ± SD. All three unadjusted, two-sided paired t-tests were performed for each endpoint; brackets and P values are displayed only for significant comparisons (p < 0.05).

The Live CD45+ cell count was 62.9 ± 17.8% of Raw after manual preprocessing and 73.3 ± 7.3% after FlowMOP preprocessing. FlowMOP differed from Raw (p = 0.024), whereas Manual did not (p = 0.069), and Manual and FlowMOP did not differ (p = 0.545). B-cell frequency was 32.5 ± 18.8% of Raw after Manual and 32.2 ± 17.0% after FlowMOP; both differed from Raw (p = 0.025 and p = 0.020, respectively), but not from each other (p = 0.855). T-cell frequency was 23.0 ± 20.4% of Raw after Manual and 22.8 ± 10.6% after FlowMOP; both differed from Raw (p = 0.023 and p = 0.006, respectively), but not from each other (p = 0.985). Thus, although both preprocessing strategies changed absolute population recovery relative to Raw, no difference between Manual and FlowMOP was detected for any of the three prespecified endpoints.

### Expert preference evaluation

Expert preference rankings were compared across nine biological datasets (Figs. S5-S7). For time gating, FlowMOP had the best mean rank among the automated methods and was preferred to FlowCut (BF = 5.39, P = 84.3%) and PeacoQC (BF = 12.10, P = 92.4%). Human gates generally ranked above FlowMOP. For debris and doublet removal, human gates were again generally preferred, but FlowMOP was competitive in several tissue-specific comparisons: it ranked first for debris removal in mouse blood, third for debris removal in human liver and mouse skin, and second for doublet removal in human liver. Full rankings are shown in Figures S5-S7, with statistical comparisons reported in the Supplementary Results.

## Discussion

### Synthetic Benchmarking

#### Synthetic Sample Time Gating

In the synthetic time-gating analysis, the sustained Segment-type artifact is perhaps the most common and consequential time artifact, as a non-negligible sample portion is often required to be removed in real samples. This type of synthetic data expects gating most similar to current manual gating, where blocks of events are removed. The objective of these samples is to simulate where there is a long blockage or sudden shift in the acquired sample. Here, FlowMOP had higher sensitivity than FlowCut at both tested bin sizes. It also had higher specificity than both competitors at 5000 events and than FlowCut at 2000 events. The Time-only mechanism benchmark suggests that the FlowMOP versus FlowCut difference is not explained solely by implementation. When local acquisition-rate structure was altered either in alignment with source-linked fluorescence changes or independently of them, FlowMOP remained unchanged, whereas FlowCut's removal behavior shifted, especially in Segment inputs (Fig. S4). This supports the interpretation that FlowCut can be affected by acquisition-density changes even when fluorescence values are unchanged, while FlowMOP is more anchored to fluorescence-population summaries across acquisition order.

Transient acquisition disturbances are not confined to the beginning or end of a run: flow-rate surges interspersed throughout acquisition and associated signal-intensity variation have been documented [4]. PeacoQC notes that temporary acquisition problems can be difficult to detect manually [5], and FlowCut describes manual identification and removal of transient acquisition problems as time-consuming and subjective [6]. We use “microblockage” operationally for a short, self-resolving instance of this broader phenomenon that produces a localized fluorescence shift, without asserting that a physical obstruction was directly observed. The Bimix and Trimix samples were designed to represent this under-addressed case. Although these synthetic samples can appear acceptable on visual inspection, the source labels show that the short altered intervals contain events from an intentionally perturbed fluorescence source and therefore should be excluded under the benchmark definition. Visual subtlety is thus a central feature of the benchmark: it demonstrates why apparent normality by eye is not sufficient ground truth.

The size of the simulated microblockage was determined by the mixing-bin size, with the 2000-event samples representing shorter and more difficult disturbances than the 5000-event samples. FlowMOP provided the strongest overall performance profile across these conditions. At 5000 events, it had higher sensitivity than FlowCut and higher specificity than both competitors across Segment, Bimix, and Trimix. At 2000 events, FlowMOP had higher sensitivity and specificity than FlowCut in Segment and retained higher specificity than PeacoQC in Bimix and Trimix without a significant sensitivity disadvantage. Thus, FlowMOP's principal advantage was its consistent combination of high sensitivity with stronger specificity across sustained and short, interspersed time artifacts.

PeacoQC's documentation identifies acquisition-bin size as a trade-off between the accuracy of within-bin density estimation, the number of bins available for evaluating signal stability, and the number of events affected when a bin is removed [5]. The Bimix and Trimix benchmarks contain short, interspersed source-defined intervals. FlowMOP's stronger performance is therefore consistent with its temporal summarisation being better matched to these brief intervals.

The computational benchmark demonstrates that FlowMOP provides speed gains with lower peak RAM usage at larger event counts. This is most evident from 300,000 events onward, where FlowMOP had both the fastest mean runtime and lowest peak memory use. At 2,000,000 events, FlowMOP was approximately 2.5-fold faster than PeacoQC and 3.5-fold faster than FlowCut, while using approximately 32% of PeacoQC’s peak RAM and 45% of FlowCut’s peak RAM.

These measurements were obtained using local non-distributed execution and therefore demonstrate the performance of the tested implementations, not a distributed-computing speedup.

Among the three evaluated decision structures, only FlowMOP exposes an inherently channel-parallel within-sample computation. After shared acquisition bins are defined, each eligible fluorescence channel independently establishes its reference, produces a fixed-shape time-bin summary, and applies smoothing and MAD filtering; cross-channel coordination occurs only in the final parameter vote. PeacoQC instead performs data-dependent peak identification and reconciliation across bin-channel combinations, while FlowCut applies adjacent-segment, contiguous-region, file-wide, and conditional rerun decisions [5,6]. Their suboperations and separate files can be parallelised, but their complete within-sample decision pipelines cannot be expressed as the same independent channel-wise reduction without changing their decision rules. Porting either implementation to another language or scheduling framework would therefore not remove these structural dependencies. This structural distinction, together with FlowMOP’s earlier data reduction, fewer full-data passes, and smaller intermediate state, is consistent with its lower runtime and memory use in our benchmarks, although the benchmark does not independently establish causation.

It is of note that there is a large variation in algorithmic performance across the dataset. One source of this variation is that the 0.5 and 1.0 relative cell concentrations oftentimes exhibited marginal differences in fluorescence intensity (Supplementary Figure S3), especially relative to the 0.5 / 3.0 cell concentrations comparison. Consequently, the 0.5/1.0 discrimination tasks can be considered especially difficult benchmarks to overcome. However, this difficulty was intentionally placed, to ensure the present benchmarking dataset could also show progressive improvement of future time-gating algorithms.

Human gating was not included for the Bimix or Trimix synthetic datasets because the short mixed bins do not provide a practical manual ground-truth target. The retained source labels instead provide an event-level reference for comparing algorithmic performance. A dataset benchmark was therefore necessary to evaluate whether automated methods can detect these subtle but labelled artifacts reproducibly.

#### Synthetic Sample Debris and Doublet Gating

In the synthetic debris and doublet gating trials, FlowMOP removed the labelled technical artifact populations effectively (Fig. 3B, 4B). In the debris task, FlowMOP enriched the low-debris component by 9.67%, which represented approximately 19% more debris removed considering the original 50:50 debris / real sample mixture. FlowMOP’s debris performance can also be interpreted in relation to the two debris populations (Fig. 3A) present: one at <10,000 FSC-A units, and the second at ~20,000 FSC-A units. Human experts were instructed that this second debris population was debris, and to gate accordingly. FlowMOP was able to independently detect this second debris population and exclude it without external information.

Similarly, in the doublet removal, the synthetic samples, owing to the rather unique preparation, yielded triplets. FlowMOP was able to handle this unexpected population and successfully removed it.

FlowMOP's sample-specific debris and doublet gating also differs from the groupwise strategy commonly used in manual analysis, where a shared gate is applied across related samples. Estimating each gate independently allows FlowMOP to adapt to sample-specific distributions without assuming that scatter characteristics are identical across a group. In the synthetic controls, expert results did not differ between shared groupwise and individual-sample debris gates (Fig. 3D), and differed for doublet gating for only one expert (Fig. 4C). This indicates that the overall comparison between FlowMOP and expert gating was not primarily determined by whether gates were shared across the group or estimated separately for each sample.

The synthetic debris benchmark measures depletion of the source-labelled high-debris component from matched mixtures of high- and low-debris samples. Because both sources contain some debris, these labels represent relative debris enrichment rather than per-event debris classification. FlowMOP targets the small, low-FSC debris phenotype observed in these controls and successfully removed the two low-FSC debris populations shown in Figure 3A. SSC-A may improve recognition of larger or internally complex debris, while pulse-width measurements may assist with aggregates. Metadata-based margin removal may also identify events at acquisition limits, but is not implemented beyond FlowMOP's current FSC-A maximum-value precleaning check. FlowMOP does not currently incorporate these additional features because broadly applicable decision rules are difficult to establish across tissues, panels, instruments, and acquisition settings. Future extensions can evaluate configurable or sample-specific multivariate approaches in tumour digests with greater necrosis, aggregation, and scatter heterogeneity.

### Biological validation

The B-, T-, and NKT-cell frequencies changed little after time gating regardless of the cleaning method, although count retention differed significantly. FlowMOP retained approximately 11–13 percentage points more cells than Expert Manual, FlowCut retained approximately 12–14 percentage points more, and PeacoQC retained approximately 12–13 percentage points fewer. From the biological data alone, it is difficult to determine whether the lower retention by Expert Manual and PeacoQC represents more sensitive artifact removal or whether the higher retention by FlowMOP and FlowCut represents more specific preservation of valid events. Considering that PeacoQC demonstrated poorer specificity than FlowMOP and FlowCut in the synthetic datasets, together with the relatively blunt nature of human manual time gating, we believe that the higher retention is more likely to reflect specific preservation of valid events than insufficient artifact removal.

The source-labelled synthetic debris benchmark provided the objective assessment of debris identification and showed that FlowMOP removed the targeted low-FSC debris populations effectively. In the PBMC analysis, FlowMOP generally removed fewer events than the expert debris gate, although removal varied between samples. No count endpoint differed directly from Expert Manual, and no population frequency differed except T-cell frequency.

By contrast, FlowMOP tended to be more aggressive than the expert doublet gate within the evaluated biological regions, producing slightly lower mean retained counts across the Live CD45+ reference, B-, T-, and NKT-cell endpoints. These differences were small, however, and no doublet count or frequency endpoint differed significantly between methods. When Time, Debris, and Doublet preprocessing were combined, the net effects varied across samples and endpoints. FlowMOP did not produce significantly different counts or frequencies from expert gating in any endpoint except B-cell frequency.

This single combined-workflow exception, together with the absence of direct count differences throughout the debris, doublet, and combined comparisons, indicates broad agreement between human- and FlowMOP-mediated pregating. Taken together with the objective synthetic benchmarks and computational performance measurements, these findings support FlowMOP as an acceptable tool for fully automated sample cleaning with downstream output quality comparable to human preprocessing.

The tumour analysis tested the other end of the debris spectrum in complex, high-debris samples lacking a sharp debris/non-debris boundary (Fig. 7). Manual and FlowMOP preprocessing both reduced Live CD45+ recovery and B- and T-cell frequencies relative to Raw, and they did not differ for any of the three prespecified endpoints. This concordance further supports FlowMOP's ability to adequately preprocess samples, even in complex tumour samples.

Across the nine-dataset expert evaluation, FlowMOP was preferred overall to FlowCut and PeacoQC for time gating. This agrees with the synthetic benchmarks, in which FlowMOP showed the strongest combined sensitivity-specificity profile, and indicates that this advantage was also reflected in expert preferences on biological datasets. Human gates were generally preferred for debris and doublet removal, although FlowMOP ranked competitively in several tissue-specific comparisons, including debris removal in mouse blood, human liver, and mouse skin and doublet removal in human liver. The variation between tissues is consistent with the greater dependence of debris and doublet gates on sample-specific scatter distributions. Taken together, these results indicate that FlowMOP-generated gates can be comparable in acceptability to human gating across many use cases.

### Other remarks

Automated Live/Dead classification of events was considered, however not implemented in the algorithm. There exist many varied protocols and methods for discriminating live/dead samples, along with great diversity in the determination of what constitutes a ‘dead’ event. Consequently, the difficulty of creating a universal live/dead discriminator is non-trivial. Finally, there may be potential significant biological insight in the ‘dead’ cells of a sample, whereby important information concerning a sample may be found in the dead events or their proportion.

The trade-off between sensitivity and specificity reflects competing biological risks rather than a purely statistical optimization. A more permissive gate may protect genuine or rare populations while allowing artifacts to remain; a more stringent gate may remove artifacts more completely while also deleting valid biological events. No balance is universally correct because the consequences of each error depend on the intended downstream analysis. Sensitivity and specificity should therefore be considered together and interpreted alongside their effects on population recovery and composition. In this context, FlowMOP's synthetic time-gating results are notable because its performance gains were not achieved simply by exchanging one error type for the other. FlowMOP had the strongest overall combined sensitivity-specificity profile across the tested conditions, including simultaneous improvements in both measures relative to FlowCut in the Segment benchmarks and higher specificity than PeacoQC in the Bimix and Trimix benchmarks without a detectable sensitivity disadvantage. Because FlowMOP's time-gating module operates on acquisition-time structure rather than population identity, rare populations are not expected to be systematically biased unless they are temporally confounded with an acquisition artifact.

The primary comparison used recommended or automatically selected settings, including fixed FlowMOP parameters, to reflect typical unsupervised use.

CTV-CFSE double-positive events provide an observable ground-truth doublet class, but same-label CTV-CTV and CFSE-CFSE doublets are not directly detectable in this validation design. FlowMOP requires appropriate acquisition voltage/gain settings; if relevant signals are poorly resolved or saturated, the lost information cannot be recovered and reliable cleaning cannot be guaranteed. FlowMOP currently expects users to identify the scatter channels used by the workflow; it has not been validated systematically across 405-nm, 488-nm, and polar 488-nm FSC/SSC configurations on instruments with multiple scatter measurements. Future versions could assess multiple scatter-channel pairs and select or combine the pair with the clearest debris/doublet separation.

## Conclusion

FlowMOP provides time-gating, conservative low-FSC debris removal, and scatter-ratio doublet removal in a single Python implementation. This facilitates integration with Python-based workflows and provides fast, memory-conscious preprocessing for large cytometry files.

Within the tested synthetic scenarios, event-level source labels enabled objective evaluation of the targeted artifact classes. FlowMOP had higher sensitivity than FlowCut for Segment anomalies at both bin sizes and higher specificity than both competitors across the 5000-event Segment, Bimix, and Trimix benchmarks. For debris and doublet removal, FlowMOP removed the labelled technical artifact populations effectively in the synthetic ground-truth datasets, including unexpected triplet events.

An expert evaluation across nine human and mouse datasets preferred FlowMOP to PeacoQC and FlowCut for time gating, while its debris and doublet gates were competitive with human gates in several tissue-specific comparisons (Figs. S5-S7), supporting the broader practical suitability of FlowMOP across diverse biological samples. In the source-labelled debris benchmark, FlowMOP removed the targeted technical populations effectively. In PBMC biological validation, FlowMOP removed debris-gated events from every paired input. No debris count endpoint differed directly from Expert Manual, and only T-cell frequency differed between the debris-cleaned outputs. Across the complete Time + Debris + Doublet pathway, no population count differed directly between FlowMOP and Expert Manual; B-cell frequency was modestly lower after FlowMOP, while the other evaluated frequencies did not differ. In the tumour samples, Manual and FlowMOP preprocessing did not substantively differ across the populations measured. Together with the synthetic and human PBMC findings, this suggests that FlowMOP performs comparably to manual preprocessing when the effects on relevant biological populations are considered. The open-source Python implementation supports reproducible preprocessing across cytometry datasets of increasing scale.

## Data and Code Availability

FlowMOP can be accessed via https://github.com/1ordinateur/FlowMOP. The code associated with the creation of this paper can be accessed at https://github.com/1ordinateur/FlowMOP_paper. The FCS Files used for this paper can be accessed at http://doi.org/10.5281/zenodo.17896445.

## Supplementary data

![Figure S1](figs_data/synthetic_time_design_schematic.svg)

Figure S1: Construction of Segment, Bimix, and Trimix synthetic time samples. No flow-rate disturbance was introduced.

![Embedded image 8](FlowMOP_submission_media/image8.png)

Figure S2: Representative CD3-versus-Time plots for Bimix samples with 2,000-event bins and Trimix samples with 2,000- and 5,000-event bins, showing the original inputs and outputs after cleaning with FlowMOP, FlowCut, and PeacoQC. Percentages below each figure represent the retained proportion of cells relative to the original representative synthetic sample.

**Table S1: Dataset Compositions**

| Dataset Name | Synthetic Time-Gating Dataset | Synthetic Debris-Gating Dataset | Synthetic Doublet-Gating Dataset | Human Liver |
| --- | --- | --- | --- | --- |
| Samples (given per benchmarker) | 6, 2 each at 0.5, 1, 3 cell concentrations, 3 datasets | 4, combined into 3 benchmarking samples | 4, combined into 3 benchmarking samples | 3 |
| Tissue Type | Peripheral blood mononuclear cells | Spleen | Spleen | Liver tissue |
| Organism | Human | Mouse | Mouse | Human |
| Collection Location | Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia | Sydney, NSW, Australia |

| Mouse Dorsal Root Ganglion | Mouse Skin | Mouse Small Intestine | Mouse Colon | Mouse Brain |
| --- | --- | --- | --- | --- |
| 3 | 3 | 3 | 3 | 4 |
| Dorsal root ganglia | Skin | Distal small intestine | Colon | Brain |
| Mouse | Mouse | Mouse | Mouse | Mouse |
| Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia |

| Mouse Central Nervous System | Mouse Spleen | Mouse Blood | Mouse Bone Marrow | Human Cultured T cells |
| --- | --- | --- | --- | --- |
| 3 | 7 | 5 | 3 | 5 |
| Spinal cord and brain | Spleen | Peripherally collected blood | Femur derived bone marrow | Peripherally collected, cultured, and restimulated T cells |
| Mouse | Mouse | Mouse | Mouse | Human |
| Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia | Canberra, ACT, Australia |

**Table S2: Jeffreys’ Scale**

| Bayes Factor | Hypothesis descriptor |
| --- | --- |
| <1 | Null hypothesis supported |
| 1-3 | Anecdotal / Weak evidence |
| 3-10 | Moderate / Substantial evidence |
| 10-30 | Very strong evidence |
| >30 | Decisive evidence |

**Table S3: Limit-of-detection precleaning in biological-validation inputs**

Events with passed_lod values below 0.5 were counted as removed. FlowMOP removes maximum-FSC-A events only when they exceed 1% of the input. This 1% cutoff is an arbitrary operational safeguard, not an empirically optimized biological threshold. Activation depends on acquisition scaling and voltage/gain settings; the observed activation rates are dataset-specific and should not be generalized across instruments.

| Dataset | Sample | Input events | Events removed | Events removed (%) | Threshold activated |
| --- | ---: | ---: | ---: | ---: | --- |
| PBMC | 1 | 143,656 | 0 | 0.00 | No |
| PBMC | 2 | 426,248 | 0 | 0.00 | No |
| PBMC | 3 | 463,512 | 0 | 0.00 | No |
| PBMC | 4 | 258,952 | 0 | 0.00 | No |
| PBMC | 5 | 140,936 | 0 | 0.00 | No |
| PBMC | 6 | 161,704 | 0 | 0.00 | No |
| PBMC | 7 | 242,344 | 0 | 0.00 | No |
| PBMC | 8 | 188,936 | 0 | 0.00 | No |
| Tumour | 1 | 287,752 | 6,863 | 2.39 | Yes |
| Tumour | 2 | 145,520 | 3,472 | 2.39 | Yes |
| Tumour | 3 | 301,872 | 9,206 | 3.05 | Yes |

**Table S4: Full-dataset FlowMOP MAD-smoothing analysis**

| Short, long smoothing factors | Sensitivity | Specificity | Balanced mean |
| --- | ---: | ---: | ---: |
| 0,0 (no-smoothing control) | 0.8021 | **0.7081** | **0.75506** |
| **0.01,0.05 (current default)** | 0.8212 | 0.6811 | **0.75114** |
| 0.02,0.05 | 0.8228 | 0.6794 | 0.75108 |
| 0.01,0.02 | 0.8112 | 0.6885 | 0.74986 |
| 0.02,0.09 | **0.8276** | 0.6709 | 0.74923 |
| 0.01,0.09 | 0.8257 | 0.6707 | 0.74823 |
| 0.05,0.09 | 0.8188 | 0.6676 | 0.74317 |
| 0.02,0.20 | 0.8248 | 0.6479 | 0.73632 |
| 0.05,0.20 | 0.8197 | 0.6520 | 0.73586 |
| 0.01,0.20 | 0.8231 | 0.6477 | 0.73540 |
| 0.10,0.20 | 0.8138 | 0.6415 | 0.72769 |
| 0.05,0.34 | 0.8194 | 0.6344 | 0.72686 |
| 0.02,0.34 | 0.8199 | 0.6286 | 0.72424 |
| 0.10,0.34 | 0.8134 | 0.6250 | 0.71921 |
| 0.10,0.50 | 0.8139 | 0.6229 | 0.71842 |
| 0.10,0.90 (former default) | 0.8139 | 0.6226 | 0.71827 |
| 0.10,1.00 | 0.8139 | 0.6226 | 0.71827 |
| 0.20,0.90 | 0.8007 | 0.6142 | 0.70747 |
| 0.40,0.90 | 0.7812 | 0.6140 | 0.69756 |

Values are equally weighted macro-averages across the six Figure 2 benchmark groups (173 primary inputs; six tied-composition inputs excluded). The balanced mean is the arithmetic mean of sensitivity and specificity. Bold indicates the highest value overall and the highest balanced mean among smoothed settings.

![Embedded image 9](FlowMOP_submission_media/image9.jpeg)

Figure S3: Fluorescence variation as a function of cell concentration.

![Figure S4](figs_data/revision_timewarp_mechanism.svg)

Figure S4: Time-only acquisition-rate perturbations reveal FlowCut sensitivity to local Time density. Points show raw-matched changes in sensitivity and specificity, with large points and intervals showing the mean and 95% confidence interval. Negative values indicate reduced performance relative to the raw control. Columns show all inputs together and the Segment, Bimix, and Trimix subsets separately. FlowMOP and PeacoQC remain unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shift after Time-only perturbation, with the clearest specificity loss in random Time-warped inputs and the strongest source-linked sensitivity loss in Segment inputs, where acquisition-rate structure aligns with source composition.

### Supplementary expert preference evaluation

#### Supplementary methods

The expert-ranking analysis was used to summarize relative preferences among gates generated by FlowMOP, comparator algorithms, and human operators; it was not treated as an absolute measure of gating adequacy [16].

Rankings were modelled using a Plackett–Luce model with latent method abilities; identifiability was enforced by fixing a reference ability to zero. Independent Normal priors were placed on non-reference abilities. Posterior inference was performed with an affine-invariant ensemble Markov Chain Monte Carlo sampler (emcee; 32 walkers, 5,000 iterations; 1,000 burn-in), and posterior medians with 95% credible intervals were reported. Directional hypotheses (superiority/inferiority) were evaluated by computing P = Pr(H1 | data) and converting to a Bayes factor BF₁₀ = p/(1 – p) under equal prior odds, interpreted using Jeffreys’ scale and reported as BF, P (Bayes factor, posterior probability of the alternative hypothesis) (Table S2).

For each Plackett–Luce fit, the ensemble sampler’s mean acceptance fraction and an integrated-autocorrelation-time-based effective sample size were monitored (both as implemented in emcee) to assess MCMC convergence. Across all analyses, mean acceptance fractions ranged from 0.520 to 0.60, and effective sample sizes ranged from 128,003 to 134,042.

Posterior predictive checks compared observed pairwise win counts with those implied by posterior draws under a Bradley–Terry formulation, using a chi-square-type discrepancy averaged over method pairs. The maximum average χ² per comparison was 0.56, indicating good agreement between the model and the observed rankings.

Computations were implemented in Python with numerically stable log-likelihoods.

#### Supplementary results

FlowMOP outputs were compared with expert-provided gates using a forced-ranking preference task across nine biological datasets. These rankings measure relative expert preference and should not be interpreted as an absolute measure of gating adequacy. The resulting time, debris, and doublet gates were ranked by an expert familiar with that sample type. FlowCut and PeacoQC were also included in the time-gating comparison. Rank 1 indicates greatest preference.

![Supplementary Figure S5](figs_data/Supp_fig_5.svg)

Figure S5. Expert preference rankings for time gates provided by four human experts, FlowMOP (black border), FlowCut, and PeacoQC across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. Abbreviations: DRG, dorsal root ganglion; CNS, central nervous system.

In the biological datasets, FlowMOP had the lowest mean rank among the algorithmic time-gating approaches (Fig. S5). In the mouse brain and mouse bone marrow tasks, it ranked third and second, respectively (Fig. S5). On a Bayesian analysis, FlowMOP was observed to be substantially preferred to FlowCut (BF = 5.39, P = 84.3%) and strongly preferred to PeacoQC (BF = 12.10, P = 92.4%). FlowMOP was ranked inferiorly to all human experts with strong to decisive evidence (Expert 2 BF = 10.55, P = 91.3%, all others BF > 100, P = 100%).

![Supplementary Figure S6](FlowMOP_submission_media/image6_revised.png)

Figure S6. Expert preference rankings for debris gates provided by four human experts and FlowMOP (black border) across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across available datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. N/A denotes an unavailable gate.

![Supplementary Figure S7](FlowMOP_submission_media/image7_revised.png)

Figure S7. Expert preference rankings for doublet gates provided by four human experts and FlowMOP (black border) across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across available datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. N/A denotes an unavailable gate.

FlowMOP had the highest mean rank when compared with the human experts for debris and doublet removal (Figs. S6, S7). On a Bayesian analysis, substantial to strong evidence was observed for FlowMOP being inferior to Expert 1 (BF = 5.87, P = 85.5%), Expert 4 (BF = 12.85, P = 92.8%), and Experts 2 and 3 (BF > 100, P = 100%) in debris removal. In doublet removal, FlowMOP was weakly inferiorly ranked to Expert 4 (BF = 3.14, P = 75.8%), substantially inferiorly ranked to Expert 1 (BF = 5.67, P = 85.0%), strongly inferiorly ranked to Expert 2 (BF = 14.38, P = 93.5%), and decisively inferiorly ranked to Expert 3 (BF > 100, P = 100%).

FlowMOP’s relative expert preference varied across datasets. For debris, it ranked first in the mouse blood task and third in the human liver and mouse skin datasets. In the doublet task, FlowMOP ranked second in the human liver task. Complete dataset-level rankings and mean ranks are shown in Figures S5-S7.

![Supplementary Figure S8](figs_data/Supp_fig_8.svg)

Figure S8. Representative debris and doublet preprocessing projections. A) Debris comparison for Ungated input, Expert Manual, and FlowMOP. B) Doublet comparison for Ungated input, Expert Manual, and FlowMOP. Each row also shows the standalone Live CD45+ reference and the corresponding B/T and NKT gates evaluated without a CD45+ parent. The corresponding module-specific statistics are reported in Figure 6B,C.

## References

[1]	Thomas Myles Ashhurst, Felix Marsh-Wakefield, Givanna Haryono Putri, Alanna Gabrielle Spiteri, Diana Shinko, Mark Norman Read, Adrian Lloyd Smith, and Nicholas Jonathan Cole King. 2022. Integration, exploration, and analysis of high-dimensional single-cell cytometry data using Spectre. Cytometry Part A 101, 3 (2022), 237–253. https://doi.org/10.1002/cyto.a.24350

[2]	Aysun Adan, Günel Alizada, Yağmur Kiraz, Yusuf Baran, and Ayten Nalbant. 2017. Flow cytometry: basic principles and applications. Critical Reviews in Biotechnology 37, 2 (February 2017), 163–176. https://doi.org/10.3109/07388551.2015.1128876

[3]	Antonio Cosma. 2020. The Nightmare of a Single Cell: Being a Doublet. Cytometry A 97, 8 (August 2020), 768–771. https://doi.org/10.1002/cyto.a.23929

[4]	Gianni Monaco, Hao Chen, Michael Poidinger, Jinmiao Chen, João Pedro de Magalhães, and Anis Larbi. 2016. flowAI: automatic and interactive anomaly discerning tools for flow cytometry data. Bioinformatics 32, 16 (August 2016), 2473–2480. https://doi.org/10.1093/bioinformatics/btw191

[5]	Annelies Emmaneel, Katrien Quintelier, Dorine Sichien, Paulina Rybakowska, Concepción Marañón, Marta E. Alarcón-Riquelme, Gert Van Isterdael, Sofie Van Gassen, and Yvan Saeys. 2022. PeacoQC: Peak-based selection of high quality cytometry data. Cytometry Part A 101, 4 (2022), 325–338. https://doi.org/10.1002/cyto.a.24501

[6]	Justin Meskas, Daniel Yokosawa, Sherrie Wang, Gabriela C. Segat, and Ryan Remy Brinkman. 2023. FlowCut: An R package for automated removal of outlier events and flagging of files based on time versus fluorescence analysis. Cytometry Part A 103, 1 (2023), 71–81. https://doi.org/10.1002/cyto.a.24670

[7]	Zicheng Hu, Alice Tang, Jaiveer Singh, Sanchita Bhattacharya, and Atul J. Butte. 2020. A robust and interpretable end-to-end deep learning model for cytometry data. Proceedings of the National Academy of Sciences 117, 35 (September 2020), 21373–21380. https://doi.org/10.1073/pnas.2003026117

[8]	Nanditha Mallesh. 2023. Automated analysis of flow cytometry using deep learning for the detection of B-cell neoplasms. Thesis. Universitäts- und Landesbibliothek Bonn. Retrieved August 7, 2023 from https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/10949

[9]	Kenneth Lo, Ryan Remy Brinkman, and Raphael Gottardo. 2008. Automated gating of flow cytometry data via robust model-based clustering. Cytometry Part A 73A, 4 (April 2008), 321–332. https://doi.org/10.1002/cyto.a.20531

[10]	Lukas Fisch, Michael Heming, Andreas Schulte-Mecklenbeck, Catharina C. Gross, Stefan Zumdick, Carlotta Barkhau, Daniel Emden, Jan Ernsting, Ramona Leenings, Kelvin Sarink, Nils R. Winter, Udo Dannlowski, Heinz Wiendl, Gerd Meyer zu Hörste, and Tim Hahn. 2024. GateNet: A novel neural network architecture for automated flow cytometry gating. Computers in Biology and Medicine 179 (September 2024), 108820. https://doi.org/10.1016/j.compbiomed.2024.108820

[11]	Jiong Chen, Matei Ionita, Yanbo Feng, Yinfeng Lu, Patryk Orzechowski, Sumita Garai, Kenneth Hassinger, Jingxuan Bao, Junhao Wen, Duy Duong-Tran, Joost Wagenaar, Michelle L. McKeague, Mark M. Painter, Divij Mathew, Ajinkya Pattekar, Nuala J. Meyer, E. John Wherry, Allison R. Greenplate, and Li Shen. 2025. Automated cytometric gating with human-level performance using bivariate segmentation. Nature Communications 16, 1 (February 2025), 1576. https://doi.org/10.1038/s41467-025-56622-2

[12]	Lisa Weijler, Florian Kowarsch, Michael Reiter, Pedro Hermosilla, Margarita Maurer-Granofszky, and Michael Dworzak. 2024. FATE: Feature-Agnostic Transformer-Based Encoder for Learning Generalized Embedding Spaces in Flow Cytometry Data. 2024. 7956–7964. Retrieved May 3, 2024 from https://openaccess.thecvf.com/content/WACV2024/html/Weijler_FATE_Feature-Agnostic_Transformer-Based_Encoder_for_Learning_Generalized_Embedding_Spaces_in_WACV_2024_paper.html

[13]	Chris P. Verschoor, Alina Lelic, Jonathan L. Bramson, and Dawn M. E. Bowdish. 2015. An introduction to automated flow cytometry gating tools and their implementation. Frontiers in Immunology 6 (July 2015), 380. https://doi.org/10.3389/fimmu.2015.00380

[14]	Christina Bligaard Pedersen, Søren Helweg Dam, Mike Bogetofte Barnkob, Michael D. Leipold, Noelia Purroy, Laura Z. Rassenti, Thomas J. Kipps, Jennifer Nguyen, James Arthur Lederer, Satyen Harish Gohil, Catherine J. Wu, and Lars Rønn Olsen. 2022. cyCombine allows for robust integration of single-cell cytometry datasets within and across technologies. Nat Commun 13, 1 (March 2022), 1698. https://doi.org/10.1038/s41467-022-29383-5

[15]	Kipper Fletez-Brant, Josef Špidlen, Ryan R. Brinkman, Mario Roederer, and Pratip K. Chattopadhyay. 2016. flowClean: Automated identification and removal of fluorescence anomalies in flow cytometry data. Cytometry Part A 89, 5 (2016), 461–471. https://doi.org/10.1002/cyto.a.22837

[16]	Noah Castelo, Maarten W. Bos, and Donald R. Lehmann. 2019. Task-Dependent Algorithm Aversion. Journal of Marketing Research 56, 5 (October 2019), 809–825. https://doi.org/10.1177/0022243719851788
