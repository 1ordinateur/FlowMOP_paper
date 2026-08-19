# FlowMOP: An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool

Running Title: Automated Python-based Sample Cleanup

**Authors:**

Tony Xu [1], Nadia A. Roberts [1], Felix Marsh-Wakefield [2,3], Rebecca A. Jaeger [1], Sarah Croft [1], Abhimanu Pandey [1], Dalton Leibold [4], Angela L. Ferguson [2,3], Lily Rodgers [2,3], Umaimainthan Palendira [3], Geoffrey W. McCaughan [2,5,6], Ben Quah [7], Robin Vlieger* [8], Anne Brüstle* [1]

**Affiliations:**

1: Department of Immunology and Infectious Disease, John Curtin School of Medical Research, the Australian National University, Canberra, Australian Capital Territory, Australia 
2: Liver Injury & Cancer Program, Cancer Innovations Centre, Centenary Institute, The University of Sydney, Sydney, New South Wales, Australia
3: Human Immunology Laboratory, School of Medical Sciences, Faculty of Medicine and Health, The University of Sydney, Sydney NSW, Australia	
4: Division of Ecology and Evolution, Research School of Biology, the Australian National University, Canberra, Australian Capital Territory, Australia 
5: A.W. Morrow Gastroenterology and Liver Centre, Royal Prince Alfred Hospital, Sydney NSW, Australia
6: Sydney Medical School , Faculty of Medicine and Health, The University of Sydney, Sydney NSW, Australia
7: Division of Genome Sciences and Cancer, John Curtin School of Medical Research, the Australian National University, Canberra, Australian Capital Territory, Australia
8: School of Medicine and Psychology, Australian National University, Canberra, Australian Capital Territory, Australia

**Contact information for all authors:**

Tony Xu <Tony.Xu@anu.edu.au>; Nadia A. Roberts <Nadia.Roberts@anu.edu.au>; Felix Marsh-Wakefield <felix.marsh-wakefield@sydney.edu.au>; Rebecca A. Jaeger <Rebecca.Jaeger@anu.edu.au>; Sarah Croft <Sarah.Croft@anu.edu.au>; Abhimanu Pandey <Abhimanu.Pandey@anu.edu.au>; Dalton Leibold <Dalton.Leibold@anu.edu.au>; Angela L. Ferguson <a.ferguson@centenary.org.au>; Lily Rodgers <lrod8310@uni.sydney.edu.au>; Umaimainthan Palendira <umaimainthan.palendira@sydney.edu.au>; Geoffrey W. McCaughan <g.mccaughan@centenary.org.au>; Ben Quah <ben.quah@anu.edu.au>;  Robin Vlieger <Robin.Vlieger@anu.edu.au>; Anne Brüstle <Anne.Bruestle@anu.edu.au>

## Acknowledgements

We thank Givanna Putri for her very helpful feedback and suggestions throughout the manuscript’s preparation. This work was also supported by computational resources provided by the Australian Government through the National Computational Infrastructure (NCI) under the ANU Merit Allocation Scheme.

**Funding information:**

TX is supported by the Australian Government Research Training Program PhD Scholarship. There are no other relevant sources of funding.

## Abstract

Flow cytometry now generates high-parameter datasets whose scale and variability challenge manual preprocessing, leading to subjectivity and poor reproducibility. <span style="color:#0066cc">Here, we introduce FlowMOP, a Python-native framework that automates three major preprocessing steps—time-gating, debris removal, and doublet exclusion.</span> <span style="color:#0066cc">FlowMOP was developed to combine these preprocessing steps in a single workflow, whereas FlowCut and PeacoQC primarily address time-dependent quality control and broader automatic gating frameworks are aimed at population identification rather than standardized preprocessing cleanup.</span>

<span style="color:#0066cc">Methodologically, FlowMOP identifies temporal artifacts—acquisition-dependent deviations in event quality or fluorescence signal—via parameter-wise peak checks, bin-level fluorescence summaries across acquisition time, and robust outlier rejection.</span> Debris is excluded by adaptive FSC-A thresholding derived from cross-parameter peak structure. Finally, doublets are removed using dynamic inflection detection on FSC-A/FSC-H and SSC-A/SSC-H ratio histograms. <span style="color:#0066cc">The implementation uses memory-conscious array operations; computational scaling was evaluated to 2 million events in a 36-channel file.</span>

<span style="color:#0066cc">Validation used synthetic datasets with event-level ground truth for acquisition-time artifacts, debris enrichment, and doublet enrichment, together with an exploratory expert comparison on biological datasets.</span> <span style="color:#0066cc">In the synthetic time-gating benchmark, FlowMOP had higher sensitivity than FlowCut for Segment artifacts at both tested bin sizes and in the 5000-event Bimix and Trimix settings, and higher specificity than both competitors across all three 5000-event scenarios.</span> <span style="color:#0066cc">FlowMOP also removed labelled debris and doublet populations effectively in technical-control datasets.</span> <span style="color:#0066cc">The expert comparison, which used an earlier FlowMOP configuration, generally favoured manual gates and is reported in the Supplementary Information.</span> <span style="color:#0066cc">**[PLACEHOLDER: Principal biological-validation finding from Figure 5.]**</span> <span style="color:#0066cc">In three tumour samples, no difference was detected between FlowMOP and manual preprocessing for the T:B-cell ratio, Live CD45+ cell count, B-cell frequency, or T-cell frequency.</span> <span style="color:#0066cc">Within the tested settings, FlowMOP provides a reproducible workflow for standardizing time, debris, and doublet preprocessing.</span> FlowMOP can be accessed at https://github.com/1ordinateur/FlowMOP.

## Introduction

<span style="color:#0066cc">Modern flow-cytometry studies can contain increasing numbers of files, events, and measured parameters [2].</span> <span style="color:#0066cc">The large scale of these data makes repeated manual preprocessing time-consuming and often impractical, potentially amplifying variability between operators.</span> <span style="color:#0066cc">Reproducible automated preprocessing is therefore valuable for large studies.</span>

Manual preprocessing of flow cytometry data, whilst still not standardised, typically involves three gating components. I) Time gating: operators perform time-gating to remove events potentially acquired erroneously due to transient or persistent artifacts in the instrument or sample, including air bubbles, blockages, or laser malfunctions. II) Debris gating: operators remove debris, which is generated by both the sample preparation process and inherent in the sample itself. Debris events are generally identified by reference to measured events with low size (determined by FSC-A (Forward SCatter – Area)) and internal complexity (determined by SSC-A (Side SCatter – Area)) [1]. III) Doublet gating: Events classified as "doublets"—two or more cells erroneously detected as one—are removed as their fluorescence measurements lack reliability. This is often done with reference to the ratio of the signal duration, and signal intensity. Doublets feature a signal peak strength comparable to a single cell, but for twice the duration. Hence, analysts may filter these events out by comparing an event’s total FSC Area relative to its FSC height (peak signal intensity) or by comparing the FSC signal height against the FSC signal width [4]. These manual preprocessing steps are time-consuming, inherently subjective, and susceptible to inconsistency across operators.

<span style="color:#0066cc">The most conspicuous time artifacts often occur at the beginning or end of acquisition, but flow-rate surges interspersed throughout acquisition and associated signal-intensity variation have also been documented [10].</span> <span style="color:#0066cc">Here, we use “microblockage” operationally to denote a short, self-resolving mid-acquisition disturbance that produces a localized fluorescence shift; the term does not imply that a physical obstruction was directly observed.</span> <span style="color:#0066cc">Temporary acquisition problems can be difficult to detect manually [5], and manual identification and removal of transient problems can be time-consuming and subjective [9].</span> <span style="color:#0066cc">Because the affected intervals may be short and interspersed with otherwise plausible events, they can be impractical to exclude reproducibly using multiple manual gates.</span>

<span style="color:#0066cc">Several tools automate cytometry population identification, including the model-based flowClust method [13], the neural-network method GateNet [14], and the bivariate-segmentation framework UNITO [15]; automated gating approaches are reviewed elsewhere [16].</span> <span style="color:#0066cc">FlowMOP is a Python-based, training-free preprocessing workflow that combines automated time-gating, debris removal, and doublet exclusion in a single headless tool.</span> <span style="color:#0066cc">FlowCut and PeacoQC provide the most direct comparison for its time-gating component.</span>

The Python implementation specifically facilitates integration with contemporary machine-learning workflows, which predominantly utilize Python-based frameworks.

<span style="color:#0066cc">To date, cytometry preprocessing algorithms have been evaluated either by comparison to human-defined gold standards (as in FlowCut and PeacoQC) or through mathematical metrics (consider cyCombine’s approach to batch correction [11]).</span> <span style="color:#0066cc">The synthetic datasets provide event-level labels for estimating sensitivity and specificity in preprocessing tasks where real ground truth is otherwise difficult to define.</span>

<span style="color:#0066cc">Here, we introduce FlowMOP, an automated preprocessing tool for time-gating, debris removal, and doublet exclusion.</span> <span style="color:#0066cc">We compare time-gating performance with PeacoQC and FlowCut, evaluate debris and doublet removal using synthetic ground-truth datasets and expert comparison, and benchmark computational scalability.</span>

## Methods

For further details concerning datasets, see Supplementary Table 1A.

### Preparation protocols for synthetic samples

#### Synthetic Time Sample Preparation and Generation

#### Preparation of Human PBMC Samples for Synthetic Time Benchmarking Samples

PBMC samples were collected from healthy donors under ethics protocols ACT Health 2019.ETH.00081 and ANU HREC 2020/047. Blood was collected into ACD-A tubes and PBMCs isolated using Lymphoprep density gradient (Stemcell Technologies) and SepMate tubes (Stemcell Technologies) per manufacturer’s instructions prior to cryopreservation. Thawed PBMCs were stained with ViaDye Red (Cytek Biosciences). The staining cocktail contained antibodies to several highly abundant antigens at less-than-saturating concentrations, and either 0.5, 1 or 3 x 10^6 cells were stained, yielding samples with different levels of fluorophore signal intensity for these common antigens. Other antibodies within the panel were present at saturating concentrations to yield similar fluorescence intensity across different cell seeding densities. Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer. <span style="color:#0066cc">The benchmarking uses real, existing datasets acquired with biology-specific antibody panels. Consequently, the antigen–fluorochrome combinations differ among datasets and are not variables under evaluation; they are not material to the preprocessing questions addressed here.</span>

#### Generation of Synthetic Time Samples

Samples were synthetically combined across differing cell concentrations in three manners. <span style="color:#0066cc">Their construction is shown in Figure S1.</span> One in a simple ‘Segmented’ fashion, where events from one or more samples were simply appended onto events from an existing sample. The second, ‘Bimix’ manner, where events from two differently stained samples were randomly synthetically combined in random proportions (e.g. 40:60, 75:25 etc), with one run containing a mixing bin size of 5000 events and the other of 2000 events. The final, ‘Trimix’ contained randomly combined events from three differently stained samples in mixing bin sizes of 5000, and 2000 events. <span style="color:#0066cc">Here, we use “microblockage” operationally to denote a short, self-resolving mid-acquisition disturbance that produces a localized fluorescence shift; the term does not imply that a physical obstruction was directly observed.</span> <span style="color:#0066cc">Segmented samples model sustained changes, whereas Bimix and Trimix model the observable fluorescence consequence in this operational definition by introducing short source-defined fluorescence shifts during acquisition.</span> <span style="color:#0066cc">They do not recreate or establish the physical mechanism itself.</span> <span style="color:#0066cc">The Bimix and Trimix files can appear acceptable on visual inspection because the altered intervals are short and interspersed with otherwise plausible events; however, the retained source labels identify the intentionally perturbed events that should be excluded under the benchmark definition.</span> <span style="color:#0066cc">This provides event-level ground truth for a class of artifact that is difficult to recognize and impractical to gate manually [5,9].</span> <span style="color:#0066cc">Flow-rate disturbances without corresponding fluorescence changes should not prompt event exclusion; these samples therefore model fluorescence changes across acquisition order without introducing flow-rate disturbances.</span> <span style="color:#0066cc">Flow-rate effects were tested separately by altering Time either in alignment with source-linked fluorescence changes or independently of them.</span>

#### <span style="color:#0066cc">Time-only acquisition-rate mechanism benchmark</span>

<span style="color:#0066cc">To distinguish acquisition-rate sensitivity from source-linked fluorescence and composition structure, we performed a matched mechanism benchmark using source-labelled smallcut synthetic-combo FCS files.</span> <span style="color:#0066cc">Thirty high-count inputs were selected: ten Bimix, ten Trimix, and ten Segment files.</span> <span style="color:#0066cc">For each file, 500,000 acquisition-order-preserving events were used without changing fluorescence, scatter, source labels, or event order.</span> <span style="color:#0066cc">Bimix and Trimix files used the first 500,000 events; Segment files used a contiguous 500,000-event window centered on the source transition so that both segment sources were represented.</span>

<span style="color:#0066cc">Each input generated three matched variants: raw, source-time-warped, and random-time-warped.</span> <span style="color:#0066cc">In the source-time-warped variant, local Time increments were multiplied according to source identity using acquisition-interval multipliers spanning 1.0x to 20.0x.</span> <span style="color:#0066cc">In the random-time-warped variant, the same multiplier range was assigned to contiguous 25,000-event chunks independently of source identity.</span> <span style="color:#0066cc">After warping, the total Time range was rescaled to match the raw input so that the benchmark tested local acquisition-rate structure rather than total acquisition duration.</span> <span style="color:#0066cc">Performance was evaluated relative to the matched raw file using sensitivity and specificity.</span>

#### <span style="color:#0066cc">MAD-smoothing ablation and default selection</span>

<span style="color:#0066cc">To test spline smoothing directly, FlowMOP was rerun across all 173 primary synthetic time-gating inputs with all non-smoothing settings fixed.</span> <span style="color:#0066cc">Nineteen short/long smoothing-factor pairs, including a no-smoothing control, were compared by equally weighting the six benchmark groups.</span> <span style="color:#0066cc">The current default (`0.01,0.05`) was selected as the marginally highest-scoring smoothed setting on this balanced comparison (Table S4).</span> <span style="color:#0066cc">Figure 2 was regenerated using this setting. All completed quantitative FlowMOP results retained in the main figures use the current selected configuration for the relevant module, and the planned biological-validation figures will likewise use the current configuration. The earlier expert preference evaluation used a previous FlowMOP configuration; because updating it would require repeating the blinded expert assessment, it is presented as an exploratory supplementary analysis (Figs. S5-S7).</span>

#### Mice

For generation of synthetic samples for FlowMOP validation (debris and doublet removal), splenocytes from 13-16-week old C57Bl/6N or C57Bl/6J mice were used. All animal experimentation was performed under ethics protocol 2024/379 at the Australian Phenomics Facility at the Australian National University, Canberra.

#### Synthetic Debris Sample Preparation and Generation

Mouse spleens were mechanically dissociated prior to lysis of red blood cells (RBC) with RBC lysis buffer (MilliQ H2O containing 150 mM NH4Cl, 10 mM KHCO3 and 1 mM EDTA). Splenocytes were then incubated with Fc block (BD), then stained with an antibody panel to delineate several major immune cell subsets (CD19 APC, CD3 PE, CD8 PE-Cy7, CD11b BB515, CD4 BV605, Fixable Viability Dye efluor780). To generate ‘high debris’ samples, cells were resuspended in MilliQ water and incubated for 2 minutes, prior to addition of 10X PBS to restore osmolarity. ‘Low debris’ samples were kept in isotonic solution throughout. Samples were acquired using a BD LSRII cytometer.

<span style="color:#0066cc">For assessment of FlowMOP debris-gating performance, high-debris and low-debris samples were combined by sampling approximately equal event numbers from each source and concatenating them into matched synthetic mixtures while retaining source labels for ground-truth quantification.</span>

#### Synthetic Doublet Sample Preparation and Generation

<span style="color:#0066cc">To generate samples with high proportions of doublets, C57BL/6 mouse spleens were injected with 1 mL digestion buffer comprising RPMI supplemented with 50 µg/mL collagenase P (Roche) and 10 µg/mL DNase I (Roche). Spleens were incubated for 20 minutes at room temperature (RT) in a further 1 mL of digestion buffer, mechanically dissociated, and incubated for a further 20 minutes at RT. Samples were passed through a 70-µm cell strainer with 10 mL FACS wash (PBS containing 2% heat-inactivated fetal bovine serum [FBS]), centrifuged (5 minutes, 500 × *g*, RT), and the supernatant discarded. Cell pellets were resuspended in 3 mL RBC lysis buffer and incubated for 3 minutes at RT, after which cells were washed twice with FACS wash and recovered by centrifugation.</span>

<span style="color:#0066cc">For dye labelling, 5 × 10^6 cells were transferred to each fresh 15-mL tube and stained with either CellTrace Violet (CTV; Invitrogen) or carboxyfluorescein succinimidyl ester (CFSE; eBioscience). Cells were centrifuged (5 minutes, 500 × *g*, 4°C), the supernatant was removed, and pellets were resuspended in 1 mL complete IMDM (cIMDM; Gibco IMDM supplemented with 10% heat-inactivated FBS, 100 U/mL penicillin, 100 µg/mL streptomycin, 2 mM L-glutamine, and 55 µM 2-mercaptoethanol). Two microlitres of 5 mM CTV or 10 mM CFSE were added to the side of the corresponding tube; samples were rapidly inverted and mixed, then incubated for 10 minutes at 37°C. Labelling was quenched by adding 5 mL ice-cold cIMDM and incubating for a further 5 minutes at RT. Cells were centrifuged and resuspended in cIMDM, after which 2 × 10^6 CTV-labelled cells and 2 × 10^6 CFSE-labelled cells were combined in a single sample and incubated for 30 minutes at 37°C and 5% CO2. Samples were centrifuged, the supernatant was removed, and cells were stained for 20 minutes at RT with Fixable Viability Dye eFluor 780 (Invitrogen; 1:1,000 in PBS). Cells were washed with PBS, centrifuged, resuspended in FACS wash, and acquired on a BD LSRII flow cytometer.</span>

### Non-synthetic samples

<span style="color:#0066cc">Human liver samples used in the non-synthetic validation datasets were collected under ethics approval from the Sydney Local Health District Ethics Review Committee (X19-0488 and 2019/ETH13790).</span>

### <span style="color:#0066cc">Biological-validation analysis</span>

<span style="color:#0066cc">**[PLACEHOLDER: Figure 5 biological-validation samples, preprocessing comparisons, prespecified biological endpoints, and statistical analysis.]**</span>

### <span style="color:#0066cc">Tumour biological-validation analysis</span>

<span style="color:#0066cc">Three human tumour-liver FCS files (LB202, LB236, and LB262) were analysed in their Raw state and after either manual or FlowMOP preprocessing.</span> <span style="color:#0066cc">The manual population was reconstructed from sequential time, cells/debris, and single-cell gates in the FlowJo workspace.</span> <span style="color:#0066cc">FlowMOP calculated time, debris, and doublet exclusions independently; the union of excluded events was removed, equivalently retaining the intersection of the three passed-event masks.</span> <span style="color:#0066cc">The limit-of-detection mask was not included.</span>

<span style="color:#0066cc">T and B cells were identified from CD3 and CD19 expression.</span> <span style="color:#0066cc">T cells were defined as CD3+CD19− (Q1) and B cells as CD3−CD19+ (Q3).</span> <span style="color:#0066cc">Four prespecified endpoints were calculated: the T:B-cell count ratio, the number of Live CD45+ cells, and T- and B-cell frequencies as percentages of the original total event count.</span> <span style="color:#0066cc">Each endpoint was normalized within sample to its matched Raw value (Raw = 100%).</span> <span style="color:#0066cc">Raw, Manual, and FlowMOP values were compared using all three unadjusted, two-sided paired t-tests.</span>

### <span style="color:#0066cc">Computational scalability benchmark</span>

<span style="color:#0066cc">Computational scalability was evaluated using clone-based real-FCS scaling.</span> <span style="color:#0066cc">A representative FCS file was subsampled/replicated to matched event counts of 10,000, 100,000, 300,000, 1,000,000, and 2,000,000 events while preserving the original 36-channel structure.</span> <span style="color:#0066cc">FlowMOP, PeacoQC, and FlowCut were run on the same generated inputs for each size.</span>

<span style="color:#0066cc">For fair timing of the shared time-gating task, FlowMOP was run using local non-distributed execution, with debris and doublet removal disabled and annotated output FCS writing disabled.</span> <span style="color:#0066cc">PeacoQC and FlowCut were run with optional plotting, reporting, and output generation disabled where supported.</span> <span style="color:#0066cc">Each condition was run once as a warm-up and then three measured times.</span> <span style="color:#0066cc">Runtime and peak resident memory were recorded using `/usr/bin/time -v`; means and standard deviations were calculated across the three measured repeats.</span>

### Coding Assistance

Parts of the code were generated with the aid of ChatGPT Codex and Claude Code. All code generated by LLMs were manually verified before implementation.

## RESULTS

![Embedded image 1](FlowMOP_submission_media/image1.png)

Figure 1. <span style="color:#0066cc">Conceptual schematics depicting FlowMOP’s time-gating (A), debris-gating (B), and doublet-gating (C) methods; plotted fluorescence intensity and signal-strength axes are schematic and use arbitrary units.</span> <span style="color:#0066cc">A) FlowMOP selects valid parameters using positive-peak detection, generates time-binned fluorescence summaries, applies two smoothing resolutions, and performs robust outlier rejection across acquisition order.</span> <span style="color:#0066cc">B) FlowMOP applies the same valid-parameter check, derives a candidate FSC-A threshold from each eligible parameter’s positive events, and uses the median candidate as the final FSC-A gate.</span> <span style="color:#0066cc">C) FlowMOP identifies doublets from inflection points in FSC-A/FSC-H and SSC-A/SSC-H ratio histograms, with a fixed-ratio fallback.</span>

### Algorithmic design

An overview of FlowMOP’s architecture is contained in Fig. 1, detailing approaches for its preprocessing, time-gating, debris removal, and doublet removal methods. This cleaned data can be applied to downstream analysis.

<span style="color:#0066cc">FlowMOP accepts `.csv`, `.fcs`, and Parquet files.</span> <span style="color:#0066cc">It reduces event-level measurements into time-bin and channel summaries, applies smoothing and robust outlier detection, and combines the resulting flags through parameter voting.</span>

#### Precleaning

<span style="color:#0066cc">FlowMOP first checks the input file for events at the limit of detection, defined here as events at the maximum FSC-A value for that sample.</span> <span style="color:#0066cc">If the number of events at this maximum exceeds a threshold (default 1%), FlowMOP removes these maximum-valued events.</span> <span style="color:#0066cc">Otherwise, it retains all values.</span> <span style="color:#0066cc">The 1% cutoff is a pragmatic safeguard intended to identify non-trivial accumulation at the acquisition maximum without responding to isolated maximum-valued events.</span>

#### Time Gating

To time gate, FlowMOP builds upon the assumptions posited in PeacoQC and FlowCut regarding fluorescence fluctuations. That is, independent of flow rate variations, sections of acquired sample with aberrant positive fluorescence averages are the target portions to be removed. To achieve this, FlowMOP checks each parameter, excluding parameters with a unimodal distribution. ‘Unimodal distribution’ is presently defined as parameters with only one identifiable peak. Subsequently, for each fluorescence parameter that satisfies this criterion, FlowMOP excludes the first peak (selecting all subsequent peaks) and measures the average fluorescence value for each time bin. FlowMOP then can operate either in the ‘Positives’ mode, or ‘Geomean’ mode. In ‘Positives’ mode, all events before the first inflection point are discarded. All results shown presently operate in ‘Positives’ mode. In the ‘Geomean’ mode, all events are considered. Subsequently, on a per-parameter basis, the sample is transformed into bins grouped by time (the default being bin having minimum of 150 events, up to a maximum of 500 bins).

The median fluorescence of each bin’s cells is returned. <span style="color:#0066cc">Two spline smoothing values, one small and one larger (current default `0.01,0.05`), are applied to the returned time-bin series before median absolute deviation (MAD) filtering.</span> <span style="color:#0066cc">The smoothing factor scales the spline fit used for the binned fluorescence summary.</span> <span style="color:#0066cc">Bins falling outside the MAD threshold in either smoothing pass are flagged for removal.</span> Time-bins across all parameters are then combined, with time bins rejected if they have been flagged in any parameter. <span style="color:#0066cc">For panels with more than 10 parameters, FlowMOP requires two or more parameters to flag a bin before rejection.</span> <span style="color:#0066cc">This empirical safeguard reduces false-positive removal caused by isolated noisy channels in high-dimensional panels, although an aberration confined to one channel may consequently be retained (Figure 1A).</span>

<span style="color:#0066cc">FlowMOP summarizes each time bin using the median fluorescence of the selected positive population.</span> <span style="color:#0066cc">If Geomean mode is selected, FlowMOP instead uses the geometric mean of all events in the bin.</span> <span style="color:#0066cc">The two smoothing resolutions target both shorter and more sustained deviations, while parameter voting limits removal driven by isolated noisy channels in higher-dimensional panels.</span>

#### Debris Gating

<span style="color:#0066cc">FlowMOP’s debris module targets small, low-FSC debris.</span> <span style="color:#0066cc">The final gate is applied on FSC-A, but its threshold is informed by FSC-A distributions across eligible fluorescence-positive populations rather than the overall FSC-A histogram alone.</span> <span style="color:#0066cc">SSC-A may improve recognition of larger or internally complex debris, while pulse-width measurements may assist with aggregates.</span> <span style="color:#0066cc">Accordingly, FlowMOP does not currently incorporate SSC-A or pulse-width measurements into its debris decision because broadly applicable thresholds for these features are difficult to establish across tissues, panels, instruments, and acquisition settings.</span> FlowMOP’s debris exclusion conducts a similar unimodality check on each fluorescence parameter, and the first peak is then excluded as the Time-gating feature. Thereafter, FlowMOP detects the global FSC-A peak as a reference point. For every parameter’s positive events, FlowMOP checks first for an FSC-A peak similar to the reference peak (default 30% of the reference peak’s value). If there is such a peak, it checks if the second FSC-A peak is the global maximum FSC-A peak. If that parameter’s positive cell’s second FSC-A is the global maxima, FlowMOP returns the FSC-A threshold as the minima between those two FSC-A peaks. If the second FSC-A is not the maximum, it returns the global interpeak minimum between the reference peak and maximal peak. Conversely, if there is no reference peak present in that parameter’s positive population, it selects the left-boundary of that parameter’s first peak. The median FSC-A threshold across all parameters is taken as the final FSC-A gate to be applied to the sample (Figure 1B).

#### Doublet Gating

To doublet gate, FlowMOP dynamically excludes sample doublets. To do this, FlowMOP creates a histogram of the FSC-A/FSC-H ratio. If there are multiple peaks all with a ratio of 1 or more, then it chooses the inflection point between those peaks, and excludes all events larger than that value. If there are insufficient peaks, it simply returns all events that have an FSC-A/FSC-H ratio smaller than a threshold (default 5). The process is repeated for the 
SSC-A/SSC-H variable. Consequently, FlowMOP is able to distill the implicit ratiometric information that current density based methodologies may overlook. <span style="color:#0066cc">FlowMOP's doublet module assumes that FSC-A/FSC-H and SSC-A/SSC-H ratios remain informative and that acquisition voltages and scatter parameters have been set appropriately.</span> <span style="color:#0066cc">If relevant scatter channels are saturated, edge-collapsed, or incorrectly configured at acquisition, the lost pulse-shape information cannot be recovered by FlowMOP or by other downstream preprocessing algorithms; such samples require acquisition review, manual intervention, or alternative pulse-shape features where available.</span>

### Algorithmic Validation

#### Synthetic Sample Benchmarking

The ability of FlowMOP to successfully detect time, debris, and doublet-perturbed data was first tested against the respective task’s synthetic datasets, namely the synthetically combined staining time samples, the high-debris + low debris samples, and the CTV / CFSE doublet samples.

<span style="color:#0066cc">For time-gating samples, sensitivity and specificity were reported for each benchmarked method using source labels as the reference.</span> <span style="color:#0066cc">Target source(s) were defined as the source(s) with the largest filename-encoded mixture proportion; tied largest proportions were treated as co-targets.</span> <span style="color:#0066cc">Sensitivity was defined as retained target-source events divided by all retained events.</span> <span style="color:#0066cc">Specificity was defined as removed non-target-source events divided by all removed events.</span>

![Figure 2](figs_data/figure_2.svg)

Figure 2. A) Representative flow cytometry plots showing CD3 fluorescence against time. The original synthetically generated sample is shown in column 1, with the resulting output following FlowMOP, PeacoQC, and FlowCut processing shown in the subsequent columns. The first row depicts a representative ‘segmentation’-based synthetic file. The second row shows a representative two-sample mixture with a 5000-event bin size. Frequency percentages shown are the percentage of cells left post-cleaning relative to the original synthetic sample (rounded to the nearest percentage point). B) <span style="color:#0066cc">Violin plots showing sensitivity and specificity, grouped by sample type (Segment, Bimix, and Trimix) and cleaning method (FlowMOP, PeacoQC, and FlowCut). Internal dashed lines indicate quartiles. The first and second rows represent mixing-bin sizes of 5000 events (n = 90: Segment, 33; Bimix, 33; Trimix, 24) and 2000 events (n = 83: Segment, 32; Bimix, 28; Trimix, 23), respectively. Brackets show significant Bonferroni-adjusted paired t-tests.</span>

#### Synthetic Time Gating Benchmark

<span style="color:#0066cc">Several computational approaches address time-dependent quality control, including flowAI [10], PeacoQC [5], flowClean [6], and FlowCut [9].</span> <span style="color:#0066cc">FlowMOP combines a time-gating component with debris and doublet modules in one Python workflow.</span> <span style="color:#0066cc">Here, its time-gating performance is evaluated against PeacoQC and FlowCut.</span>

<span style="color:#0066cc">The Segment, Bimix, and Trimix samples were designed as complementary time-artifact stress tests with event-level source information for objective scoring (Fig. 2A). Sensitivity was defined as the proportion of retained events derived from the target source or sources, and specificity as the proportion of removed events derived from the non-target source or sources. Direct comparison with manual gating was not attempted because the short, interspersed intervals in the Bimix and Trimix samples are impractical to remove reproducibly by eye.</span>

<span style="color:#0066cc">At the 5000-event bin size, FlowMOP had higher sensitivity than FlowCut in Segment (p < 0.001), Bimix (p < 0.001), and Trimix (p = 0.03) (Fig. 2B). PeacoQC also had higher sensitivity than FlowCut in Bimix (p = 0.03). FlowMOP had higher specificity than both PeacoQC and FlowCut in Segment (p = 0.03 and p = 0.009), Bimix (p < 0.001 and p = 0.001), and Trimix (p < 0.001 and p = 0.001), respectively.</span>

<span style="color:#0066cc">At the 2000-event bin size, FlowMOP and PeacoQC had higher sensitivity than FlowCut in Segment (p < 0.001 and p = 0.007, respectively), while FlowMOP had higher specificity than FlowCut (p = 0.03). In Bimix, PeacoQC had lower sensitivity than FlowCut (p = 0.02) and lower specificity than both FlowMOP (p < 0.001) and FlowCut (p = 0.007). In Trimix, sensitivity did not differ significantly among methods, while PeacoQC had lower specificity than FlowMOP (p = 0.009) and FlowCut (p = 0.003). All comparisons were paired t-tests with Bonferroni correction, using each algorithm's fixed recommended or default settings.</span>

<span style="color:#0066cc">To test the effect of flow-rate disturbances aligned with or independent of source-linked fluorescence changes, we altered only the Time channel while leaving fluorescence, scatter, source labels, and event order unchanged (Fig. S4).</span> <span style="color:#0066cc">FlowMOP and PeacoQC were unchanged under both source-linked and random Time warping.</span> <span style="color:#0066cc">In contrast, FlowCut's sensitivity and specificity shifted after Time-only perturbation.</span> <span style="color:#0066cc">Across all inputs, random Time warping reduced FlowCut's specificity by 11.45 percentage points relative to matched raw inputs.</span> <span style="color:#0066cc">In Segment inputs, source-linked Time warping reduced FlowCut's sensitivity by 7.84 percentage points and specificity by 15.25 percentage points.</span> <span style="color:#0066cc">These results support the interpretation that FlowCut responds to local acquisition-density structure even when fluorescence values are unchanged, whereas FlowMOP and PeacoQC are unaffected by rate-only variation in these inputs.</span>

<span style="color:#0066cc">Dual-resolution smoothing changed the balance between sensitivity and specificity (Table S4). The no-smoothing control had the highest equal-weight balanced mean (0.7551), but its sensitivity was 1.92 percentage points lower than that of `0.01,0.05`. Among smoothed settings, `0.01,0.05` had the highest balanced mean (0.7511), with specificity 2.70 percentage points lower than the no-smoothing control. We therefore selected `0.01,0.05` as the default and regenerated Figure 2 using this setting.</span>

Representative plots for the 2000 bin Bimix method, and the 2000, and 5000 bin Trimix methods can be found in Supp. 1.

#### <span style="color:#0066cc">Computational scalability</span>

<span style="color:#0066cc">Clone-based runtime and memory benchmarking showed that FlowMOP scaled favorably at larger event counts (Table 1).</span> <span style="color:#0066cc">Values are mean ± SD across three measured repeats after one warm-up run.</span> <span style="color:#0066cc">FlowMOP was fastest from 100,000 events onward and had the lowest peak RAM at all sizes except 100,000 events.</span> <span style="color:#0066cc">At 1,000,000 events, FlowMOP completed in 10.69 ± 0.16 s, compared with 24.21 ± 0.23 s for PeacoQC and 30.34 ± 0.48 s for FlowCut, while using 658.0 ± 0.3 MB peak RAM compared with 1483.4 ± 86.8 MB and 1115.1 ± 0.3 MB.</span> <span style="color:#0066cc">At 2,000,000 events, FlowMOP completed in 17.42 ± 0.88 s, compared with 43.18 ± 4.23 s and 60.61 ± 2.37 s, while using 839.5 ± 0.4 MB peak RAM compared with 2614.8 ± 0.1 MB and 1849.8 ± 0.3 MB.</span>

<span style="color:#0066cc">**Table 1: Clone-based time-gating runtime and memory benchmark**</span>

<span style="color:#0066cc">Mean runtime and peak RAM across three measured repeats are shown as mean ± SD.</span> <span style="color:#0066cc">The best-performing runtime and RAM value for each event count is shown in bold.</span>

| <span style="color:#0066cc">Events</span> | <span style="color:#0066cc">FlowMOP time (s)</span> | <span style="color:#0066cc">PeacoQC time (s)</span> | <span style="color:#0066cc">FlowCut time (s)</span> | <span style="color:#0066cc">FlowMOP RAM (MB)</span> | <span style="color:#0066cc">PeacoQC RAM (MB)</span> | <span style="color:#0066cc">FlowCut RAM (MB)</span> |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <span style="color:#0066cc">10,000</span> | <span style="color:#0066cc">2.67 ± 0.50</span> | <span style="color:#0066cc">4.40 ± 0.57</span> | <span style="color:#0066cc">**1.72 ± 0.02**</span> | <span style="color:#0066cc">**237.9 ± 0.2**</span> | <span style="color:#0066cc">446.6 ± 0.2</span> | <span style="color:#0066cc">261.8 ± 0.3</span> |
| <span style="color:#0066cc">100,000</span> | <span style="color:#0066cc">**4.34 ± 0.11**</span> | <span style="color:#0066cc">13.74 ± 1.34</span> | <span style="color:#0066cc">4.78 ± 0.33</span> | <span style="color:#0066cc">351.6 ± 0.1</span> | <span style="color:#0066cc">606.6 ± 26.1</span> | <span style="color:#0066cc">**326.4 ± 0.1**</span> |
| <span style="color:#0066cc">300,000</span> | <span style="color:#0066cc">**8.42 ± 0.85**</span> | <span style="color:#0066cc">15.08 ± 0.84</span> | <span style="color:#0066cc">10.91 ± 0.49</span> | <span style="color:#0066cc">**462.3 ± 0.4**</span> | <span style="color:#0066cc">948.5 ± 5.9</span> | <span style="color:#0066cc">505.4 ± 0.4</span> |
| <span style="color:#0066cc">1,000,000</span> | <span style="color:#0066cc">**10.69 ± 0.16**</span> | <span style="color:#0066cc">24.21 ± 0.23</span> | <span style="color:#0066cc">30.34 ± 0.48</span> | <span style="color:#0066cc">**658.0 ± 0.3**</span> | <span style="color:#0066cc">1483.4 ± 86.8</span> | <span style="color:#0066cc">1115.1 ± 0.3</span> |
| <span style="color:#0066cc">2,000,000</span> | <span style="color:#0066cc">**17.42 ± 0.88**</span> | <span style="color:#0066cc">43.18 ± 4.23</span> | <span style="color:#0066cc">60.61 ± 2.37</span> | <span style="color:#0066cc">**839.5 ± 0.4**</span> | <span style="color:#0066cc">2614.8 ± 0.1</span> | <span style="color:#0066cc">1849.8 ± 0.3</span> |

#### Synthetic Debris Gating Benchmark

![Embedded image 3](FlowMOP_submission_media/image3.png)

Figure 3. A) Representative flow cytometry plots showing FSC-A/SSC-A debris plots. The first plot represents a representative combined high + low debris sample, the second and third represent the high debris portion and the low debris portion of that sample respectively. The percentages below denote the proportion of the combined sample represented by the annotated plot. B) FSC-A/SSC-A flow plots of the Fig. 3A’s representative sample post-processing. Again, the first column shows the combined high + low debris sample, the second and third show the high debris and low debris samples separately respectively. The percentages in the first column denote the proportion of the sample remaining relative to the original combined sample. Percentages in the  high debris and low debris columns denote the post-filtering proportion that each comprises (rounded to the nearest percentage). C) Bar plot representing mean proportion percentage ± SD of low debris sample remaining post-processing for FlowMOP and four human experts. FlowMOP has significantly higher low debris sample proportion than Expert 4 (Bonferroni adjusted paired t-test, p = 0.04). D) Bar plots showing mean low debris sample proportion percentage ± SD following human expert gating using either a group or per-sample based strategy (Blue bars denote group-wise results, green sample-wise). No difference was found between the two methods (un-adjusted paired t-test, p > 0.05).

<span style="color:#0066cc">FlowMOP’s debris gating performance was tested by its ability to deplete the high-debris component in the combined high- and low-debris samples (Fig. 3A).</span> <span style="color:#0066cc">FlowMOP reduced the high-debris source proportion from 50 ± 1.10% to 28.8 ± 0.1% (paired t-test, p < 0.001) (Fig. 3B).</span> <span style="color:#0066cc">The final gate is applied on FSC-A, but its threshold is derived from candidate FSC-A distributions across eligible fluorescence-positive populations rather than from the overall FSC-A histogram alone.</span> <span style="color:#0066cc">FlowMOP did not differ significantly from any human evaluator except Expert 4, for whom FlowMOP removed more labelled high-debris events (Bonferroni-adjusted paired t-test, p = 0.04) (Fig. 3C).</span>

FlowMOP by default conducts debris removal on a per-sample basis. <span style="color:#0066cc">For comparison with the standard per-group gating methodology, human experts were also requested to apply per-sample and per-group gating strategies for comparison.</span> No difference was found between per-sample and group-based gating for any expert (Fig. 3D, unadjusted paired t-Test p > 0.05).

#### Synthetic Doublet Gating Benchmark

FlowMOP’s doublet removal performance was also examined through synthetic technical controls (Fig. 4A). <span style="color:#0066cc">Events positive for both CFSE and CTV provide an observable class of heterologous labelled doublets, subject to rare dye-transfer events; same-label CTV-CTV and CFSE-CFSE doublets are not identifiable from these labels alone.</span> <span style="color:#0066cc">FlowMOP was then run on these samples, and the proportion of remaining CFSE/CTV double-positive cells was compared with the proportions remaining after human-expert gating (Fig. 4A).</span>

![Embedded image 4](FlowMOP_submission_media/image4.png)

Figure 4. A) Representative flow cytometry plots of synthetic doublet samples and gating. The first row shows all samples by CTV against CFSE, the second FSC-A/FSC-H, and the last by SSC-H/SSC-A. The columns show representative CTV-only stained, CFSE-only stained, mixed CTV-CFSE, human expert doublet-removed, and FlowMOP doublet removed samples. B) Bar graph showing the mean percentage ± SD of CTV-CFSE double positive events removed (relative to the original sample) following FlowMOP or human expert processing. C) Bar graphs showing mean frequency ± SD of remaining CTV-CFSE double positive events following human expert processing, comparing group and per-sample based gating strategies (Blue bars denote group-wise results, green sample-wise). Only Expert 3 had significantly different results (unadjusted paired t-test, p = 0.009, all others p > 0.05).

<span style="color:#0066cc">FlowMOP significantly decreased the frequency of CTV-CFSE double-positive events from 7.84 ± 1.21% to 0.27 ± 0.11% (paired t-test; p = 0.001).</span> <span style="color:#0066cc">No statistically significant difference was detected between FlowMOP and any expert for this endpoint (Fig. 4B, paired t-test; unadjusted p > 0.05).</span> <span style="color:#0066cc">To assess whether sample-wise gating systematically affected the comparison, human experts also performed sample-wise and groupwise doublet gating.</span> <span style="color:#0066cc">No statistical difference was detected between these approaches except for Expert 3 (Fig. 4C, paired t-test; unadjusted p values).</span> <span style="color:#0066cc">Expert 3 consistently removed fewer doublets with the sample-wise method than with the group method (Fig. 4C, paired t-test; p = 0.009).</span>

### <span style="color:#0066cc">Biological validation</span>

<span style="color:#0066cc">Following the objective synthetic benchmarks, FlowMOP will be evaluated using prespecified biological population endpoints in biological flow-cytometry datasets.</span>

<span style="color:#0066cc">**[PLACEHOLDER: Figure 5 biological-validation cohort, endpoints, results, and statistical comparisons.]**</span>

<span style="color:#0066cc">**[PLACEHOLDER: Figure 5. Biological validation of FlowMOP cleaning.]**</span>

### <span style="color:#0066cc">Tumour biological validation</span>

<span style="color:#0066cc">Next, we wished to examine FlowMOP's performance in a more complex sample type, such as tumour samples.</span> <span style="color:#0066cc">Consequently, we tested whether manual and FlowMOP preprocessing produced different downstream population measurements in three tumour samples (Fig. 6A).</span> <span style="color:#0066cc">Relative to matched Raw values, the mean ± SD T:B-cell ratio was 85.6 ± 59.2% after manual preprocessing and 108.6 ± 112.2% after FlowMOP preprocessing.</span> <span style="color:#0066cc">No comparison was significant (Manual versus Raw, p = 0.715; FlowMOP versus Raw, p = 0.906; Manual versus FlowMOP, p = 0.636).</span>

<span style="color:#0066cc">The Live CD45+ cell count was 62.9 ± 17.8% of Raw after manual preprocessing and 73.3 ± 7.3% after FlowMOP preprocessing.</span> <span style="color:#0066cc">FlowMOP differed from Raw (p = 0.024), whereas Manual did not (p = 0.069), and Manual and FlowMOP did not differ (p = 0.545).</span> <span style="color:#0066cc">B-cell frequency was 32.5 ± 18.8% of Raw after Manual and 32.2 ± 17.0% after FlowMOP; both differed from Raw (p = 0.025 and p = 0.020, respectively), but not from each other (p = 0.855).</span> <span style="color:#0066cc">T-cell frequency was 23.0 ± 20.4% of Raw after Manual and 22.8 ± 10.6% after FlowMOP; both differed from Raw (p = 0.023 and p = 0.006, respectively), but not from each other (p = 0.985).</span> <span style="color:#0066cc">Thus, although both preprocessing strategies changed absolute population recovery relative to Raw, no difference between Manual and FlowMOP was detected for any prespecified endpoint.</span>

![Figure 6](figs_data/figure_6.png)

<span style="color:#0066cc">Figure 6. Downstream tumour-population measurements after manual and FlowMOP preprocessing. A) CD3-versus-CD19 plots for tumour Samples 1, 2, and 3 in the Raw data and after Manual or FlowMOP preprocessing. T cells are CD3+CD19− (Q1), and B cells are CD3−CD19+ (Q3). B) Live CD45+ cell count, B-cell frequency, and T-cell frequency after normalization within sample to the matched Raw value (Raw = 100%). Small circles show individual samples, lines connect the matched samples across preprocessing methods, and diamonds and error bars show mean ± SD. All three unadjusted, two-sided paired t-tests were performed for each endpoint; brackets and P values are displayed only for significant comparisons (p < 0.05).</span>

## Discussion

<span style="color:#0066cc">FlowMOP was evaluated for time, debris, and doublet gating using synthetic technical controls and biological datasets.</span>

### Synthetic Benchmarking

#### Synthetic Sample Time Gating

In the synthetic time-gating analysis, the Segment time gate is perhaps the most common and consequential time-artifact, as a non-negligible sample portion is often required to be removed in real samples. This type of synthetic data expects gating most similar to current manual gating, where blocks of events are removed. The objective of these samples is to simulate where there is a long blockage or sudden shift in the acquired sample. <span style="color:#0066cc">Here, FlowMOP had higher sensitivity than FlowCut at both tested bin sizes. It also had higher specificity than both competitors at 5000 events and than FlowCut at 2000 events.</span> <span style="color:#0066cc">The Time-only mechanism benchmark suggests that the FlowMOP versus FlowCut difference is not explained solely by implementation.</span> <span style="color:#0066cc">When local acquisition-rate structure was altered either in alignment with source-linked fluorescence changes or independently of them, FlowMOP remained unchanged, whereas FlowCut's removal behavior shifted, especially in Segment inputs (Fig. S4).</span> <span style="color:#0066cc">This supports the interpretation that FlowCut can be affected by acquisition-density changes even when fluorescence values are unchanged, while FlowMOP is more anchored to fluorescence-population summaries across acquisition order.</span>

<span style="color:#0066cc">Transient acquisition disturbances are not confined to the beginning or end of a run: flow-rate surges interspersed throughout acquisition and associated signal-intensity variation have been documented [10].</span> <span style="color:#0066cc">PeacoQC notes that temporary acquisition problems can be difficult to detect manually [5], and FlowCut describes manual identification and removal of transient acquisition problems as time-consuming and subjective [9].</span> <span style="color:#0066cc">We use “microblockage” operationally for a short, self-resolving instance of this broader phenomenon that produces a localized fluorescence shift, without asserting that a physical obstruction was directly observed.</span> <span style="color:#0066cc">The Bimix and Trimix samples were designed to represent this under-addressed case.</span> <span style="color:#0066cc">Although these synthetic samples can appear acceptable on visual inspection, the source labels show that the short altered intervals contain events from an intentionally perturbed fluorescence source and therefore should be excluded under the benchmark definition.</span> <span style="color:#0066cc">Visual subtlety is thus a central feature of the benchmark: it demonstrates why apparent normality by eye is not sufficient ground truth.</span>

<span style="color:#0066cc">The size of the simulated microblockage was determined by the mixing-bin size, with the 2000-event samples representing shorter and more difficult disturbances than the 5000-event samples. FlowMOP provided the strongest overall performance profile across these conditions. At 5000 events, it had higher sensitivity than FlowCut and higher specificity than both competitors across Segment, Bimix, and Trimix. At 2000 events, FlowMOP had higher sensitivity and specificity than FlowCut in Segment and retained higher specificity than PeacoQC in Bimix and Trimix without a significant sensitivity disadvantage. Thus, FlowMOP's principal advantage was its consistent combination of high sensitivity with stronger specificity across sustained and short, interspersed time artifacts.</span>

<span style="color:#0066cc">PeacoQC's documentation identifies acquisition-bin size as a trade-off between the accuracy of within-bin density estimation, the number of bins available for evaluating signal stability, and the number of events affected when a bin is removed [5]. The Bimix and Trimix benchmarks contain short, interspersed source-defined intervals. FlowMOP's stronger performance is therefore consistent with its temporal summarisation being better matched to these brief intervals.</span>

<span style="color:#0066cc">The computational benchmark demonstrates that FlowMOP provides speed gains with lower peak RAM usage at larger event counts.</span> <span style="color:#0066cc">This is most evident from 300,000 events onward, where FlowMOP had both the fastest mean runtime and lowest peak memory use.</span> <span style="color:#0066cc">At 2,000,000 events, FlowMOP was approximately 2.5-fold faster than PeacoQC and 3.5-fold faster than FlowCut, while using approximately 32% of PeacoQC’s peak RAM and 45% of FlowCut’s peak RAM.</span>

<span style="color:#0066cc">These measurements were obtained using local non-distributed execution and therefore demonstrate the performance of the tested implementations, not a distributed-computing speedup.</span>

<span style="color:#0066cc">Among the three evaluated decision structures, only FlowMOP exposes an inherently channel-parallel within-sample computation.</span> <span style="color:#0066cc">After shared acquisition bins are defined, each eligible fluorescence channel independently establishes its reference, produces a fixed-shape time-bin summary, and applies smoothing and MAD filtering; cross-channel coordination occurs only in the final parameter vote.</span> <span style="color:#0066cc">PeacoQC instead performs data-dependent peak identification and reconciliation across bin-channel combinations, while FlowCut applies adjacent-segment, contiguous-region, file-wide, and conditional rerun decisions [5,9].</span> <span style="color:#0066cc">Their suboperations and separate files can be parallelised, but their complete within-sample decision pipelines cannot be expressed as the same independent channel-wise reduction without changing their decision rules.</span> <span style="color:#0066cc">Porting either implementation to another language or scheduling framework would therefore not remove these structural dependencies.</span> <span style="color:#0066cc">This structural distinction, together with FlowMOP’s earlier data reduction, fewer full-data passes, and smaller intermediate state, is consistent with its lower runtime and memory use in our benchmarks, although the benchmark does not independently establish causation.</span>

It is of note that there is a large variation in algorithmic performance across the dataset. <span style="color:#0066cc">One source of this variation is that the 0.5 and 1.0 relative cell concentrations oftentimes exhibited marginal differences in fluorescence intensity (Supp. Fig. 3), especially relative to the 0.5 / 3.0 cell concentrations comparison.</span> Consequently, the 0.5/1.0 discrimination tasks can be considered especially difficult benchmarks to overcome. However, this difficulty was intentionally placed, to ensure the present benchmarking dataset could also show progressive improvement of future time-gating algorithms.

<span style="color:#0066cc">Human gating was not included for the Bimix or Trimix synthetic datasets because the short mixed bins do not provide a practical manual ground-truth target.</span> <span style="color:#0066cc">The retained source labels instead provide an event-level reference for comparing algorithmic performance.</span> <span style="color:#0066cc">A dataset benchmark was therefore necessary to evaluate whether automated methods can detect these subtle but labelled artifacts reproducibly.</span>

#### Synthetic Sample Debris and Doublet Gating

<span style="color:#0066cc">In the synthetic debris and doublet gating trials, FlowMOP removed the labelled technical artifact populations effectively (Fig. 3B, 4B).</span> <span style="color:#0066cc">In the debris task, FlowMOP enriched the low-debris component by 9.67%, which represented approximately 19% more debris removed considering the original 50:50 debris / real sample mixture.</span> <span style="color:#0066cc">FlowMOP’s debris performance can also be interpreted in relation to the two debris populations (Fig. 3A) present: one at <10,000 FSC-A units, and the second at ~20,000 FSC-A units.</span> <span style="color:#0066cc">Human experts were instructed that this second debris population was debris, and to gate accordingly.</span> <span style="color:#0066cc">FlowMOP was able to independently detect this second debris population and exclude it without external information.</span>

Similarly, in the doublet removal, the synthetic samples, owing to the rather unique preparation, yielded triplets. FlowMOP was able to handle this unexpected population and successfully removed it.

<span style="color:#0066cc">The synthetic debris benchmark measures depletion of the source-labelled high-debris component from matched mixtures of high- and low-debris samples.</span> <span style="color:#0066cc">Because both sources contain some debris, these labels represent relative debris enrichment rather than per-event debris classification.</span> <span style="color:#0066cc">FlowMOP targets the small, low-FSC debris phenotype observed in these controls and successfully removed the two low-FSC debris populations shown in Figure 3A.</span> <span style="color:#0066cc">SSC-A may improve recognition of larger or internally complex debris, while pulse-width measurements may assist with aggregates.</span> <span style="color:#0066cc">Accordingly, FlowMOP does not currently incorporate these features because broadly applicable decision rules are difficult to establish across tissues, panels, instruments, and acquisition settings.</span> <span style="color:#0066cc">Future extensions can evaluate configurable or sample-specific multivariate approaches in tumour digests with greater necrosis, aggregation, and scatter heterogeneity.</span>

### <span style="color:#0066cc">Biological validation</span>

<span style="color:#0066cc">**[PLACEHOLDER: Interpretation of the Figure 5 biological-validation endpoint and its implications for preservation of biologically meaningful populations following cleaning.]**</span>

<span style="color:#0066cc">The tumour analysis provides an initial downstream biological assessment in a complex sample type (Fig. 6).</span> <span style="color:#0066cc">Manual and FlowMOP preprocessing both reduced Live CD45+ recovery and B- and T-cell frequencies relative to Raw, but neither produced a detectable change in the T:B-cell ratio.</span> <span style="color:#0066cc">More importantly for method concordance, Manual and FlowMOP did not differ for any of the four prespecified endpoints.</span> <span style="color:#0066cc">FlowMOP therefore appears to extend to complex tumour samples, with broadly comparable downstream results to matched human-expert manual gating.</span>

<span style="color:#0066cc">An earlier expert preference exercise across nine human and mouse datasets is reported in Supplementary Figures S5-S7. The exercise used an earlier FlowMOP configuration and involved one relevant expert per tissue type. These rankings are therefore interpreted as exploratory relative preferences rather than absolute quality scores or evidence of the performance of the current implementation.</span>

### Other remarks

Automated Live/Dead classification of events was considered, however not implemented in the algorithm. There exist many varied protocols and methods for discriminating live/dead samples, along with great diversity in the determination of what constitutes a ‘dead’ event. Consequently, the difficulty of creating a universal live/dead discriminator is non-trivial. Finally, there may be potential significant biological insight in the ‘dead’ cells of a sample, whereby important information concerning a sample may be found in the dead events or their proportion.

<span style="color:#0066cc">The biological cost of preprocessing errors is difficult to measure directly, and neither under-cleaning nor over-cleaning is preferable.</span> <span style="color:#0066cc">Under-cleaning may allow acquisition-time artifacts with abnormal staining patterns to confound downstream results, including by creating, inflating, or obscuring apparent rare populations.</span> <span style="color:#0066cc">Conversely, over-cleaning could remove rare or transient biological populations.</span> <span style="color:#0066cc">Because FlowMOP's time-gating module operates on acquisition-time structure rather than population identity, rare populations are not expected to be systematically biased unless they are temporally confounded with an acquisition artifact.</span>

<span style="color:#0066cc">The primary comparison used recommended or automatically selected settings, including fixed FlowMOP parameters, to reflect typical unsupervised use.</span>

<span style="color:#0066cc">CTV-CFSE double-positive events provide an observable ground-truth doublet class, but same-label CTV-CTV and CFSE-CFSE doublets are not directly detectable in this validation design.</span> <span style="color:#0066cc">FlowMOP requires appropriate acquisition voltage/gain settings; if relevant signals are poorly resolved or saturated, the lost information cannot be recovered and reliable cleaning cannot be guaranteed. FlowMOP currently expects users to identify the scatter channels used by the workflow; it has not been validated systematically across 405-nm, 488-nm, and polar 488-nm FSC/SSC configurations on instruments with multiple scatter measurements. Future versions could assess multiple scatter-channel pairs and select or combine the pair with the clearest debris/doublet separation.</span>

## Conclusion

<span style="color:#0066cc">FlowMOP provides time-gating, conservative low-FSC debris removal, and scatter-ratio doublet removal in a single Python implementation.</span> <span style="color:#0066cc">This facilitates integration with Python-based workflows and provides fast, memory-conscious preprocessing for large cytometry files.</span>

<span style="color:#0066cc">Within the tested synthetic scenarios, event-level source labels enabled objective evaluation of the targeted artifact classes.</span> <span style="color:#0066cc">FlowMOP had higher sensitivity than FlowCut for Segment anomalies at both bin sizes and higher specificity than both competitors across the 5000-event Segment, Bimix, and Trimix benchmarks.</span> <span style="color:#0066cc">For debris and doublet removal, FlowMOP removed the labelled technical artifact populations effectively in the synthetic ground-truth datasets, including unexpected triplet events.</span>

<span style="color:#0066cc">An exploratory expert comparison using an earlier FlowMOP configuration generally favoured manual gates and is reported in the Supplementary Information.</span> <span style="color:#0066cc">**[PLACEHOLDER: Figure 5 biological-validation conclusion.]**</span> <span style="color:#0066cc">In three tumour samples, Manual and FlowMOP preprocessing did not differ for the T:B-cell ratio, Live CD45+ cell count, B-cell frequency, or T-cell frequency.</span> <span style="color:#0066cc">The open-source Python implementation supports reproducible preprocessing across cytometry datasets of increasing scale.</span>

## Data and Code Availability

FlowMOP can be accessed via https://github.com/1ordinateur/FlowMOP. The code associated with the creation of this paper can be accessed at https://github.com/1ordinateur/FlowMOP_paper. The FCS Files used for this paper can be accessed at http://doi.org/10.5281/zenodo.17896445.

## Supplementary data

<span style="color:#0066cc">Figure S1: Construction of Segment, Bimix, and Trimix synthetic time samples. No flow-rate disturbance was introduced.</span>

![Figure S1](figs_data/synthetic_time_design_schematic.svg)

<span style="color:#0066cc">Figure S2: Representative flow cytometry CD3 / Time plots for Bimix 2000 bin, Trimix 5000 bin, and Trimix 2000 bin synthetic datasets, with original data inputs, and following cleaning by FlowMOP, FlowCut, and PeacoQC.</span> Percentages below each figure represent the retained proportion of cells relative to the original representative synthetic sample.

![Embedded image 8](FlowMOP_submission_media/image8.png)

**Table S1A: Dataset Compositions**

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

**Table S1B: Expert / Algorithm IDs**

| Expert / Algorithm | ID |
| --- | --- |
| Expert 1 | 1 |
| Expert 2 | 2 |
| Expert 3 | 3 |
| Expert 4 | 4 |
| FlowMOP | 5 |
| FlowCut | 6 |
| PeacoQC | 7 |

<span style="color:#0066cc">**Table S1C: Time gating rankings (1 = best)**</span>

| Rankings | Mouse DRG | Mouse Skin | Human Cultured T Cells | Mouse Bone Marrow |
| --- | --- | --- | --- | --- |
| 1 | 3 | 3 | 3 | 3 |
| 2 | 1 | 1 | 2 | 5 |
| 3 | 4 | 4 | 1 | 4 |
| 4 | 2 | 2 | 4 | 1 |
| 5 | 5 | 6 | 7 | 2 |
| 6 | 6 | 5 | 5 | 6 |
| 7 | 7 | 7 | 6 | 7 |
| Mouse Spleen | Mouse Blood | Mouse Brain | Mouse Central Nervous System | Human Liver |
| 3 | 4 | 4 | 1 | 3 |
| 4 | 3 | 1 | 4 | 1 |
| 1 | 1 | 5 | 3 | 2 |
| 7 | 2 | 3 | 2 | 4 |
| 5 | 6 | 2 | 5 | 6 |
| 6 | 5 | 7 | 7 | 7 |
| 2 | 7 | 6 | 6 | 5 |

<span style="color:#0066cc">**Table S1D: Debris gating rankings (1 = best)**</span>

| Rankings | Mouse DRG | Mouse Skin | Human Cultured T cells | Mouse Bone Marrow |
| --- | --- | --- | --- | --- |
| 1 | 3 | 2 | 4 | 3 |
| 2 | 2 | 3 | 2 | 2 |
| 3 | 1 | 5 | 3 | 1 |
| 4 | 4 | 4 | 1 | 4 |
| 5 | 5 | 1 | 5 | 5 |
| Mouse Spleen | Mouse Blood | Mouse Brain | Mouse Central Nervous System | Human Liver |
| 3 | 5 | 3 | 3 | 2 |
| 2 | 3 | 2 | 2 | 1 |
| 1 | 2 | 4 | 4 | 5 |
| 4 | 4 | 1 | 5 | 3 |
| 5 | 1 | 5 | 1 |  |

<span style="color:#0066cc">**Table S1E: Doublet gating rankings (1 = best)**</span>

| Rankings | Mouse DRG | Mouse Skin | Human Cultured T cells | Mouse Bone Marrow |
| --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 1 | 4 |
| 2 | 3 | 3 | 4 | 2 |
| 3 | 1 | 1 | 3 | 1 |
| 4 | 5 | 5 | 2 | 5 |
| 5 | 4 | 4 | 5 |  |
| Mouse Spleen | Mouse Blood | Mouse Brain | Mouse Central Nervous System | Human Liver |
| 3 | 3 | 3 | 4 | 3 |
| 4 | 2 | 1 | 3 | 5 |
| 1 | 1 | 4 | 1 | 2 |
| 5 | 5 | 5 | 2 | 4 |
| 2 | 4 | 2 | 5 | 1 |

**Table S2: Jeffreys’ Scale**

| Bayes Factor | Hypothesis descriptor |
| --- | --- |
| <1 | Null hypothesis supported |
| 1-3 | Anecdotal / Weak evidence |
| 3-10 | Moderate / Substantial evidence |
| 10-30 | Very strong evidence |
| >30 | Decisive evidence |

<span style="color:#0066cc">**Table S4: Full-dataset FlowMOP MAD-smoothing analysis**</span>

| <span style="color:#0066cc">Short, long smoothing factors</span> | <span style="color:#0066cc">Sensitivity</span> | <span style="color:#0066cc">Specificity</span> | <span style="color:#0066cc">Balanced mean</span> |
| --- | ---: | ---: | ---: |
| <span style="color:#0066cc">`0,0` (no-smoothing control)</span> | <span style="color:#0066cc">0.8021</span> | <span style="color:#0066cc">**0.7081**</span> | <span style="color:#0066cc">**0.75506**</span> |
| <span style="color:#0066cc">**`0.01,0.05` (current default)**</span> | <span style="color:#0066cc">0.8212</span> | <span style="color:#0066cc">0.6811</span> | <span style="color:#0066cc">**0.75114**</span> |
| <span style="color:#0066cc">`0.02,0.05`</span> | <span style="color:#0066cc">0.8228</span> | <span style="color:#0066cc">0.6794</span> | <span style="color:#0066cc">0.75108</span> |
| <span style="color:#0066cc">`0.01,0.02`</span> | <span style="color:#0066cc">0.8112</span> | <span style="color:#0066cc">0.6885</span> | <span style="color:#0066cc">0.74986</span> |
| <span style="color:#0066cc">`0.02,0.09`</span> | <span style="color:#0066cc">**0.8276**</span> | <span style="color:#0066cc">0.6709</span> | <span style="color:#0066cc">0.74923</span> |
| <span style="color:#0066cc">`0.01,0.09`</span> | <span style="color:#0066cc">0.8257</span> | <span style="color:#0066cc">0.6707</span> | <span style="color:#0066cc">0.74823</span> |
| <span style="color:#0066cc">`0.05,0.09`</span> | <span style="color:#0066cc">0.8188</span> | <span style="color:#0066cc">0.6676</span> | <span style="color:#0066cc">0.74317</span> |
| <span style="color:#0066cc">`0.02,0.20`</span> | <span style="color:#0066cc">0.8248</span> | <span style="color:#0066cc">0.6479</span> | <span style="color:#0066cc">0.73632</span> |
| <span style="color:#0066cc">`0.05,0.20`</span> | <span style="color:#0066cc">0.8197</span> | <span style="color:#0066cc">0.6520</span> | <span style="color:#0066cc">0.73586</span> |
| <span style="color:#0066cc">`0.01,0.20`</span> | <span style="color:#0066cc">0.8231</span> | <span style="color:#0066cc">0.6477</span> | <span style="color:#0066cc">0.73540</span> |
| <span style="color:#0066cc">`0.10,0.20`</span> | <span style="color:#0066cc">0.8138</span> | <span style="color:#0066cc">0.6415</span> | <span style="color:#0066cc">0.72769</span> |
| <span style="color:#0066cc">`0.05,0.34`</span> | <span style="color:#0066cc">0.8194</span> | <span style="color:#0066cc">0.6344</span> | <span style="color:#0066cc">0.72686</span> |
| <span style="color:#0066cc">`0.02,0.34`</span> | <span style="color:#0066cc">0.8199</span> | <span style="color:#0066cc">0.6286</span> | <span style="color:#0066cc">0.72424</span> |
| <span style="color:#0066cc">`0.10,0.34`</span> | <span style="color:#0066cc">0.8134</span> | <span style="color:#0066cc">0.6250</span> | <span style="color:#0066cc">0.71921</span> |
| <span style="color:#0066cc">`0.10,0.50`</span> | <span style="color:#0066cc">0.8139</span> | <span style="color:#0066cc">0.6229</span> | <span style="color:#0066cc">0.71842</span> |
| <span style="color:#0066cc">`0.10,0.90` (former default)</span> | <span style="color:#0066cc">0.8139</span> | <span style="color:#0066cc">0.6226</span> | <span style="color:#0066cc">0.71827</span> |
| <span style="color:#0066cc">`0.10,1.00`</span> | <span style="color:#0066cc">0.8139</span> | <span style="color:#0066cc">0.6226</span> | <span style="color:#0066cc">0.71827</span> |
| <span style="color:#0066cc">`0.20,0.90`</span> | <span style="color:#0066cc">0.8007</span> | <span style="color:#0066cc">0.6142</span> | <span style="color:#0066cc">0.70747</span> |
| <span style="color:#0066cc">`0.40,0.90`</span> | <span style="color:#0066cc">0.7812</span> | <span style="color:#0066cc">0.6140</span> | <span style="color:#0066cc">0.69756</span> |

<span style="color:#0066cc">Values are equally weighted macro-averages across the six Figure 2 benchmark groups (173 primary inputs; six tied-composition inputs excluded). The balanced mean is the arithmetic mean of sensitivity and specificity. Bold indicates the highest value overall and the highest balanced mean among smoothed settings.</span>


<span style="color:#0066cc">Figure S3:</span>

Fluorescence variation as a function of cell concentration.

![Embedded image 9](FlowMOP_submission_media/image9.jpeg)

<span style="color:#0066cc">Figure S4:</span>

<span style="color:#0066cc">Time-only acquisition-rate perturbations reveal FlowCut sensitivity to local Time density. Points show raw-matched changes in sensitivity and specificity, with large points and intervals showing the mean and 95% confidence interval. Negative values indicate reduced performance relative to the raw control. Columns show all inputs together and the Segment, Bimix, and Trimix subsets separately. FlowMOP and PeacoQC remain unchanged under both source-linked and random Time warping. In contrast, FlowCut's sensitivity and specificity shift after Time-only perturbation, with the clearest specificity loss in random Time-warped inputs and the strongest source-linked sensitivity loss in Segment inputs, where acquisition-rate structure aligns with source composition.</span>

![Figure S4](figs_data/revision_timewarp_mechanism.svg)

### <span style="color:#0066cc">Supplementary expert preference evaluation</span>

<span style="color:#0066cc">The expert-ranking exercise was performed using gates generated with an earlier FlowMOP configuration. It is therefore retained as an exploratory historical evaluation rather than a primary assessment of the current selected implementation. The rankings measure relative preference and do not establish absolute gating adequacy or event-level accuracy.</span>

#### <span style="color:#0066cc">Supplementary methods</span>

<span style="color:#0066cc">The expert-ranking analysis was used to summarize relative preferences among gates generated by FlowMOP, comparator algorithms, and human operators; it was not treated as an absolute measure of gating adequacy.</span>

<span style="color:#0066cc">Rankings were modelled using a Plackett–Luce model with latent method abilities; identifiability was enforced by fixing a reference ability to zero. Independent Normal priors were placed on non-reference abilities. Posterior inference was performed with an affine-invariant ensemble Markov Chain Monte Carlo sampler (emcee; 32 walkers, 5,000 iterations; 1,000 burn-in), and posterior medians with 95% credible intervals were reported. Directional hypotheses (superiority/inferiority) were evaluated by computing P = Pr(H1 | data) and converting to a Bayes factor BF₁₀ = p/(1 – p) under equal prior odds, interpreted using Jeffreys’ scale and reported as BF, P (Bayes factor, posterior probability of the alternative hypothesis) (Table S2).</span>

<span style="color:#0066cc">For each Plackett–Luce fit, the ensemble sampler’s mean acceptance fraction and an integrated-autocorrelation-time-based effective sample size were monitored (both as implemented in emcee) to assess MCMC convergence. Across all analyses, mean acceptance fractions ranged from 0.520 to 0.60, and effective sample sizes ranged from 128,003 to 134,042.</span>

<span style="color:#0066cc">Posterior predictive checks compared observed pairwise win counts with those implied by posterior draws under a Bradley–Terry formulation, using a chi-square-type discrepancy averaged over method pairs. The maximum average χ² per comparison was 0.56, indicating good agreement between the model and the observed rankings.</span>

<span style="color:#0066cc">Computations were implemented in Python with numerically stable log-likelihoods.</span>

#### <span style="color:#0066cc">Supplementary results</span>

<span style="color:#0066cc">FlowMOP outputs were compared with expert-provided gates using a forced-ranking preference task across nine biological datasets. These rankings measure relative expert preference and should not be interpreted as an absolute measure of gating adequacy. The resulting time, debris, and doublet gates were ranked by an expert familiar with that sample type. FlowCut and PeacoQC were also included in the time-gating comparison. Rank 1 indicates greatest preference.</span>

![Supplementary Figure S5](FlowMOP_submission_media/image5_revised.png)

<span style="color:#0066cc">Figure S5. Expert preference rankings for time gates provided by four human experts, FlowMOP (black border), FlowCut, and PeacoQC across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. Abbreviations: DRG, dorsal root ganglion; CNS, central nervous system.</span>

<span style="color:#0066cc">In the biological datasets, FlowMOP had the lowest mean rank among the algorithmic time-gating approaches (Fig. S5). In the mouse brain and mouse bone marrow tasks, it ranked third and second, respectively (Fig. S5). On a Bayesian analysis, FlowMOP was observed to be substantially preferred to FlowCut (BF = 5.39, P = 84.3%) and strongly preferred to PeacoQC (BF = 12.10, P = 92.4%). FlowMOP was ranked inferiorly to all human experts with strong to decisive evidence (Expert 2 BF = 10.55, P = 91.3%, all others BF > 100, P = 100%).</span>

![Supplementary Figure S6](FlowMOP_submission_media/image6_revised.png)

<span style="color:#0066cc">Figure S6. Expert preference rankings for debris gates provided by four human experts and FlowMOP (black border) across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across available datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. N/A denotes an unavailable gate.</span>

![Supplementary Figure S7](FlowMOP_submission_media/image7_revised.png)

<span style="color:#0066cc">Figure S7. Expert preference rankings for doublet gates provided by four human experts and FlowMOP (black border) across nine datasets. Rows indicate the gate provider, and the final column shows the mean rank across available datasets. Rank 1 indicates greatest preference; therefore, a lower average score indicates greater preference. N/A denotes an unavailable gate.</span>

<span style="color:#0066cc">FlowMOP had the highest mean rank when compared with the human experts for debris and doublet removal (Figs. S6, S7). On a Bayesian analysis, substantial to strong evidence was observed for FlowMOP being inferior to Expert 1 (BF = 5.87, P = 85.5%), Expert 4 (BF = 12.85, P = 92.8%), and Experts 2 and 3 (BF > 100, P = 100%) in debris removal. In doublet removal, FlowMOP was weakly inferiorly ranked to Expert 4 (BF = 3.14, P = 75.8%), substantially inferiorly ranked to Expert 1 (BF = 5.67, P = 85.0%), strongly inferiorly ranked to Expert 2 (BF = 14.38, P = 93.5%), and decisively inferiorly ranked to Expert 3 (BF > 100, P = 100%).</span>

<span style="color:#0066cc">FlowMOP’s relative expert preference varied across datasets. For debris, it ranked first in the mouse blood task and third in the human liver and mouse skin datasets. In the doublet task, FlowMOP ranked second in the human liver task. Full tabular rankings are provided in Tables S1B-E.</span>

![Supplementary Figure S8](figs_data/Supp_fig_8.png)

<span style="color:#0066cc">Figure S8. Representative preprocessing strategies for tumour Sample 1. Manual preprocessing applies time, cells/debris, and single-cell gates sequentially from left to right. FlowMOP calculates time, debris, and doublet exclusions independently in parallel. The intersection of events retained by the time, debris, and doublet gates forms the final FlowMOP population.</span>

## References

[1]	Aysun Adan, Günel Alizada, Yağmur Kiraz, Yusuf Baran, and Ayten Nalbant. 2017. Flow cytometry: basic principles and applications. Critical Reviews in Biotechnology 37, 2 (February 2017), 163–176. https://doi.org/10.3109/07388551.2015.1128876

[2]	Thomas Myles Ashhurst, Felix Marsh-Wakefield, Givanna Haryono Putri, Alanna Gabrielle Spiteri, Diana Shinko, Mark Norman Read, Adrian Lloyd Smith, and Nicholas Jonathan Cole King. 2022. Integration, exploration, and analysis of high-dimensional single-cell cytometry data using Spectre. Cytometry Part A 101, 3 (2022), 237–253. https://doi.org/10.1002/cyto.a.24350

[3]	Noah Castelo, Maarten W. Bos, and Donald R. Lehmann. 2019. Task-Dependent Algorithm Aversion. Journal of Marketing Research 56, 5 (October 2019), 809–825. https://doi.org/10.1177/0022243719851788

[4]	Antonio Cosma. 2020. The Nightmare of a Single Cell: Being a Doublet. Cytometry A 97, 8 (August 2020), 768–771. https://doi.org/10.1002/cyto.a.23929

[5]	<span style="color:#0066cc">Annelies Emmaneel, Katrien Quintelier, Dorine Sichien, Paulina Rybakowska, Concepción Marañón, Marta E. Alarcón-Riquelme, Gert Van Isterdael, Sofie Van Gassen, and Yvan Saeys. 2022. PeacoQC: Peak-based selection of high quality cytometry data. Cytometry Part A 101, 4 (2022), 325–338. https://doi.org/10.1002/cyto.a.24501</span>

[6]	<span style="color:#0066cc">Kipper Fletez-Brant, Josef Špidlen, Ryan R. Brinkman, Mario Roederer, and Pratip K. Chattopadhyay. 2016. flowClean: Automated identification and removal of fluorescence anomalies in flow cytometry data. Cytometry Part A 89, 5 (2016), 461–471. https://doi.org/10.1002/cyto.a.22837</span>

[7]	<span style="color:#0066cc">Zicheng Hu, Alice Tang, Jaiveer Singh, Sanchita Bhattacharya, and Atul J. Butte. 2020. A robust and interpretable end-to-end deep learning model for cytometry data. Proceedings of the National Academy of Sciences 117, 35 (September 2020), 21373–21380. https://doi.org/10.1073/pnas.2003026117</span>

[8]	<span style="color:#0066cc">Nanditha Mallesh. 2023. Automated analysis of flow cytometry using deep learning for the detection of B-cell neoplasms. Thesis. Universitäts- und Landesbibliothek Bonn. Retrieved August 7, 2023 from https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/10949</span>

[9]	<span style="color:#0066cc">Justin Meskas, Daniel Yokosawa, Sherrie Wang, Gabriela C. Segat, and Ryan Remy Brinkman. 2023. FlowCut: An R package for automated removal of outlier events and flagging of files based on time versus fluorescence analysis. Cytometry Part A 103, 1 (2023), 71–81. https://doi.org/10.1002/cyto.a.24670</span>

[10]	<span style="color:#0066cc">Gianni Monaco, Hao Chen, Michael Poidinger, Jinmiao Chen, João Pedro de Magalhães, and Anis Larbi. 2016. flowAI: automatic and interactive anomaly discerning tools for flow cytometry data. Bioinformatics 32, 16 (August 2016), 2473–2480. https://doi.org/10.1093/bioinformatics/btw191</span>

[11]	<span style="color:#0066cc">Christina Bligaard Pedersen, Søren Helweg Dam, Mike Bogetofte Barnkob, Michael D. Leipold, Noelia Purroy, Laura Z. Rassenti, Thomas J. Kipps, Jennifer Nguyen, James Arthur Lederer, Satyen Harish Gohil, Catherine J. Wu, and Lars Rønn Olsen. 2022. cyCombine allows for robust integration of single-cell cytometry datasets within and across technologies. Nat Commun 13, 1 (March 2022), 1698. https://doi.org/10.1038/s41467-022-29383-5</span>

[12]	<span style="color:#0066cc">Lisa Weijler, Florian Kowarsch, Michael Reiter, Pedro Hermosilla, Margarita Maurer-Granofszky, and Michael Dworzak. 2024. FATE: Feature-Agnostic Transformer-Based Encoder for Learning Generalized Embedding Spaces in Flow Cytometry Data. 2024. 7956–7964. Retrieved May 3, 2024 from https://openaccess.thecvf.com/content/WACV2024/html/Weijler_FATE_Feature-Agnostic_Transformer-Based_Encoder_for_Learning_Generalized_Embedding_Spaces_in_WACV_2024_paper.html</span>

[13]	<span style="color:#0066cc">Kenneth Lo, Ryan Remy Brinkman, and Raphael Gottardo. 2008. Automated gating of flow cytometry data via robust model-based clustering. Cytometry Part A 73A, 4 (April 2008), 321–332. https://doi.org/10.1002/cyto.a.20531</span>

[14]	<span style="color:#0066cc">Lukas Fisch, Michael Heming, Andreas Schulte-Mecklenbeck, Catharina C. Gross, Stefan Zumdick, Carlotta Barkhau, Daniel Emden, Jan Ernsting, Ramona Leenings, Kelvin Sarink, Nils R. Winter, Udo Dannlowski, Heinz Wiendl, Gerd Meyer zu Hörste, and Tim Hahn. 2024. GateNet: A novel neural network architecture for automated flow cytometry gating. Computers in Biology and Medicine 179 (September 2024), 108820. https://doi.org/10.1016/j.compbiomed.2024.108820</span>

[15]	<span style="color:#0066cc">Jiong Chen, Matei Ionita, Yanbo Feng, Yinfeng Lu, Patryk Orzechowski, Sumita Garai, Kenneth Hassinger, Jingxuan Bao, Junhao Wen, Duy Duong-Tran, Joost Wagenaar, Michelle L. McKeague, Mark M. Painter, Divij Mathew, Ajinkya Pattekar, Nuala J. Meyer, E. John Wherry, Allison R. Greenplate, and Li Shen. 2025. Automated cytometric gating with human-level performance using bivariate segmentation. Nature Communications 16, 1 (February 2025), 1576. https://doi.org/10.1038/s41467-025-56622-2</span>

[16]	<span style="color:#0066cc">Chris P. Verschoor, Alina Lelic, Jonathan L. Bramson, and Dawn M. E. Bowdish. 2015. An introduction to automated flow cytometry gating tools and their implementation. Frontiers in Immunology 6 (July 2015), 380. https://doi.org/10.3389/fimmu.2015.00380</span>
