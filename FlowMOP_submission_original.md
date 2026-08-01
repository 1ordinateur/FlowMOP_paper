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

Flow cytometry now generates high-parameter datasets whose scale and variability challenge manual preprocessing, leading to subjectivity and poor reproducibility. This study introduces FlowMOP, a Python-native framework that automates three major preprocessing steps—time-gating, debris removal, and doublet exclusion. Compared to existing time-gating methods, FlowMOP demonstrates both superior sensitivity and specificity. FlowMOP is the first automated approach capable of debris cleaning, and the first method to integrate all three aforementioned steps together into a comprehensive package.

Methodologically, temporal artifacts are identified via parameter-wise peak checks, bin-level median smoothing on the time channel, and robust outlier rejection. Debris is excluded by adaptive FSC-A thresholding derived from cross-parameter peak structure. Finally, doublets are removed using dynamic inflection detection on FSC-A/FSC-H and SSC-A/SSC-H ratio histograms. All operations are chunked and memory-efficient, enabling consistent processing on extremely large files.

Validation employed synthetic controls to permit performance evaluation with objective ground truths (concatenated and mixed time perturbations, high-debris mixtures, and CFSE/CTV co-labeled doublets). Additionally, comparative benchmarking against FlowCut and PeacoQC, and expert human evaluations across diverse biological datasets was conducted. In the synthetic benchmarking, FlowMOP demonstrated higher sensitivity than FlowCut and greater specificity than PeacoQC for segment-type temporal artifacts and was not inferior to expert performance in debris and doublet removal. Though subjective rankings often favored manual gates, Bayesian analyses indicated non-inferiority to at least one expert across all tasks. FlowMOP provides a comprehensive, scalable, and reproducible preprocessing solution that standardizes cytometry data cleanup and strengthens downstream quantitative analyses. FlowMOP can be accessed at https://github.com/1ordinateur/FlowMOP.

## Introduction

The field of flow cytometry faces increasing complexity challenges. Modern acquisition hardware, exemplified by spectral cytometry, coupled with sophisticated analysis software utilizing deep-learning approaches, has resulted in datasets of increasingly large size and dimensionality [2]. This evolution necessitates correspondingly sophisticated preprocessing methodologies to enable satisfactory performance in both scale and quality of sample cleaning. Traditionally, preprocessing has been conducted manually, creating significant challenges where datasets exceed human analytical capacity and where reproducibility and objectivity are increasingly prioritized.

Manual preprocessing of flow cytometry data, whilst still not standardised, typically involves three gating components. I) Time gating: operators perform time-gating to remove events potentially acquired erroneously due to transient or persistent artifacts in the instrument or sample, including air bubbles, blockages, or laser malfunctions. II) Debris gating: operators remove debris, which is generated by both the sample preparation process and inherent in the sample itself. Debris events are generally identified by reference to measured events with low size (determined by FSC-A (Forward SCatter – Area)) and internal complexity (determined by SSC-A (Side SCatter – Area)) [1]. III) Doublet gating: Events classified as "doublets"—two or more cells erroneously detected as one—are removed as their fluorescence measurements lack reliability. This is often done with reference to the ratio of the signal duration, and signal intensity. Doublets feature a signal peak strength comparable to a single cell, but for twice the duration. Hence, analysts may filter these events out by comparing an event’s total FSC Area relative to its FSC height (peak signal intensity) or by comparing the FSC signal height against the FSC signal width [4]. These manual preprocessing steps are time-consuming, inherently subjective, and susceptible to inconsistency across operators.

This paper proposes a novel cleaning algorithm, FlowMOP, that seeks to address these shortcomings. It uses a Python-based method featuring time-gating, debris-gating, and doublet-gating capabilities. FlowMOP is the first algorithmic approach capable of debris removal, and the first algorithm to combine the three aforementioned cleaning steps into an integrated package.

The Python implementation specifically facilitates integration with contemporary machine-learning workflows, which predominantly utilize Python-based frameworks. Furthermore, owing to its Dask-based framework, FlowMOP enables streamlined integration in distributed compute operations than current algorithms.  This Dask-based design, which uses a lightweight Python framework for parallel and distributed computation, allows FlowMOP to run efficiently across distributed compute resources.

To date, cytometry preprocessing algorithms have been evaluated either by comparison to human-defined gold standards (as in FlowCut and PeacoQC) or through mathematical metrics (consider cyCombine’s approach to batch correction [12]). Conversely, this paper seeks to compare performance not only through traditional human-expert defined standards, but also the development of bespoke data generated explicitly for pre-processing validation. This will allow for the true ‘cleaning’ ability of both algorithms and humans to be objectively compared to a ground truth. Accordingly, the proposed methodology facilitates the detection of algorithms that potentially outperform manual approaches—a capability current methods do not adequately provide.

## Methods

For further details concerning datasets, see Supplementary Table 1A.

### Preparation protocols for synthetic samples

#### Synthetic Time Sample Preparation and Generation

#### Preparation of Human PBMC Samples for Synthetic Time Benchmarking Samples

PBMC samples were collected from healthy donors under ethics protocols ACT Health 2019.ETH.00081 and ANU HREC 2020/047. Blood was collected into ACD-A tubes and PBMCs isolated using Lymphoprep density gradient (Stemcell Technologies) and SepMate tubes (Stemcell Technologies) per manufacturer’s instructions prior to cryopreservation. Thawed PBMCs were stained with ViaDye Red (Cytek Biosciences). The staining cocktail contained antibodies to several highly abundant antigens at less-than-saturating concentrations, and either 0.5, 1 or 3 x 10^6 cells were stained, yielding samples with different levels of fluorophore signal intensity for these common antigens. Other antibodies within the panel were present at saturating concentrations to yield similar fluorescence intensity across different cell seeding densities. Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer. . Samples were acquired on a Cytek Northern Lights 3-laser (V/B/R) spectral flow cytometer.

#### Generation of Synthetic Time Samples

Samples were synthetically combined across differing cell concentrations in three manners. One in a simple ‘Segmented’ fashion, where events from one or more samples were simply appended onto events from an existing sample. The second, ‘Bimix’ manner, where events from two differently stained samples were randomly synthetically combined in random proportions (e.g. 40:60, 75:25 etc), with one run containing a mixing bin size of 5000 events and the other of 2000 events. The final, ‘Trimix’ contained randomly combined events from three differently stained samples in mixing bin sizes of 5000, and 2000 events.

#### Mice

For generation of synthetic samples for FlowMOP validation (debris and doublet removal), splenocytes from 13-16-week old C57Bl/6N or C57Bl/6J mice were used. All animal experimentation was performed under ethics protocol 2024/379 at the Australian Phenomics Facility at the Australian National University, Canberra.

#### Synthetic Debris Sample Preparation and Generation

Mouse spleens were mechanically dissociated prior to lysis of red blood cells (RBC) with RBC lysis buffer (MilliQ H2O containing 150 mM NH4Cl, 10 mM KHCO3 and 1 mM EDTA). Splenocytes were then incubated with Fc block (BD), then stained with an antibody panel to delineate several major immune cell subsets (CD19 APC, CD3 PE, CD8 PE-Cy7, CD11b BB515, CD4 BV605, Fixable Viability Dye efluor780). To generate ‘high debris’ samples, cells were resuspended in MilliQ water and incubated for 2 minutes, prior to addition of 10X PBS to restore osmolarity. ‘Low debris’ samples were kept in isotonic solution throughout. Samples were acquired using a BD LSRII cytometer.

For assessment of FlowMOP gating performance, ‘high debris’ and ‘low debris’ samples were synthetically combined.

#### Synthetic Doublet Sample Preparation and Generation

To generate samples with high proportions of doublets, mouse spleens were digested through injection with digestion buffer comprising RPMI (Gibco) supplemented with collagenase P (Roche) and DNAse I (Roche). After incubation and mechanical dissociation, red blood cells (RBCs) were lysed with RBC lysis buffer. Cells were then divided and stained with either CellTrace Violet (CTV) (Invitrogen) or Carboxyfluorescein succinimidyl ester (CFSE) (eBioscience). Following this, cells were recombined and incubated for 30 minutes at 37oC 5% CO2 to encourage further doublet formation, then stained with Fixable Viability Dye eFluor780 (Invitrogen). Samples were acquired on a BD LSRII cytometer.

### Non-synthetic samples

### Bayesian Modelling

Rankings were modelled using a Plackett–Luce model with latent method abilities; identifiability was enforced by fixing a reference ability to zero. Independent Normal priors were placed on non‑reference abilities. Posterior inference was performed with an affine‑invariant ensemble Markov Chain Monte Carlo sampler (emcee; 32 walkers, 5,000 iterations; 1,000 burn‑in), and posterior medians with 95% credible intervals were reported. Directional hypotheses (superiority/inferiority) were evaluated by computing P = Pr(H1 | data) and converting to a Bayes factor BF₁₀ = p/(1 – p) under equal prior odds, interpreted using Jeffreys’ scale and reported as BF, P (Bayes factor, Posterior probability of alternative hypothesis) (see Supp. Table 2).

For each Plackett–Luce fit, the ensemble sampler’s mean acceptance fraction and an integrated‑autocorrelation‑time–based effective sample size was monitored (both as implemented in emcee) to assess MCMC convergence. Across all analyses, mean acceptance fractions ranged from 0.520 – 0.60, effective sample sizes ranging from 134042 to 128003.

Posterior predictive checks were carried out by comparing observed pairwise win counts with those implied by posterior draws under a Bradley–Terry formulation, using a chi‑square‑type discrepancy averaged over method pairs. The resulting average χ² per comparison with a maximum of 0.56, indicating good agreement between the model and the observed rankings.

Computations were implemented in Python with numerically stable log‑likelihoods.

### Coding Assistance

Parts of the code were generated with the aid of ChatGPT Codex and Claude Code. All code generated by LLMs were manually verified before implementation.

## RESULTS

![Embedded image 1](FlowMOP_submission_media/image1.png)

Figure 1. Schematics depicting the methodology behind FlowMOP’s time-gating (A), debris-gating (B), and doublet gating methodologies (C). A) Depicts FlowMOP’s time-gating algorithm which selects valid parameters using positive-peak detection. Using these parameters, FlowMOP generates time-binned fluorescence medians, by selecting that time bin’s median positive peak value, or overall geometric mean. These are compared against each other using the smoothed and unsmoothed means.  B) Depicts FlowMOP’s debris-removal logic. It undertakes the same valid-parameters check as in A), and then selects the appropriate threshold based on that parameter’s fluorescence peaks. The median threshold is selected as the FSC-A gate. C) Depicts FlowMOP’s doublet-gating strategy. Doublets are either removed by the FSC-A/H, and/or SSC-A/H histogram inflection points, or by a fixed ratio.

### Algorithmic design

An overview of FlowMOP’s architecture is contained in Fig. 1, detailing approaches for its preprocessing, time-gating, debris removal, and doublet removal methods. This cleaned data can be applied to downstream analysis. FlowMOP ingests flow cytometry data stored in any file format, including .csv, .fcs, and parquet formats. The algorithm leverages Dask arrays for distributed computing, enabling efficient processing of large-scale datasets across multiple cores or compute nodes [13]. This architecture allows FlowMOP to scale seamlessly from single samples to cohort-level analyses without memory constraints.

All gating operations maintain compatibility with Dask’s chunked arrays [5], ensuring memory-efficient processing regardless of dataset size, with multiple workers able to simultaneously process a file.

#### Precleaning

FlowMOP first checks the input file for events at the limit of detection, defined as the maximum fluorescence value for that sample. If the number of events that equal this maximum fluorescence exceeds a threshold (default 5%), then FlowMOP automatically removes these events. Otherwise, it retains all the values.

#### Time Gating

To time gate, FlowMOP builds upon the assumptions posited in PeacoQC and FlowCut regarding fluorescence fluctuations. That is, independent of flow rate variations, sections of acquired sample with aberrant positive fluorescence averages are the target portions to be removed. To achieve this, FlowMOP checks each parameter, excluding parameters with a unimodal distribution. ‘Unimodal distribution’ is presently defined as parameters with only one identifiable peak. Subsequently, for each fluorescence parameter that satisfies this criterion, FlowMOP excludes the first peak (selecting all subsequent peaks) and measures the average fluorescence value for each time bin. FlowMOP then can operate either in the ‘Positives’ mode, or ‘Geomean’ mode. In ‘Positives’ mode, all events before the first inflection point are discarded. All results shown presently operate in ‘Positives’ mode. In the ‘Geomean’ mode, all events are considered. Subsequently, on a per-parameter basis, the sample is transformed into bins grouped by time (the default being bin having minimum of 150 events, up to a maximum of 500 bins).

The median fluorescence of each bin’s cells is returned. Two smoothing values, one small, one large (default 0.1, 1.0) are applied to the returned time bins. The smoothed events are then filtered using the median absolute deviation (MAD) from all time-bins. Bins falling outside that MAD threshold in either the smoothed or unsmoothed bins are flagged for removal. Time-bins across all parameters are then combined, with time bins rejected if they have been flagged in any parameter. However, when >10 parameters are present, bins are rejected if they are flagged by two or more parameters (Figure 1A).

Most substantially, FlowMOP differs from the existing methods by choosing to measure how each time bin’s positive events differ, rather than re-calculating per time bin the ‘positive’ peak like in PeacoQC. Conversely, it also does not eschew positive peak detection completely (unless in Geomean mode) unlike FlowCut. Finally, FlowMOP also novelly employs the use of smoothing at multiple resolutions to ensure multiple types of abherrations are detected, and parameter voting to ensure that higher dimensional sample sets are not overly degraded.

#### Debris Gating

To debris gate, FlowMOP applies an FSC-A based threshold to exclude debris. FlowMOP’s debris exclusion conducts a similar unimodality check on each fluorescence parameter, and the first peak is then excluded as the Time-gating feature. Thereafter, FlowMOP detects the global FSC-A peak as a reference point. For every parameter’s positive events, FlowMOP checks first for an FSC-A peak similar to the reference peak (default 30% of the reference peak’s value). If there is such a peak, it checks if the second FSC-A peak is the global maximum FSC-A peak. If that parameter’s positive cell’s second FSC-A is the global maxima, FlowMOP returns the FSC-A threshold as the minima between those two FSC-A peaks. If the second FSC-A is not the maximum, it returns the global interpeak minimum between the reference peak and maximal peak. Conversely, if there is no reference peak present in that parameter’s positive population, it selects the left-boundary of that parameter’s first peak. The median FSC-A threshold across all parameters is taken as the final FSC-A gate to be applied to the sample (Figure 1B). To date, this is the first algorithmic approach targeted at the removal of debris.

#### Doublet Gating

To doublet gate, FlowMOP dynamically excludes sample doublets. To do this, FlowMOP creates a histogram of the FSC-A/FSC-H ratio. If there are multiple peaks all with a ratio of 1 or more, then it chooses the inflection point between those peaks, and excludes all events larger than that value. If there are insufficient peaks, it simply returns all events that have an FSC-A/FSC-H ratio smaller than a threshold (default 5). The process is repeated for the 
SSC-A/SSC-H variable. Consequently, FlowMOP is able to distill the implicit ratiometric information that current density based methodologies may overlook.

### Algorithmic Validation

#### Synthetic Sample Benchmarking

The ability of FlowMOP to successfully detect time, debris, and doublet-perturbed data was first tested against the respective task’s synthetic datasets, namely the synthetically combined staining time samples, the high-debris + low debris samples, and the CTV / CFSE doublet samples.

For time gating samples, sensitivity and specificity were reported for each of the benchmarked methods. Sensitivity is defined as proportion of truly contaminated events that were removed. Specificity is defined as the proportion of events removed that were from the sample(s) that needed to be removed.

![Embedded image 2](FlowMOP_submission_media/image2.png)

Figure 2. A) Representative flow cytometry plots to showing CD3 fluorescence against time. The original synthetically generated sample is shown in column 1, with the resulting output following FlowMOP, PeacoQC, and FlowCut processing shown in the subsequent columns. The first row depicts a representative ‘segmentation’ based synthetic file. The second row shows a representative two-sample mixture, 5000-event bin sample. Frequency percentages shown are the percentage of cells left post-cleaning relative to the original synthetic sample (rounded to the nearest percentage point). B) Bar plots showing algorithmic performance, grouped by sample type (Segmented, Bimix, and Trimix), and algorithm (FlowMOP, PeacoQC, and FlowCut). The first column depicts algorithmic performance sensitivity ± SD, with points representing individual samples. The second column depicts specificity ± SD. The first and second rows representing a 5000 and 2000 event mixing bin size respectively (n=87, 92). All comparisons were performed using a Bonferroni-adjusted paired t-test.

#### Synthetic Time Gating Benchmark

Several computational approaches have been proposed to address specific preprocessing challenges, primarily focusing on time-gating. Notable examples include flowAI [11], PeacoQC [6], flowClean [7], and FlowCut [10]. These algorithms are differentiated from FlowMOP by their focus on time-gating and R implementation. FlowMOP also uniquely offers debris cleaning capabilities and doublet removal (with the exception of PeacoQC). Here, FlowMOP’s time gating performance is evaluated against PeacoQC and FlowCut, the two most recent and best rated methods.

The synthetically generated Segment, Bimix, and Trimix samples sought to emulate the fluorescence fluctuations that time-gating should remove, whilst allowing for an objective performance metric as each event’s origin is known (Fig. 2A). For samples with two or more equally weighted samples, algorithms were benchmarked on their ability to retain both considering that neither could be considered ‘superior’ in quality. Algorithmic performance against humans was considered not appropriate given the inability for humans to accurately remove multiple <1000 event bins from a sample in the mixed category. Results are reported with sensitivity and specificity.

In the Segmented method, FlowMOP and PeacoQC had significantly higher sensitivity than FlowCut (p < 0.001) (Fig. 2B). FlowMOP also had higher specificity than PeacoQC (p < 0.001). In the Bimix method, for the 5000-bin size, PeacoQC showed superior sensitivity to both FlowMOP and FlowCut (p < 0.001). FlowMOP demonstrated significantly higher specificity than FlowCut (p = 0.02). In 2000-bin size, PeacoQC showed inferior sensitivity to FlowCut (p = 0.002), and far worse specificity compared to both FlowMOP and FlowCut (p < 0.001). In the Trimix method, for the 5000 bin samples, FlowMOP exhibited higher specificity than both other methods (p = 0.004 PeacoQC, p = 0.003 FlowCut). Similarly, in the 2000 bin samples, PeacoQC was inferior in specificity to both FlowMOP and FlowCut (p = 0.01, p = 0.03).

Representative plots for the 2000 bin Bimix method, and the 2000, and 5000 bin Trimix methods can be found in Supp. 1.

#### Synthetic Debris Gating Benchmark

![Embedded image 3](FlowMOP_submission_media/image3.png)

Figure 3. A) Representative flow cytometry plots showing FSC-A/SSC-A debris plots. The first plot represents a representative combined high + low debris sample, the second and third represent the high debris portion and the low debris portion of that sample respectively. The percentages below denote the proportion of the combined sample represented by the annotated plot. B) FSC-A/SSC-A flow plots of the Fig. 3A’s representative sample post-processing. Again, the first column shows the combined high + low debris sample, the second and third show the high debris and low debris samples separately respectively. The percentages in the first column denote the proportion of the sample remaining relative to the original combined sample. Percentages in the  high debris and low debris columns denote the post-filtering proportion that each comprises (rounded to the nearest percentage). C) Bar plot representing mean proportion percentage ± SD of low debris sample remaining post-processing for FlowMOP and four human experts. FlowMOP has significantly higher low debris sample proportion than Expert 4 (Bonferroni adjusted paired t-test, p = 0.04). D) Bar plots showing mean low debris sample proportion percentage ± SD following human expert gating using either a group or per-sample based strategy (Blue bars denote group-wise results, green sample-wise). No difference was found between the two methods (un-adjusted paired t-test, p > 0.05).

FlowMOP’s debris gating performance was tested by its ability to deplete the high debris component in the combined high + low debris samples (Fig. 3A). FlowMOP was able to consistently reduce the proportion of high debris sample in the high + low debris sample from 50 ± 1.10% content to 28.8 ± 0.1% (paired t-test, p < 0.001), even whilst only using the FSC-A parameter to define the debris population (Fig. 3B). FlowMOP did not differ in performance to any human evaluators save for Expert 4 where FlowMOP performed better (Bonferroni adjusted paired t-test, p = 0.04) (Fig. 3C).

FlowMOP by default conducts debris removal on a per-sample basis. For comparison with the standard per-group gating methodology, human benchmarkers were also requested to apply per-sample and per-group gating strategies for comparison. No difference was found between per-sample and group-based gating for any expert (Fig. 3D, unadjusted paired t-Test p > 0.05).

#### Synthetic Doublet Gating Benchmark

FlowMOP’s doublet removal performance was also examined through synthetic technical controls (Fig. 4A). Events that were positive for both CFSE and CTV could then be confidently labelled as doublets, as no actual double positive events can be present (save for very rare dye-transfer events). FlowMOP was then run on these samples, and the proportion of remaining CFSE/CTV double positive cells was compared against that of human experts (Fig. 4A).

![Embedded image 4](FlowMOP_submission_media/image4.png)

Figure 4. A) Representative flow cytometry plots of synthetic doublet samples and gating. The first row shows all samples by CTV against CFSE, the second FSC-A/FSC-H, and the last by SSC-H/SSC-A. The columns show representative CTV-only stained, CFSE-only stained, mixed CTV-CFSE, human expert doublet-removed, and FlowMOP doublet removed samples. B) Bar graph showing the mean percentage ± SD of CTV-CFSE double positive events removed (relative to the original sample) following FlowMOP or human expert processing. C) Bar graphs showing mean frequency ± SD of remaining CTV-CFSE double positive events following human expert processing, comparing group and per-sample based gating strategies (Blue bars denote group-wise results, green sample-wise). Only Expert 3 had significantly different results (un-adjusted paired t-test, p = 0.001, all others p > 0.05).

FlowMOP was able to significantly decrease the number of CTV-CFSE double-expressing events from an initial frequency of 7.84 ± 1.21% to 0.27 ± 0.11% (paired t-test; p = 0.001), successfully performing at the same standard as all experts (Fig. 4B, paired t-test; unadjusted p > 0.05). Again, to ensure that FlowMOP’s sample-wise doublet gating was not systemically biasing the results, human experts were asked to also perform sample wise and groupwise doublet gating. No statistical difference was detected between the sample-wise and groupwise methods, save for Expert 3 (Fig. 4C, paired t-test unadjusted p values). Expert 3 consistently removed fewer doublets in the sample-wise method than in the group method (Fig. 4C, paired t-test; p = 0.009).

### Human Subjective Rankings

To examine FlowMOP’s utility in real-world samples, its performance was evaluated across different flow cytometry files. FlowMOP’s outputs in these samples were benchmarked against a set of human experts, and the resulting time, debris, and doublet gates were ranked by that sample type’s expert. In the time-gating task, FlowCut and PeacoQC were also included for comparison. Rankings were provided by a relevant expert in that sample type. Lower rankings indicate lower preference.

![Embedded image 5](FlowMOP_submission_media/image5.png)

Figure 5. A) Table showing ranking preferences of each cleanup method for each dataset for human experts 1-4, FlowMOP (our method, black border), FlowCut, and PeacoQC (Abbreviations: DRG – Dorsal Root Ganglion, CNS – Central Nervous System). Methods were ranked 1-7 for 9 different datasets by an expert in that respective data type. Lower ranking is worse.  B) Average ranking score across all 9 methods.

In the biological datasets, FlowMOP ranked the best amongst the algorithmic time gating approaches in the time gate (Fig. 5A, 5B). In the mouse brain and mouse bone marrow tasks, it was ranked sixth and fifth respectively (Fig. 5B). On a Bayesian analysis, FlowMOP was observed to be substantially preferred to FlowCut (BF = 5.39, P = 84.3%) and strongly preferred to PeacoQC (BF = 12.10, P = 92.4%). FlowMOP was ranked inferiorly to all human experts with strong to decisive evidence (Expert 2 BF = 10.55, P = 91.3%, all others BF > 100, P = 100%).

![Embedded image 6](FlowMOP_submission_media/image6.png)

Figure 6. A) Table showing ranking preferences for human experts 1-4, and FlowMOP, when ranked 1-5 for 9 different datasets by an expert in debris removal. N/A denotes where samples for that human expert are not available. Lower ranking is worse. B) Bar graphs showing the average ranking score for each method across all datasets.

![Embedded image 7](FlowMOP_submission_media/image7.png)

Figure 7. A) Table showing ranking preferences for human experts 1-4, and FlowMOP, when ranked 1-5 for 9 different datasets by an expert in doublet removal. NA denotes where samples for that human expert are not available. Lower ranking is worse. B) Bar graphs showing the average ranking score for each method across all datasets. Ranking score is determined by the inverse rank, out of five.

FlowMOP scored the poorest overall when ranked against the human benchmarkers in debris, and in doublets (Fig. 6B, 7B). On a Bayesian analysis, substantial to strong evidence was observed for FlowMOP being inferior to Expert 1 (BF = 5.87, P = 85.5%), Expert 4 (BF = 12.85, P = 92.8%), and Experts 2,3 (BF > 100, P = 100%) in debris removal. In doublet removal, FlowMOP was weakly inferiorly ranked to Expert 4 (BF = 3.14, P = 75.8%), substantially inferiorly ranked to Expert 1 (BF = 5.67, P = 85.0%), strongly inferiorly ranked to Expert 2 (BF = 14.38, P = 93.5%), and decisively inferiorly ranked to Expert 3 (BF > 100, P = 100%).

However, it is of note that in 4/5 and 7/9 datasets in the debris and doublets tasks respectively, FlowMOP was not the least preferred, indicating superiority to at least one human benchmarker (Fig. 6A, 7A). Indeed, FlowMOP’s good performance occurred across a wide range of datasets. For debris, it rated first in the mouse blood task, and third in human liver and mouse skin datasets. In the doublets task, FlowMOP scored second in the human liver task again. For full tabular rankings, see supplementary data Tables 1B-E.

## Discussion

FlowMOP demonstrates an ability to conduct time, debris, and doublet gating in a wide variety of sample types, including in difficult synthetic datasets.

### Synthetic Benchmarking

#### Synthetic Sample Time Gating

In the synthetic time-gating analysis, the Segment time gate is perhaps the most common and consequential time-artifact, as a non-negligible sample portion is often required to be removed in real samples. This type of synthetic data expects gating most similar to current manual gating, where blocks of events are removed. The objective of these samples is to simulate where there is a long blockage or sudden shift in the acquired sample. Here, FlowMOP demonstrated overall a greater sensitivity than FlowCut, and with better specificity than PeacoQC. It is hypothesised that this sensitivity boost relative to FlowCut is resultant from FlowCut’s binning strategy which applies no smoothing over sample bins, possibly hindering the ability to detect systemic sample shifts.

The Bimix synthetic samples seek to emulate transient microblockages that self-resolve quickly. The size of the simulated microblockage is determined by the ‘bin size’, where the larger bin sizes emulate larger blockages. As expected, algorithm sensitivity and specificity reduced in the smaller bin sized samples. This degradation was particularly marked for PeacoQC’s sensitivity. Therefore, despite PeacoQC demonstrating superior sensitivity in the larger bin size to both algorithms, in the smaller bin-size runs PeacoQC proved to be inferior. FlowMOP at large Bimix bin sizes more specifically removed events than FlowCut. This difference may be attributed to FlowMOP’s ‘smoothing’ implementation. Again, FlowMOP employs both approaches, with a ‘smoothed’ stream to detect trends like in PeacoQC, and the ‘non-pooled’ method akin to FlowCut. This allows FlowMOP to detect both transient and systemic shifts without them interfering with one another. The benefits of this approach can be seen in the Trimix results, where FlowMOP demonstrated superior specificity compared to both other algorithms in the large bin size, and to PeacoQC in the smaller Trimix samples. This advantage is illustrated in the combined approach’s ability to correctly identify aberrant samples in even very ‘noisy’ sample sets.

FlowMOP is also more naturally suited to Dask-style lazy computation and vectorised execution than PeacoQC or FlowCut. This is because FlowMOP reduces event-level cytometry data into regular time-bin summaries early in the pipeline. These summaries have a consistent bin-by-channel structure, allowing the same operations to be applied efficiently across many bins, channels, and samples without needing each step to be resolved immediately.

By contrast, PeacoQC and FlowCut rely more heavily on intermediate decisions that are specific to each bin, channel, or sample. PeacoQC must identify and track peaks over time, while FlowCut calculates broader segment-level statistics and applies additional decision steps. These approaches can still be parallelised in parts, but they contain more points where intermediate results must be gathered, compared, or reinterpreted before the next stage can proceed.

Consequently, FlowMOP is more vectorisable and more compatible with Dask’s lazy, chunked execution model at an algorithmic level. This makes it particularly well suited to large event counts, high-dimensional panels, and distributed cytometry preprocessing.

It is of note that there is a large variation in algorithmic performance across the dataset. One source of this variation is that the 0.5 and 1.0 relative cell concentrations oftentimes exhibited marginal differences in fluorescence intensity (Supp. Fig. 2), especially relative to the 0.5 / 3.0 cell concentrations comparison. Consequently, the 0.5/1.0 discrimination tasks can be considered especially difficult benchmarks to overcome. However, this difficulty was intentionally placed, to ensure the present benchmarking dataset could also show progressive improvement of future time-gating algorithms.

Finally, it is germane to note that human expert gating on the synthetic dataset was considered but deemed inappropriate. This is due to humans being, for all practical purposes, incapable of gating the Bimix or Trimix in a manner that would yield a meaningful contaminated sample reduction. Given the above approaches provide an objective ‘gold standard’ to target, the addition of human benchmarkers would only highlight the superiority of algorithm mediated time-gating, and no further indication of algorithmic performance.

#### Synthetic Sample Debris and Doublet Gating

FlowMOP performed equally or significantly better in the synthetic debris and doublet gating trials (Fig. 3B, 4B) when compared to human expert gaters. In the debris instance of superiority, FlowMOP enriched the data cleanliness by 9.67%, which represented ~19% more debris removed considering the original 50:50 debris / real sample mixture. Furthermore, FlowMOP’s debris performance is further underlined when considering the two debris populations (Fig. 3A) present. The first at <10,000 FSC-A units, and the second at ~20,000 FSC-A units. All human benchmarkers were instructed that this second debris population was in fact debris, and to gate accordingly. FlowMOP was able to independently detect this second debris population independently and exclude it without any external information.

Similarly, in the doublet removal, the synthetic samples, owing to the rather unique preparation, yielded triplets. FlowMOP was able to handle this unexpected population and successfully removed it.

When interpreting the synthetic debris and doublet samples, it is prudent to note that it is the relative enrichment of the non-lysed sample which provides the ground-truth label. This is because the standard-lysed samples actually contain debris too, albeit in a smaller proportion than the lysed sample. Similarly, some debris components may be quite large. Consequently, the resulting proportion of debris, at least when defined solely with an FSC-A / SSC-A gate must contain some contamination. Future techniques may look to define debris on a per-event methodology based on that event’s entire fluorescence profile, rather than a traditional ‘gate’ that sets universal thresholds.

### Biological datasets: Human Ranking Data

For completeness, nine datasets were selected for human evaluation and ranking. These datasets were chosen to be a representative sample of human and mouse-based flow cytometry research, whilst also including samples that were intentionally difficult to gate. FlowMOP was preferred the least relative to the other human experts, however this is not unexpected.

There are several limitations of subjective human rankings for the evaluation of algorithm performance. The bias against algorithmic methodologies when compared to manual approaches is well documented, particularly where objective ground truths are difficult to ascertain such as in flow cleaning [3]. Due to FlowMOP’s unique approaches to gating (consider the difference in style in debris gating (Fig. 4A), or micro-time gates present (Fig. 2A)), FlowMOP samples were readily apparent to benchmarkers. Furthermore, reproducibility and the ‘correctness’ of evaluations was a recurrent problem. Benchmarkers would, more than on one occasion, rate themselves poorly in the rankings and deem their own gates as unacceptable. Together, these findings should condition any conclusion drawn from the rankings, especially in light of the synthetic sample benchmark results which include an objective ground truth.

The subjective human rankings do though indicate that FlowMOP is an acceptable substitute for at scale automated debris and doublet gating. This is demonstrated with a < 85% posterior probability of FlowMOP’s rating inferiority with at least one human expert in both debris and doublet gating approaches, with neither of these experts ranking poorly in the relevant synthetic debris / doublet tasks. This behaviour was not isolated to one single dataset either, with FlowMOP obtaining last in both debris and doublet gating in only the human T cell differentiation dataset. Consequently, whilst FlowMOP is, per human evaluation, inferiorly preferred in general, it represents an acceptable substitute for scalable pre-processing. This fills the capability gap present in many cytometry based machine learning methods [8, 9, 14], and may increase model performance.

### Other remarks

Automated Live/Dead classification of events was considered, however not implemented in the algorithm. There exist many varied protocols and methods for discriminating live/dead samples, along with great diversity in the determination of what constitutes a ‘dead’ event. Consequently, the difficulty of creating a universal live/dead discriminator is non-trivial. Finally, there may be potential significant biological insight in the ‘dead’ cells of a sample, whereby important information concerning a sample may be found in the dead events or their proportion.

## Conclusion

FlowMOP addresses existing gaps in automated flow cytometry preprocessing by providing comprehensive time-gating, debris removal, and doublet removal capabilities in a Python implementation suitable for distributed computation. This facilitates integration with modern machine learning workflows while maintaining or exceeding performance of existing R-based alternatives.

Validation using synthetic datasets with objective ground truth demonstrated robust performance across diverse scenarios. FlowMOP showed superior specificity to PeacoQC and better sensitivity than FlowCut for segment-type anomalies, while maintaining consistent performance across varying bin sizes. For debris and doublet removal, FlowMOP matched or exceeded human expert performance when evaluated against ground truth, successfully identifying complex populations including unexpected triplet events.

Although human rankings on biological datasets favoured manual gating, synthetic benchmarking results provide stronger evidence of algorithmic capabilities, particularly given observed inconsistencies in human evaluations and the synthetic dataset’s ground-truthing. The open-source Python implementation enables reproducible and scalable analysis essential for increasingly complex cytometry datasets.

## Data and Code Availability

FlowMOP can be accessed via https://github.com/1ordinateur/FlowMOP. The code associated with the creation of this paper can be accessed at https://github.com/1ordinateur/FlowMOP_paper. The FCS Files used for this paper can be accessed at http://doi.org/10.5281/zenodo.17896445.

## Supplementary data

Figure S1: Representative flow cytometry CD3 / Time plots for Bimix 2000 bin, Trimix 5000 bin, and Trimix 2000 bin synthetic datasets, with original data inputs, and following cleaning by FlowMOP, FlowCut, and PeacoQC.  Percentages below each figure represent the retained proportion of cells relative to the original representative synthetic sample.

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

**Table S1C: Time gating rankings**

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

**Table S1D: Debris gating rankings**

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

**Table S1E: Doublet gating rankings**

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

Figure S2:

Fluorescence variation as a function of cell concentration.

![Embedded image 9](FlowMOP_submission_media/image9.jpeg)

## References

[1]	Aysun Adan, Günel Alizada, Yağmur Kiraz, Yusuf Baran, and Ayten Nalbant. 2017. Flow cytometry: basic principles and applications. Critical Reviews in Biotechnology 37, 2 (February 2017), 163–176. https://doi.org/10.3109/07388551.2015.1128876

[2]	Thomas Myles Ashhurst, Felix Marsh-Wakefield, Givanna Haryono Putri, Alanna Gabrielle Spiteri, Diana Shinko, Mark Norman Read, Adrian Lloyd Smith, and Nicholas Jonathan Cole King. 2022. Integration, exploration, and analysis of high-dimensional single-cell cytometry data using Spectre. Cytometry Part A 101, 3 (2022), 237–253. https://doi.org/10.1002/cyto.a.24350

[3]	Noah Castelo, Maarten W. Bos, and Donald R. Lehmann. 2019. Task-Dependent Algorithm Aversion. Journal of Marketing Research 56, 5 (October 2019), 809–825. https://doi.org/10.1177/0022243719851788

[4]	Antonio Cosma. 2020. The Nightmare of a Single Cell: Being a Doublet. Cytometry A 97, 8 (August 2020), 768–771. https://doi.org/10.1002/cyto.a.23929

[5]	Dask Development Team. 2016. Dask: Library for dynamic task scheduling.

[6]	Annelies Emmaneel, Katrien Quintelier, Dorine Sichien, Paulina Rybakowska, Concepción Marañón, Marta E. Alarcón-Riquelme, Gert Van Isterdael, Sofie Van Gassen, and Yvan Saeys. 2022. PeacoQC: Peak-based selection of high quality cytometry data. Cytometry Part A 101, 4 (2022), 325–338. https://doi.org/10.1002/cyto.a.24501

[7]	Kipper Fletez-Brant, Josef Špidlen, Ryan R. Brinkman, Mario Roederer, and Pratip K. Chattopadhyay. 2016. flowClean: Automated identification and removal of fluorescence anomalies in flow cytometry data. Cytometry Part A 89, 5 (2016), 461–471. https://doi.org/10.1002/cyto.a.22837

[8]	Zicheng Hu, Alice Tang, Jaiveer Singh, Sanchita Bhattacharya, and Atul J. Butte. 2020. A robust and interpretable end-to-end deep learning model for cytometry data. Proceedings of the National Academy of Sciences 117, 35 (September 2020), 21373–21380. https://doi.org/10.1073/pnas.2003026117

[9]	Nanditha Mallesh. 2023. Automated analysis of flow cytometry using deep learning for the detection of B-cell neoplasms. Thesis. Universitäts- und Landesbibliothek Bonn. Retrieved August 7, 2023 from https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/10949

[10]	Justin Meskas, Daniel Yokosawa, Sherrie Wang, Gabriela C. Segat, and Ryan Remy Brinkman. 2023. FlowCut: An R package for automated removal of outlier events and flagging of files based on time versus fluorescence analysis. Cytometry Part A 103, 1 (2023), 71–81. https://doi.org/10.1002/cyto.a.24670

[11]	Gianni Monaco, Hao Chen, Michael Poidinger, Jinmiao Chen, João Pedro de Magalhães, and Anis Larbi. 2016. flowAI: automatic and interactive anomaly discerning tools for flow cytometry data. Bioinformatics 32, 16 (August 2016), 2473–2480. https://doi.org/10.1093/bioinformatics/btw191

[12]	Christina Bligaard Pedersen, Søren Helweg Dam, Mike Bogetofte Barnkob, Michael D. Leipold, Noelia Purroy, Laura Z. Rassenti, Thomas J. Kipps, Jennifer Nguyen, James Arthur Lederer, Satyen Harish Gohil, Catherine J. Wu, and Lars Rønn Olsen. 2022. cyCombine allows for robust integration of single-cell cytometry datasets within and across technologies. Nat Commun 13, 1 (March 2022), 1698. https://doi.org/10.1038/s41467-022-29383-5

[13]	Matthew Rocklin. 2015. Dask: Parallel Computation with Blocked algorithms and Task Scheduling. 2015. Austin, Texas, 126–132. https://doi.org/10.25080/Majora-7b98e3ed-013

[14]	Lisa Weijler, Florian Kowarsch, Michael Reiter, Pedro Hermosilla, Margarita Maurer-Granofszky, and Michael Dworzak. 2024. FATE: Feature-Agnostic Transformer-Based Encoder for Learning Generalized Embedding Spaces in Flow Cytometry Data. 2024. 7956–7964. Retrieved May 3, 2024 from https://openaccess.thecvf.com/content/WACV2024/html/Weijler_FATE_Feature-Agnostic_Transformer-Based_Encoder_for_Learning_Generalized_Embedding_Spaces_in_WACV_2024_paper.html
