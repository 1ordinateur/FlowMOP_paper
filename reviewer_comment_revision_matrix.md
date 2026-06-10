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

## P03b - E1 / R1.5 / R2.20: FlowMOP Versus FlowCut Mechanistic Validation

🔴 🔵 **Comment**

> The manuscript needs a clearer explanation of why FlowMOP performs differently from FlowCut, and whether this reflects an algorithmic difference rather than implementation speed, smoothing speculation, or default-parameter behavior.

🟦 **Status: STAGED - add raw-file time-warp mechanism benchmark and manuscript explanation.**

The revised analysis should include a focused mechanism benchmark that separates source-linked fluorescence/composition structure from acquisition-rate structure. This is important because the existing Bimix and Trimix files already contain source-specific fluorescence differences through `SampleIDInt`, while their Time channel is approximately normalized during synthetic-combo construction. Segment files preserve stronger source/time-density structure. The new benchmark therefore should not add artificial fluorescence perturbations. Instead, it should preserve the raw events, source labels, scatter channels, fluorescence channels, and acquisition order, and then alter only the Time channel to test whether each algorithm responds to acquisition-rate variation itself.

The benchmark should use source-labelled smallcut synthetic-combo FCS files so that event-level truth is preserved throughout the analysis. The paper run should use 15 high-count files: five Bimix, five Trimix, and five Segment inputs. Each benchmark input should contain 500,000 acquisition-order-preserving events. For Bimix and Trimix files, the first 500,000 events are used because the source mixture is already shuffled through the file. For Segment files, the 500,000-event window is centered on the source transition so that both segment sources are represented while preserving contiguous acquisition order. Each file should then produce three matched variants: raw, source-time-warped, and random-time-warped.

The source-time-warped variant should multiply local inter-event Time increments according to `SampleIDInt` while leaving all non-Time channels unchanged. For two-source files, the sorted source IDs use 1.0x and 2.0x local Time multipliers; for three-source files, the sorted source IDs use 1.0x, 1.5x, and 2.0x. The random-time-warped variant should use the same multiplier set, but assign multipliers to contiguous 2,000-event chunks with a fixed random seed independent of source identity. After warping, the total Time range is rescaled back to the raw file's Time range so that the benchmark tests local acquisition-rate structure rather than total run duration. FlowMOP, FlowCut, and PeacoQC should be run on the exact same generated FCS inputs using the same fluorescence-channel exclusions used in the main benchmarking.

The primary endpoint should use the same source-label truth logic as the MAD-smoothing benchmark: filename proportions identify the target source or sources with the largest mixture proportion; sensitivity is retained target-source events divided by retained events; specificity is removed non-target-source events divided by removed events; and balanced score is the mean of sensitivity and specificity. Results should also report removal fractions and deltas relative to the matched raw variant. The expected interpretation is specific and falsifiable: if FlowCut is sensitive to local acquisition density, it should change removal under Time-only warping even though fluorescence values are unchanged; if FlowMOP is more fluorescence/population-summary anchored, it should be less affected by Time-only warping while preserving source-label specificity. PeacoQC provides a complementary comparison because it can react to source-linked peak instability that is real but not necessarily low-quality under the source-label ground truth.

🟣 **Draft manuscript wording**

> To distinguish acquisition-rate sensitivity from source-linked fluorescence and composition structure, we added a matched mechanism benchmark using 15 source-labelled smallcut synthetic-combo FCS files: five Bimix, five Trimix, and five Segment inputs. For each file, 500,000 acquisition-order-preserving events were used without changing fluorescence, scatter, source labels, or event order. Bimix and Trimix files used the first 500,000 events, whereas Segment files used a contiguous 500,000-event window centered on the source transition so that both segment sources were represented. We then generated a raw control, a source-time-warped variant in which local Time increments were multiplied according to `SampleIDInt`, and a random-time-warped variant in which the same multipliers were assigned to 2,000-event chunks independently of source identity. Total Time range was rescaled to match the raw input. Performance was evaluated relative to the matched raw file and by source-label sensitivity, specificity, and balanced score.

🟣 **Draft response-letter wording**

> We agree that the original manuscript did not sufficiently distinguish why FlowMOP and FlowCut behaved differently. We therefore added a mechanism benchmark that keeps the raw synthetic-combo fluorescence and source composition intact while changing only the Time channel. This benchmark is important because acquisition-rate changes can occur without invalidating fluorescence measurements, whereas the Bimix, Trimix, and Segment inputs already contain source-linked fluorescence and composition differences. By comparing each Time-warped file with its matched raw control across FlowMOP, FlowCut, and PeacoQC, the analysis tests whether each method responds to local acquisition-rate structure itself or to source-linked fluorescence/composition structure. This directly addresses the reviewer concern by replacing a speculative smoothing explanation with a focused algorithmic test of FlowMOP's fluorescence/population-summary time-gating behavior versus FlowCut's sensitivity to time-density structure.

## P21 - R2.16: Bayesian Modelling Placement

🟠 🔵 **Comment**

> Bayesian modelling is misplaced and its purpose is unclear.

🟦 **Status: RESPONSE LETTER ONLY - justify Bayesian rank-preference modelling.**

We will address this point in the response letter rather than adding further manuscript text. Bayesian modelling is appropriate here because the expert-assessment outcome is ordinal rank preference, not a continuous measurement with normally distributed residuals or a binary ground-truth label. A rank-preference model directly estimates relative preference probabilities while accounting for uncertainty across raters and samples. Standard tests such as t-tests or ordinary linear regression would treat rank intervals as if they were quantitatively equal, while pairwise tests would fragment the ranked data into multiple comparisons and discard part of the ordering information. The response should clarify that this model summarizes relative expert preference only and is not being used as an objective measure of ground-truth gating quality.

<details>
<summary><strong>Suggested revision options - response-letter rationale selected</strong></summary>

🟣 **Option A: section move**

> <span style="color:#b00020">~~### Bayesian Modelling~~</span>
> <span style="color:#007a3d">### Statistical Analysis Of Expert Preference Rankings</span>

🟣 **Option B: purpose sentence**

> <span style="color:#007a3d">Bayesian modelling was used to summarize relative expert preference rankings across gate providers; it was not used to define objective ground-truth gating quality.</span>

🟣 **Response-letter wording**

> <span style="color:#007a3d">We agree that the purpose of the Bayesian model should be made clear. The expert task generated ordinal rank-preference data, rather than continuous outcomes or objective binary labels. We therefore used Bayesian rank-preference modelling to estimate relative preference probabilities across gate providers while retaining the ordering information and representing uncertainty across raters and samples. Conventional continuous-outcome tests would require treating rank differences as equal-interval measurements, while multiple pairwise tests would discard part of the ranking structure and inflate comparison burden. We have clarified that this model summarizes relative expert preference only and is not used as an objective ground-truth measure of gating quality.</span>

</details>

## P22 - R2.17: Event Removal And 5% Threshold

🟠 🔵 **Comment**

> Report how many events are removed and justify the 5% threshold.

🟣 **Option A: supplementary table**

> <span style="color:#007a3d">Supplementary Table [X] reports the number and percentage of events removed by each preprocessing step for each dataset and method.</span>

🟣 **Option B: threshold rationale**

> <span style="color:#007a3d">The 5% threshold was selected as a conservative default to avoid triggering exclusion on very small event fractions; sensitivity to this threshold is reported in Supplementary Figure/Table [X] where available.</span>

## P24 - R2.19 / R2.23: Figure 1 Labels

🟠 🔵 **Comment**

> Figure 1A/B/C lack axis titles and clear annotations.

🟣 **Option A: caption edit**

> <span style="color:#007a3d">Figure 1 has been revised to include x- and y-axis labels for all schematic plots, explicit labels for the smoothed and unsmoothed time-bin summaries, and a y-axis label for the doublet-ratio histogram.</span>

🟣 **Option B: response-letter wording**

> <span style="color:#007a3d">We revised Figure 1A-C to label all axes and clarify the two time-summary panels following time-binned fluorescence calculation.</span>

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
