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
