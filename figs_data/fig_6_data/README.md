# Tumour Figure 7

`generate_tumour_figure.py` rebuilds Figure 7 and its statistics tables from
`../flowmop_data/tumour_data/Shared_tumour_FlowMOP.zip`.

Install the figure-specific PDF renderer if it is not already available:

```bash
pip install PyMuPDF
python figs_data/fig_6_data/generate_tumour_figure.py
```

The archived FlowJo workspace defines BUV395-A as CD3 and BV510-A as
CD19. T cells are the CD3+CD19- Q1 population, B cells are the
CD3-CD19+ Q3 population, and frequencies use the original event count as
their denominator. Statistical values are normalized within sample to
the matched Raw value before the unadjusted, two-sided paired t-tests.
Figure 7A shows the Manual and FlowMOP Time, Debris, and Doublet preprocessing plots, Figure 7B
shows the T/B gates, and Figure 7C shows the three downstream endpoints.
FlowJo's numerical axes are removed,
and a compact shared pair of directional CD3/CD19 arrows at the lower-left
identifies increasing protein signal across Panel B. The rasterised FlowJo quadrant annotations are
masked and redrawn as larger vector text using percentages calculated from the
workspace population counts. Only statistically significant comparisons are
annotated. Panel C displays Live CD45+ cell count, B-cell frequency, and T-cell
frequency.
