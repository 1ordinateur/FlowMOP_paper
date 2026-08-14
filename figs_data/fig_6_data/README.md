# Tumour Figure 6

`generate_tumour_figure.py` rebuilds Figure 6, Supplementary Figure S8,
and their statistics tables from
`../flowmop_data/tumour_data/Shared_tumour_FlowMOP.zip`.

Install the figure-specific PDF renderer if it is not already available:

```bash
pip install PyMuPDF
python figs_data/fig_5_data/generate_tumour_figure.py
```

The archived FlowJo workspace defines BUV395-A as CD3 and BV510-A as
CD19. T cells are the CD3+CD19- Q1 population, B cells are the
CD3-CD19+ Q3 population, and frequencies use the original event count as
their denominator. Statistical values are normalized within sample to
the matched Raw value before the unadjusted, two-sided paired t-tests.
Figure 6 displays only the T/B gates. FlowJo's numerical axes are removed,
and one shared pair of directional CD3/CD19 arrows identifies increasing
protein signal across Panel A. The rasterised FlowJo quadrant annotations are
masked and redrawn as larger vector text using percentages calculated from the
workspace population counts. Only statistically significant comparisons are
annotated.
