# ui/

The triage console (roadmap Section 3.8, Week 14): one screen, one
slider. Built in Streamlit, which the plan allows in place of React.

- A slider for the guaranteed maximum missed-defect rate (epsilon).
- A thumbnail grid of a batch of held-out images that re-partitions live
  into "auto-cleared" and "to a human". Each image is inspected once by the
  service; the slider then re-uses the lambda_hat the service publishes
  per epsilon, so it responds instantly.
- HRF, auto-cleared count, projected review hours per day (images per day
  x HRF x minutes per review), and the ground-truth defects cleared in
  this batch (red border = defect), so a miss is visible.
- The plain-English risk statement as the footer.

## Run

Start the service first (see `service/README.md`), then:

```
streamlit run ui/console.py
```
