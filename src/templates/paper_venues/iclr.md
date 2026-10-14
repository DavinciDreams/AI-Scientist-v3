# ICLR 2026 Venue Brief

Use this brief to tailor the manuscript for the International Conference on Learning
Representations 2026 submission.

## Hard Constraints
- Target venue: ICLR 2026.
- Submission type: conference paper.
- Body-page budget: {{PAPER_PAGE_BUDGET}}
- Review model: double-blind. The submission must be fully anonymized.
- Omit author names, affiliations, acknowledgements, and funding statements from the manuscript.
- Do not include self-identifying links such as personal repositories, lab pages, or internal URLs.
- Refer to your own prior work in the third person; never write phrasing that reveals authorship.
- Include a separate `LLM Usage` section, which may appear in the appendix, describing the precise role
  of LLMs in research ideation and writing. Significant LLM use must be disclosed; nondisclosure can
  cause desk rejection.
- The paper must satisfy the ICLR template and formatting rules selected by `--paper-format {{PAPER_FORMAT}}`.

## Audience
- Machine learning researchers working on representation learning, optimization, architectures, and
  empirical methodology.
- Readers expect a clearly stated contribution, honest baselines, controlled comparisons, ablations
  that isolate the mechanism, and explicit statements of scope and limitations.
- Reviewers reward reproducibility and calibrated claims far more than headline numbers.

## Framing Priorities
- Lead with the research question and the specific claim the evidence supports.
- State the contribution precisely in the abstract and introduction: what is new, what it improves,
  and under which conditions.
- Motivate the method mechanistically. Explain why it should work, not only that it does.
- Support the central claim with controlled comparisons against meaningful baselines.
- Use ablations to attribute the gain to the proposed component rather than to incidental changes.
- Report variance across seeds or runs whenever the artifacts provide it.
- Be explicit about the regime studied and where the result is not expected to hold.

## Manuscript Guidance
- Prefer a title and abstract that name the mechanism and the setting, not a generic improvement claim.
- In the introduction, make the contribution list concrete and verifiable against the artifacts.
- In the method section, give enough detail to reimplement: objective, architecture, and the
  hyperparameters that matter.
- In the results section, foreground controlled comparisons, ablations, sensitivity, and failure modes.
- Include a limitations discussion that a reviewer would otherwise raise.
- Figures must meet top-tier conference standards in both design (clean typography, consistent colors,
  proper labels and legends, high resolution) and scientific content (clear message, no filler).
- The template is single-column: size figures for the full text width and keep axis labels legible.

## Anti-Goals
- Do not deanonymize the submission in any way.
- Do not overclaim generality beyond the settings actually evaluated.
- Do not present a benchmark delta as a mechanistic insight without evidence connecting the two.
- Do not omit baselines or ablations that the grounded artifacts already support.
- Do not describe results as state of the art unless the comparison in the artifacts establishes it.
- Do not pad the paper to fill the page budget.

## Decision Rule
- When venue framing and grounded evidence conflict, follow the grounded evidence.
- Use this brief to choose emphasis, terminology, and narrative framing, not to invent stronger claims
  than the run supports.
