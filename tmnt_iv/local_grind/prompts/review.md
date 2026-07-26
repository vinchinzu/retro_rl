Review this trial result and return JSON only:
{
  "notes": "short observation from metrics and any screenshots",
  "next_hint": "what to try next (knob names / direction)",
  "suspect_regression": true/false
}

Trial proposal:
{proposal}

Metrics:
{metrics}

Score: {score} (baseline {baseline_score}, delta {delta_score})
Decision already computed by harness: {decision}

If screenshots are attached, mention what looks wrong (spacing, spin hits,
stuck on wall, idle thrash). Keep notes under 80 words.
