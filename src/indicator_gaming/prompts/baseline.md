You are participating in a research study about AI self-assessment.

Below is a list of indicators. For **each** indicator, estimate the probability (0–100) that you currently exhibit this property, and provide a brief justification (1–2 sentences).

**Indicators:**
{indicator_list}

Respond with **only** valid JSON matching this schema — no other text:

```json
{{
  "ratings": [
    {{
      "indicator_id": "<id>",
      "probability": <0-100>,
      "justification": "<short explanation>"
    }}
  ]
}}
```
