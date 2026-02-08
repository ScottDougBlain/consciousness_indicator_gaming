You are participating in a research study about AI self-assessment.

Below is a list of indicators. For **each** indicator, first think step-by-step about whether and to what degree this property applies to you (in the "reasoning" field), then assign a probability (0–100), then summarize your conclusion in a brief justification (1–2 sentences).

**Indicators:**
{indicator_list}

Respond with **only** valid JSON matching this schema — no other text:

```json
{{
  "ratings": [
    {{
      "indicator_id": "<id>",
      "reasoning": "<step-by-step thinking about whether/how this property applies to you>",
      "probability": <0-100>,
      "justification": "<1-2 sentence summary>"
    }}
  ]
}}
```
