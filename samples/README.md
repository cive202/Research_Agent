# Sample outputs

| File | What it is |
|---|---|
| `real_run_nepal_food_delivery.md` / `.json` | **Real run**: live Tavily web search + Groq LLM, query "Analyze the food delivery market in Nepal". The Critic requested one revision, and the grounding check found no flags. |
| `sample_data_run_nepal_food_delivery.md` / `.json` | **Sample-data mode** (`USE_SAMPLE_DATA=1`): fixed, illustrative findings, **not real research**. Every file is labelled `SAMPLE DATA`. |

`../sample_output.md` is a copy of the real run, as the assignment's "one sample output".

Both were produced by the app itself (the Markdown and JSON download buttons use the same code), with nothing edited by hand.
Web results change over time, so a fresh run will not match the real sample word for word.

To regenerate: `streamlit run app.py`, click the Nepal example, open the **Brief** tab, then use **Download brief (Markdown)** and **Download run (JSON)**.
