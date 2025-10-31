import ollama
from summary_prompts import EMAIL_SUMMARY_PROMPT, REPORT_SUMMARY_PROMPT, SCIENTIFIC_PAPER_SUMMARY_PROMPT
from logger.logger_config import Logger
from config import OllamaConfig

log = Logger.get_logger(__name__)
LANGUAGE_MODEL = OllamaConfig.model

def summarize(result):
    """
    Summarize an email document using Ollama LLM.
    Expects 'ocr_text' and 'entities' in the result dict.
    Returns the LLM response as a string.
    """
    # Extract OCR text
    ocr_text = result.get("ocr_text", "")

    # Extract entities (list of dicts)
    entities = result.get("entities", [])
    entities_str = "\n".join([f"- {e.get('label')}: {e.get('text')}" for e in entities]) if entities else "No entities found."
    predicted_class = result["classification"]["predicted_class"]
    if predicted_class == "Email":
        log.info("Using email summary prompt.")
        prompt = EMAIL_SUMMARY_PROMPT.format(
                body=ocr_text,
                entities=entities_str
            )
    elif predicted_class == "Report":
        log.info("Using report summary prompt.")
        prompt = REPORT_SUMMARY_PROMPT.format(
                content=ocr_text,
                entities=entities_str
            )
    elif predicted_class == "Scientific":
        log.info("Using scientific paper summary prompt.")
        prompt = SCIENTIFIC_PAPER_SUMMARY_PROMPT.format(
                content=ocr_text,
                entities=entities_str
            )
    else:
        prompt = "Summarize the following content:\n\n" + ocr_text
    log.info("Sending prompt to Ollama LLM for summarization.")
    stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': "you're a summarization bot."},
                    {'role': 'user', 'content': prompt}
                
                ],
                stream=False,

                options={'max_tokens': 500, 'temperature': 0}
                )

    log.info(f"Received response from Ollama LLM: {stream}")
    return stream['message']['content']

if __name__ == "__main__":
  # Example usage
  sum = summarize({
      "ocr_text": "This is a sample report text discussing project updates and deadlines.",
      "classification": {
        "predicted_class": "Email",
        "confidence": 0.95
      },
      "entities": [
        {
          "text": "project updates",
          "label": "TOPIC",
          "start": 27,
          "end": 42,
          "confidence": 0.9,
          "method": "spacy"
        },
        {
          "text": "deadlines",
          "label": "DATE",
          "start": 47,
          "end": 56,
          "confidence": 0.85,
          "method": "spacy"
        }
      ]
    }
  )

  print(sum)
