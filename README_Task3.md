# Task 3: Intelligent Feature — Evaluation, Failure Cases & Demo

This builds on Task 2 by adding more evaluation examples, documented
failure cases, and a simple interactive interface for live testing.

## Problem
Same as Task 1/2: classify a comment's sentiment as **Positive**,
**Negative**, or **Neutral**, using my own trained model
(TF-IDF + Logistic Regression).

## 1. Evaluation Examples
Confirmed correct predictions against known expected labels:

```python
eval_examples = [
    ('This trailer looks incredible!', 'Positive'),
    ('Bring back physical games!', 'Negative'),
    ('5/10', 'Neutral'),
]
```

| Input | Expected | Predicted | Result |
|---|---|---|---|
| 'This trailer looks incredible!' | Positive | Positive | ✅ |
| 'Bring back physical games!' | Negative | Negative | ✅ |
| '5/10' | Neutral | Neutral | ✅ |

## 2. Edge Cases
Short, sarcastic, repetitive, emoji-only, and mixed-sentiment inputs, to
probe the limits of the model:

| Input | Predicted | Note |
|---|---|---|
| 'ok' | Negative | Too short/ambiguous, misleading |
| 'yeah right, amazing launch' | Negative | Sarcasm, correct by coincidence |
| 'SONY SONY SONY' | Negative | No sentiment words, unreliable |
| '🤩🤩🤩' | Negative | Emoji-only, model can't read emoji meaning |
| 'Not bad, not great either' | Negative | Mixed sentiment, oversimplified |

## 3. Failure Cases (from the real test set)
Pulled directly from test-set predictions where the model's output did
not match the true label:

| Comment | Actual | Predicted | Likely Reason |
|---|---|---|---|
| "'play has no limits'. If you follow the straight line that it does have limits." | Positive | Negative | Figurative language, model reads negated phrasing as negative |
| "recent decisions the last few years have been awful. Thank you for destroying the once beloved brand." | Neutral | Negative | Mixed tone; strong negative words dominate despite a neutral framing |
| "it ends itself with digital only dynamic pricing? At this rate it won't last to see dynamic pricing?" | Neutral | Negative | Rhetorical question read as a complaint |
| "I bought the physical version of this game! Great job Insomniac! Shame on you Sony!" | Negative | Neutral | Mixed praise and criticism in one comment confuses the model |
| "GREED HAS NO LIMITS!" | Negative | Neutral | All-caps emotional phrasing not captured by TF-IDF |
| "...while I still can before I make the switch to PC next generation. It's been fun. RIP PlayStation" | Negative | Neutral | Sarcasm/resignation tone, no explicit negative keywords |
| "Keep the Physical Format alive beyond 2028!" | Neutral | Negative | A request/opinion misread as a complaint |

**Pattern:** most failures involve sarcasm, mixed sentiment in one comment,
or emotional tone without explicit negative/positive keywords. These are
known weaknesses of a TF-IDF + Logistic Regression model, since it scores
individual words/phrases and has no understanding of context, tone, or
irony.

## 4. Simple Interface (Notebook Demo)
A small interactive widget built with `ipywidgets`, so the model can be
tested live inside the notebook without writing code for each test:

```python
import ipywidgets as widgets
from IPython.display import display

text_box = widgets.Text(placeholder='Type a comment...')
button = widgets.Button(description='Predict')
output = widgets.Output()

def on_click(b):
    output.clear_output()
    with output:
        print(predict_sentiment(text_box.value))

button.on_click(on_click)
display(text_box, button, output)
```

**Demo result:** typing `"Not bad, not great either"` and clicking
**Predict** returned `Negative`, showing the model's weakness on mixed
or hedged sentiment in real time.

## Constraints
- No external API or secret keys; fully self-contained model
- Free tools only (Google Colab, scikit-learn, joblib, ipywidgets)
- Small, imbalanced training set (83 labeled comments)

## Evaluation Summary
- Accuracy: 58.8%
- Macro F1: 0.32
- Per-class F1: Negative 0.72, Neutral 0.25, Positive 0.00
- Main weaknesses: sarcasm, mixed-sentiment comments, emoji-only input,
  and very short inputs

## Future Work
- Train on more data, especially sarcastic and mixed-sentiment examples
- Try a pretrained transformer (e.g., DistilBERT) for better context
  understanding
- Add a confidence score to each prediction so low-confidence cases can
  be flagged for manual review

## Files
- `sentiment_model.pkl`: saved trained model
- `notebook.ipynb`: training, evaluation, failure case analysis, and
  interactive demo
- `README.md`: this document
