import re
import math
from collections import Counter

class AdvancedTextAnalyzer:
    def __init__(self, text: str):
        self.raw_text = text
        self.cleaned_text = self._clean_text()
        self.words = self._tokenize()
        self.sentences = self._split_sentences()

    def _clean_text(self) -> str:
        return re.sub(r'\s+', ' ', self.raw_text).strip()

    def _tokenize(self) -> list:
        return re.findall(r'\b\w+\b', self.cleaned_text.lower())

    def _split_sentences(self) -> list:
        return [s.strip() for s in re.split(r'[.!?]+', self.raw_text) if s.strip()]

    def calculate_basic_metrics(self) -> dict:
        char_count_with_spaces = len(self.raw_text)
        char_count_no_spaces = len(self.raw_text.replace(" ", ""))
        word_count = len(self.words)
        sentence_count = max(1, len(self.sentences))
        
        avg_word_length = sum(len(w) for w in self.words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count

        return {
            "char_count_total": char_count_with_spaces,
            "char_count_clean": char_count_no_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2)
        }

    def extract_top_keywords(self, top_n: int = 5) -> list:
        stop_words = {'a', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'it', 'of', 'with', 'by'}
        filtered_words = [w for w in self.words if w not in stop_words and len(w) > 2]
        return Counter(filtered_words).most_common(top_n)

    def estimate_readability(self) -> float:
        word_count = len(self.words)
        char_count = len(self.cleaned_text.replace(" ", ""))
        sentence_count = max(1, len(self.sentences))

        if word_count == 0:
            return 0.0

        score = 4.71 * (char_count / word_count) + 0.5 * (word_count / sentence_count) - 21.43
        return round(max(0.0, score), 1)

    def analyze_sentiment_scores(self) -> dict:
        positive_lexicon = {'great', 'excellent', 'love', 'perfect', 'good', 'awesome', 'smart', 'open-source'}
        negative_lexicon = {'bad', 'error', 'slow', 'fail', 'issue', 'problem', 'worst', 'broken'}

        pos_count = sum(1 for w in self.words if w in positive_lexicon)
        neg_count = sum(1 for w in self.words if w in negative_lexicon)

        total = pos_count + neg_count
        if total == 0:
            sentiment = "Neutral"
        elif pos_count > neg_count:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        return {
            "sentiment_label": sentiment,
            "positive_matches": pos_count,
            "negative_matches": neg_count
        }

if __name__ == "__main__":
    sample_dataset = (
        "Anthropic models provide excellent performance for developers worldwide. "
        "Building open-source software is great, but maintaining clean code can be a complex problem. "
        "This platform executes localized data parsing instantly and avoids any severe error."
    )

    print("[SYSTEM] Initializing AuraText NLP Core Engine...")
    analyzer = AdvancedTextAnalyzer(sample_dataset)
    
    metrics = analyzer.calculate_basic_metrics()
    keywords = analyzer.extract_top_keywords(3)
    readability = analyzer.estimate_readability()
    sentiment = analyzer.analyze_sentiment_scores()

    print("\n=== TELEMETRY REPORT ===")
    for key, value in metrics.items():
        print(f"-> {key.replace('_', ' ').title()}: {value}")
        
    print(f"-> Top Core Keywords: {keywords}")
    print(f"-> Automated Readability Index: {readability}")
    print(f"-> Derived Sentiment: {sentiment['sentiment_label']} (Pos: {sentiment['positive_matches']}, Neg: {sentiment['negative_matches']})")
    print("=========================")
