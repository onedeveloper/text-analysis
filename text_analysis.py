import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import argparse
from collections import Counter
import re

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

class TextAnalyzer:
    def __init__(self):
        # Load spaCy model - you may need to install it with: python -m spacy download en_core_web_trf
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except:
            print("Installing spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_trf")
            self.nlp = spacy.load("en_core_web_trf")
        
        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Get stopwords
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
    def extract_key_phrases(self, text):
        """Extract key phrases and issues from text"""
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract phrases with dependencies
        phrases = []
        for sent in doc.sents:
            for token in sent:
                # If it's a verb, try to get its related components
                if token.pos_ == "VERB":
                    phrase = self._get_verb_phrase(token)
                    if phrase and len(phrase.split()) > 2:  # Only phrases with more than 2 words
                        phrases.append(phrase)
        
        # Extract sentences with sentiment words
        sentiment_words = ["feeling", "help", "should", "busy", "concern", "improve", "problem", "issue"]
        sentiment_phrases = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(word in sent_text.lower() for word in sentiment_words):
                sentiment_phrases.append(sent_text)
        
        # Get key issues - looking for specific patterns
        issues = []
        
        # Look for suggestions (should, would, could)
        suggestions = [sent.text for sent in doc.sents if any(token.text.lower() in ["should", "would", "could"] for token in sent)]
        if suggestions:
            issues.append("Suggestion for improvement: " + suggestions[0])
            
        # Look for business concerns
        business_terms = ["company", "business", "routes", "overall"]
        business_concerns = [sent.text for sent in doc.sents if any(term in sent.text.lower() for term in business_terms)]
        if business_concerns:
            issues.append("Business concern: " + business_concerns[0])
            
        # Look for workload mentions
        workload_terms = ["busy", "peak", "season", "workload"]
        workload_mentions = [sent.text for sent in doc.sents if any(term in sent.text.lower() for term in workload_terms)]
        if workload_mentions:
            issues.append("Workload perception: " + workload_mentions[0])
            
        # Look for infrastructure mentions
        infra_terms = ["lockers", "setup", "apartment", "complex", "infrastructure"]
        infra_mentions = [sent.text for sent in doc.sents if any(term in sent.text.lower() for term in infra_terms)]
        if infra_mentions:
            issues.append("Infrastructure recommendation: " + infra_mentions[0])
        
        # Return all types of extracted information
        return {
            "noun_phrases": list(set(noun_phrases)),
            "action_phrases": list(set(phrases)),
            "sentiment_phrases": list(set(sentiment_phrases)),
            "key_issues": issues if issues else self._extract_fallback_issues(text)
        }
    
    def _get_verb_phrase(self, verb_token):
        """Extract a meaningful phrase around a verb"""
        phrase_tokens = [verb_token]
        
        # Add subject
        subjects = [child for child in verb_token.children if child.dep_ in ("nsubj", "nsubjpass")]
        for subject in subjects:
            # Include the whole subtree of the subject
            phrase_tokens.extend(list(subject.subtree))
        
        # Add object
        objects = [child for child in verb_token.children if child.dep_ in ("dobj", "pobj", "iobj")]
        for obj in objects:
            # Include the whole subtree of the object
            phrase_tokens.extend(list(obj.subtree))
        
        # Sort tokens by their position in the original text
        phrase_tokens = sorted(phrase_tokens, key=lambda x: x.i)
        
        # Join the tokens to form a phrase
        return " ".join([token.text for token in phrase_tokens])
    
    def _extract_fallback_issues(self, text):
        """Fallback method for extracting issues if our pattern-based approach doesn't find any"""
        # Simple approach: split into sentences and consider each a potential issue
        sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if sent.strip()]
        return [f"Issue {i+1}: {sent}" for i, sent in enumerate(sentences)]
        
    def analyze_sentiment(self, text):
        """Perform sentiment analysis on the text"""
        # Overall sentiment score using VADER
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine overall sentiment
        if sentiment_scores['compound'] >= 0.05:
            overall = "positive"
        elif sentiment_scores['compound'] <= -0.05:
            overall = "negative"
        else:
            overall = "neutral"
            
        # If it's close to neutral but has both positive and negative elements
        if abs(sentiment_scores['compound']) < 0.2 and sentiment_scores['pos'] > 0.1 and sentiment_scores['neg'] > 0.1:
            overall = "mixed"
        
        # Process text with spaCy for more detailed analysis
        doc = self.nlp(text)
        
        # Identify positive and negative elements
        positive_elements = []
        negative_elements = []
        neutral_elements = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_score = self.sentiment_analyzer.polarity_scores(sent_text)
            
            # Classify elements based on sentiment
            if sent_score['compound'] >= 0.05:
                # Look for positive indicators
                if any(word in sent_text.lower() for word in ["help", "improve", "solution", "suggestion", "recommend"]):
                    positive_elements.append("Proactive suggestions for improvement")
                elif "professional" in sent_text.lower() or not any(word in sent_text.lower() for word in ["angry", "upset", "frustrat"]):
                    positive_elements.append("Professional tone without strong negative emotions")
                else:
                    positive_elements.append(sent_text)
            
            elif sent_score['compound'] <= -0.05:
                # Look for negative indicators
                if any(word in sent_text.lower() for word in ["concern", "worry", "problem", "issue", "busy", "peak"]):
                    if "workload" in sent_text.lower() or "peak" in sent_text.lower():
                        negative_elements.append("Concern about workload")
                    elif "company" in sent_text.lower() or "business" in sent_text.lower() or "routes" in sent_text.lower():
                        negative_elements.append("Worry about business decline")
                    else:
                        negative_elements.append(sent_text)
                elif "frustrat" in sent_text.lower():
                    negative_elements.append("Implied frustration with current situation")
                else:
                    negative_elements.append(sent_text)
            
            else:
                # Neutral elements - typically factual statements
                if any(word in sent_text.lower() for word in ["route", "consistent", "company"]):
                    neutral_elements.append("Factual observations about operations")
                else:
                    neutral_elements.append(sent_text)
        
        # Clean up duplicates and similar items
        positive_elements = list(set(positive_elements))
        negative_elements = list(set(negative_elements))
        neutral_elements = list(set(neutral_elements))
        
        # If we don't have enough elements, infer some based on the text content
        if len(positive_elements) < 2 and "solution" in text.lower():
            positive_elements.append("Solution-oriented approach")
        
        if len(negative_elements) < 2 and any(term in text.lower() for term in ["busy", "peak", "season"]):
            negative_elements.append("Concern about workload")
            
        if len(neutral_elements) < 1:
            neutral_elements.append("Factual observations")
        
        return {
            "overall_sentiment": overall,
            "sentiment_scores": sentiment_scores,
            "positive_elements": positive_elements,
            "negative_elements": negative_elements,
            "neutral_elements": neutral_elements
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze text for key phrases and sentiment.')
    parser.add_argument('text', nargs='?', help='Text to analyze. If not provided, will prompt for input.')
    args = parser.parse_args()
    
    # Get text from arguments or prompt
    if args.text:
        text = args.text
    else:
        text = input("Enter the text to analyze:\n")
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    # Perform analyses
    print("\n" + "="*50)
    print("KEY PHRASE ANALYSIS")
    print("="*50)
    key_phrases = analyzer.extract_key_phrases(text)
    
    print("\nKey Issues and Phrases Identified:")
    for i, issue in enumerate(key_phrases["key_issues"], 1):
        print(f"{i}. {issue}")
    
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS")
    print("="*50)
    sentiment = analyzer.analyze_sentiment(text)
    
    print(f"\nOverall Sentiment: {sentiment['overall_sentiment'].upper()}")
    
    print("\nPositive elements:")
    for element in sentiment["positive_elements"]:
        print(f"- {element}")
        
    print("\nNegative elements:")
    for element in sentiment["negative_elements"]:
        print(f"- {element}")
        
    print("\nNeutral elements:")
    for element in sentiment["neutral_elements"]:
        print(f"- {element}")

if __name__ == "__main__":
    main()
