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
        
        # Dynamic sentiment words initialization
        self.sentiment_threshold = 0.4  # Threshold for considering a word as having strong sentiment
        self.sentiment_words = self._initialize_sentiment_words()
        
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
        
        # Extract sentences with sentiment words and update sentiment words dynamically
        sentiment_phrases = []
        document_sentiment_words = self._extract_sentiment_words(doc)
        
        # Update our sentiment words based on the current text
        self.sentiment_words.update(document_sentiment_words)
        
        # Find sentences containing sentiment words
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(word in sent_text.lower() for word in self.sentiment_words):
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
    
    def _initialize_sentiment_words(self):
        """Initialize a set of sentiment words from VADER lexicon and common sentiment terms"""
        # Start with a baseline of common sentiment words
        baseline_sentiment_words = {"feeling", "help", "should", "busy", "concern", 
                                    "improve", "problem", "issue", "excellent", "terrible", 
                                    "frustrat", "worry", "difficult", "solution"}
        
        # Get words from VADER lexicon that have strong sentiment
        lexicon = self.sentiment_analyzer.lexicon
        strong_sentiment_words = {word for word, score in lexicon.items() 
                                 if abs(score) > self.sentiment_threshold
                                 and len(word) > 3  # Filter out very short words
                                 and '_' not in word}  # Filter out multi-word phrases with underscores
        
        # Combine both sets
        return baseline_sentiment_words.union(strong_sentiment_words)
    
    def _extract_sentiment_words(self, doc):
        """Extract sentiment words from the given spaCy document"""
        sentiment_words = set()
        
        # Track which words were found in this particular document
        self.found_sentiment_words = set()  # Store words found in the current text
        
        # Extract words with notable sentiment scores
        for sent in doc.sents:
            # Check individual tokens
            for token in sent:
                # Skip stopwords, punctuation, and very short words
                if (token.is_stop or token.is_punct or len(token.text) < 4 or
                    token.text.lower() in self.stopwords):
                    continue
                
                # Focus on content words more likely to carry sentiment
                if token.pos_ in ('ADJ', 'ADV', 'VERB', 'NOUN'):
                    word = token.text.lower()
                    # Get sentiment score for this word
                    word_score = self.sentiment_analyzer.polarity_scores(token.text)
                    
                    # If compound score exceeds threshold in either direction, add to sentiment words
                    if abs(word_score['compound']) > self.sentiment_threshold:
                        sentiment_words.add(word)
                        
                        # Check if this word is also in our lexicon
                        if word in self.sentiment_words:
                            self.found_sentiment_words.add(word)
                        
                    # Check for specific word types that often indicate sentiment
                    if token.pos_ == 'ADJ' and abs(word_score['compound']) > 0.2:
                        sentiment_words.add(word)
                        
                        # Check if this word is also in our lexicon
                        if word in self.sentiment_words:
                            self.found_sentiment_words.add(word)
                        
                    # Add our baseline sentiment words if found in text
                    if word in self.sentiment_words:
                        self.found_sentiment_words.add(word)
        
        return sentiment_words
    
    def _extract_fallback_issues(self, text):
        """Fallback method for extracting issues if our pattern-based approach doesn't find any"""
        # Simple approach: split into sentences and consider each a potential issue
        sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if sent.strip()]
        return [f"Issue {i+1}: {sent}" for i, sent in enumerate(sentences)]
        
    def analyze_sentiment(self, text):
        """Perform sentiment analysis on the text"""
        # First, extract sentiment words to build our keyword sets
        # Process text with spaCy 
        doc = self.nlp(text)
        
        # Extract sentiment words and update our knowledge base
        self._extract_sentiment_words(doc)
        
        # Now determine sentiment with our enhanced understanding
        # Get words from the text that have sentiment value
        found_positive_words = []
        found_negative_words = []
        
        # Get words' sentiment polarity from our lexicon
        lexicon = self.sentiment_analyzer.lexicon
        
        # Create word-to-sentiment mapping for found words
        word_sentiments = {}
        for word in self.found_sentiment_words:
            # Get sentiment from VADER lexicon if available
            if word in lexicon:
                score = lexicon[word]
                word_sentiments[word] = score
                
                # Categorize by polarity
                if score > 0.2:
                    found_positive_words.append(word)
                elif score < -0.2:
                    found_negative_words.append(word)
        
        # Also check for our baseline sentiment words not in VADER
        for word in self.found_sentiment_words:
            # These are our additional positive sentiment words
            if word in ["improve", "help", "solution", "fresh", "smoother", "stronger"] and word not in word_sentiments:
                found_positive_words.append(word)
                word_sentiments[word] = 0.4  # Assign a moderately positive score
                
            # These are our additional negative sentiment words
            elif word in ["concern", "busy", "problem", "issue", "peak"] and word not in word_sentiments:
                found_negative_words.append(word)
                word_sentiments[word] = -0.3  # Assign a moderately negative score
        
        # Apply word-aware sentiment analysis
        # Get base sentiment score using VADER
        base_sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Create enhanced scores that take into account our found words
        enhanced_scores = base_sentiment_scores.copy()
        
        # Adjust based on found sentiment words
        if len(found_positive_words) > len(found_negative_words) * 2:
            # Many more positive words - boost positive score
            enhanced_scores['pos'] += 0.15
            enhanced_scores['compound'] += 0.1
        elif len(found_negative_words) > len(found_positive_words) * 2:
            # Many more negative words - boost negative score
            enhanced_scores['neg'] += 0.15
            enhanced_scores['compound'] -= 0.1
            
        # Determine overall sentiment using enhanced scores
        if enhanced_scores['compound'] >= 0.05:
            overall = "positive"
        elif enhanced_scores['compound'] <= -0.05:
            overall = "negative"
        else:
            overall = "neutral"
            
        # If it's close to neutral but has both positive and negative elements
        if abs(enhanced_scores['compound']) < 0.2 and enhanced_scores['pos'] > 0.1 and enhanced_scores['neg'] > 0.1:
            overall = "mixed"
        
        # Identify positive and negative elements
        positive_elements = []
        negative_elements = []
        neutral_elements = []
        
        # Use the specific sentiment words we found to enhance sentence classification
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_words = set(token.text.lower() for token in sent)
            
            # Check for sentiment words in this sentence
            sent_pos_words = sent_words.intersection(found_positive_words)
            sent_neg_words = sent_words.intersection(found_negative_words)
            
            # Base sentiment score from VADER
            base_sent_score = self.sentiment_analyzer.polarity_scores(sent_text)
            
            # Adjust sentence score based on found sentiment words
            sent_score = base_sent_score.copy()
            
            # Boost scores based on found words
            if sent_pos_words:
                sent_score['pos'] += 0.05 * len(sent_pos_words)
                sent_score['compound'] += 0.05 * len(sent_pos_words)
            if sent_neg_words:
                sent_score['neg'] += 0.05 * len(sent_neg_words)
                sent_score['compound'] -= 0.05 * len(sent_neg_words)
            
            # Classify elements based on enhanced sentiment score
            if sent_score['compound'] >= 0.05:
                # Look for positive indicators using our found words
                if any(word in sent_text.lower() for word in found_positive_words):
                    if any(word in sent_text.lower() for word in ["help", "improve", "solution"]):
                        positive_elements.append("Proactive suggestions for improvement")
                    elif "fresh" in sent_text.lower() or "smoother" in sent_text.lower() or "stronger" in sent_text.lower():
                        positive_elements.append("Appreciation for efficiency and quality")
                    else:
                        positive_elements.append("Professional tone with positive sentiment")
                else:
                    positive_elements.append(sent_text)
            
            elif sent_score['compound'] <= -0.05:
                # Look for negative indicators using our found words
                if any(word in sent_text.lower() for word in found_negative_words):
                    if "busy" in sent_text.lower() or "peak" in sent_text.lower():
                        negative_elements.append("Concern about workload")
                    elif "company" in sent_text.lower() or "business" in sent_text.lower() or "routes" in sent_text.lower():
                        negative_elements.append("Worry about business decline")
                    else:
                        negative_elements.append("Issue identified: " + sent_text)
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
        
        # Store the found sentiment words and their polarities for reference
        self.word_sentiments = word_sentiments
        
        return {
            "overall_sentiment": overall,
            "sentiment_scores": enhanced_scores,
            "positive_elements": positive_elements,
            "negative_elements": negative_elements,
            "neutral_elements": neutral_elements,
            "found_positive_words": found_positive_words,
            "found_negative_words": found_negative_words
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze text for key phrases and sentiment.')
    parser.add_argument('text', nargs='?', help='Text to analyze. If not provided, will prompt for input.')
    parser.add_argument('--show-sentiment-words', action='store_true', help='Display identified sentiment words')
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
    
    # Show sentiment words if requested
    if args.show_sentiment_words:
        print("\n" + "="*50)
        print("SENTIMENT WORDS IDENTIFIED")
        print("="*50)
        print("\nSentiment words detected in this text:")
        
        # Get words actually found in the text
        if hasattr(analyzer, 'found_sentiment_words') and analyzer.found_sentiment_words:
            # Display positive and negative words separately
            if hasattr(analyzer, 'word_sentiments') and hasattr(sentiment, 'found_positive_words') and hasattr(sentiment, 'found_negative_words'):
                print("\nPositive sentiment words:")
                if sentiment['found_positive_words']:
                    # Get sentiment scores for context
                    positive_with_scores = [(word, analyzer.word_sentiments.get(word, 'N/A')) 
                                          for word in sorted(sentiment['found_positive_words'])]
                    for word, score in positive_with_scores:
                        print(f"  {word} ({score})")
                else:
                    print("  No positive sentiment words found")
                    
                print("\nNegative sentiment words:")
                if sentiment['found_negative_words']:
                    # Get sentiment scores for context
                    negative_with_scores = [(word, analyzer.word_sentiments.get(word, 'N/A')) 
                                          for word in sorted(sentiment['found_negative_words'])]
                    for word, score in negative_with_scores:
                        print(f"  {word} ({score})")
                else:
                    print("  No negative sentiment words found")
            else:
                # Fallback to the basic display if new features aren't available
                sorted_words = sorted(analyzer.found_sentiment_words)
                
                # Generate a user-friendly output
                if sorted_words:
                    for i in range(0, len(sorted_words), 5):
                        print(", ".join(sorted_words[i:i+5]))
                else:
                    print("No sentiment words identified in this text.")
        else:
            print("No sentiment words identified in this text.")

if __name__ == "__main__":
    main()
