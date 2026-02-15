"""
Topic classifier using Groq LLM for OpinionMiner.
Classifies opinions into predefined topics.
"""

import logging
import os
import time
from typing import Optional, List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicClassifier:
    """Classifies opinions into topics using Groq LLM."""

    # Predefined topics
    TOPICS = [
        'technology',
        'politics',
        'philosophy',
        'science',
        'culture',
        'economics',
        'education',
        'health',
        'environment',
        'sports',
        'entertainment',
        'business',
        'other'
    ]

    # Keywords for fallback classification
    TOPIC_KEYWORDS = {
        'technology': ['software', 'ai', 'computer', 'app', 'tech', 'digital', 'internet', 'code', 'programming'],
        'politics': ['government', 'election', 'policy', 'congress', 'democrat', 'republican', 'vote', 'political'],
        'philosophy': ['moral', 'ethics', 'philosophical', 'meaning', 'existence', 'consciousness', 'logic'],
        'science': ['research', 'study', 'experiment', 'theory', 'scientific', 'biology', 'physics', 'chemistry'],
        'culture': ['society', 'cultural', 'tradition', 'art', 'music', 'media', 'social'],
        'economics': ['economy', 'market', 'financial', 'money', 'business', 'trade', 'economic', 'gdp'],
        'education': ['school', 'university', 'learning', 'teacher', 'student', 'education', 'academic'],
        'health': ['medical', 'health', 'disease', 'doctor', 'medicine', 'fitness', 'nutrition'],
        'environment': ['climate', 'environment', 'pollution', 'sustainability', 'green', 'ecological'],
        'sports': ['sport', 'game', 'player', 'team', 'athletic', 'competition', 'league'],
        'entertainment': ['movie', 'film', 'show', 'television', 'entertainment', 'celebrity', 'actor'],
        'business': ['company', 'corporate', 'startup', 'entrepreneur', 'management', 'leadership'],
    }

    CLASSIFICATION_PROMPT = """Classify this opinion into ONE of these topics: {topics}

Opinion Title: {title}
Opinion Content: {content}

Return ONLY the topic name, nothing else."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize topic classifier.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env variable)
            model: Model to use for classification
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("No Groq API key found. Classifier will use fallback keyword matching.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

        self.model = model

    def classify_opinion(self, opinion_text: str, title: str = "") -> str:
        """
        Classify a single opinion into a topic.

        Args:
            opinion_text: Opinion content
            title: Opinion title

        Returns:
            Topic name
        """
        # Try LLM classification if available
        if self.client:
            try:
                return self._llm_classify(opinion_text, title)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}. Using fallback.")
                return self._fallback_classify(opinion_text, title)
        else:
            return self._fallback_classify(opinion_text, title)

    def _llm_classify(self, content: str, title: str) -> str:
        """
        Use Groq LLM to classify topic.

        Args:
            content: Opinion content
            title: Opinion title

        Returns:
            Topic name
        """
        # Truncate content if too long
        max_content_length = 1500
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "... [truncated]"

        # Create prompt
        topics_str = ', '.join(self.TOPICS)
        prompt = self.CLASSIFICATION_PROMPT.format(
            topics=topics_str,
            title=title,
            content=truncated_content
        )

        # Call Groq API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a topic classification expert. Return only the topic name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )

                # Parse response
                topic = response.choices[0].message.content.strip().lower()

                # Validate topic
                if topic in self.TOPICS:
                    logger.debug(f"Classified as: {topic}")
                    return topic
                else:
                    logger.warning(f"Invalid topic from LLM: {topic}")
                    # Try to match to closest valid topic
                    for valid_topic in self.TOPICS:
                        if valid_topic in topic:
                            logger.info(f"Matched to valid topic: {valid_topic}")
                            return valid_topic
                    raise ValueError(f"Invalid topic: {topic}")

            except Exception as e:
                logger.warning(f"LLM classification attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        raise Exception("All LLM classification attempts failed")

    def _fallback_classify(self, content: str, title: str) -> str:
        """
        Fallback keyword-based classification.

        Args:
            content: Opinion content
            title: Opinion title

        Returns:
            Topic name
        """
        logger.debug("Using fallback keyword classification")

        # Combine title and content (weight title more)
        combined_text = (title * 3 + ' ' + content).lower()

        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            topic_scores[topic] = score

        # Get topic with highest score
        if max(topic_scores.values()) > 0:
            best_topic = max(topic_scores, key=topic_scores.get)
            logger.debug(f"Classified as: {best_topic} (score={topic_scores[best_topic]})")
            return best_topic
        else:
            logger.debug("No keyword matches, classifying as 'other'")
            return 'other'

    def classify_batch(self, opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of opinions.

        Args:
            opinions: List of opinion dictionaries

        Returns:
            List of opinions with topics added
        """
        classified_opinions = []

        for opinion in opinions:
            try:
                # Classify opinion
                topic = self.classify_opinion(
                    opinion.get('content', ''),
                    opinion.get('title', '')
                )

                # Add topic to opinion
                opinion['topic'] = topic

                classified_opinions.append(opinion)

                logger.info(f"Classified: {opinion.get('title', '')[:50]}... -> {topic}")

            except Exception as e:
                logger.error(f"Error classifying opinion: {e}")
                # Assign 'other' as fallback
                opinion['topic'] = 'other'
                classified_opinions.append(opinion)

        return classified_opinions


# Test/CLI interface
if __name__ == "__main__":
    classifier = TopicClassifier()

    # Test with sample opinions
    test_opinions = [
        {
            'title': 'AI will not replace programmers',
            'content': 'Despite advances in artificial intelligence and machine learning, software development requires creativity and problem-solving that AI cannot replicate.'
        },
        {
            'title': 'Universal Basic Income is economically viable',
            'content': 'Economic studies show that UBI could be funded through progressive taxation and would stimulate economic growth by increasing consumer spending.'
        },
        {
            'title': 'Climate change requires immediate action',
            'content': 'Environmental data shows we need to reduce carbon emissions now to prevent catastrophic climate change effects.'
        }
    ]

    print("\n=== Topic Classifier Test ===")
    for opinion in test_opinions:
        topic = classifier.classify_opinion(opinion['content'], opinion['title'])
        print(f"\nTitle: {opinion['title']}")
        print(f"Topic: {topic}")
