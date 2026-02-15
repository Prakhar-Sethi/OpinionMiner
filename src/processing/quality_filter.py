"""
Quality filter using Groq LLM for OpinionMiner.
Filters out low-quality and rage-bait content.
"""

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityFilter:
    """Filters opinions for quality using Groq LLM."""

    QUALITY_PROMPT_TEMPLATE = """You are analyzing a contrarian opinion for quality. Rate the following:

Opinion Title: {title}
Opinion Content: {content}

Evaluate on these criteria:
1. Argument Quality (0-10): Is the reasoning sound and well-structured?
2. Evidence (0-10): Does it cite sources, data, or concrete examples?
3. Coherence (0-10): Is the logic internally consistent?
4. Rage Bait (yes/no): Is it just provocative without substance?
5. Intellectual Value (0-10): Does it provide genuine insight or just controversy?

Return ONLY a JSON object with no other text:
{{
  "quality_score": <average of scores 1,2,3,5>,
  "argument_quality": <score>,
  "evidence": <score>,
  "coherence": <score>,
  "is_rage_bait": <true/false>,
  "intellectual_value": <score>,
  "reasoning": "<brief explanation>"
}}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize quality filter.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env variable)
            model: Model to use for quality assessment
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("No Groq API key found. Quality filter will use fallback heuristics.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

        self.model = model
        self.quality_threshold = float(os.getenv('QUALITY_THRESHOLD', '6.0'))

    def filter_opinion(self, opinion_text: str, title: str = "") -> Dict[str, Any]:
        """
        Filter a single opinion for quality.

        Args:
            opinion_text: Opinion content
            title: Opinion title

        Returns:
            Dictionary with quality_score, is_rage_bait, and reasoning
        """
        # Try LLM filtering if available
        if self.client:
            try:
                return self._llm_filter(opinion_text, title)
            except Exception as e:
                logger.warning(f"LLM filtering failed: {e}. Using fallback.")
                return self._fallback_filter(opinion_text, title)
        else:
            return self._fallback_filter(opinion_text, title)

    def _llm_filter(self, content: str, title: str) -> Dict[str, Any]:
        """
        Use Groq LLM to assess quality.

        Args:
            content: Opinion content
            title: Opinion title

        Returns:
            Quality assessment dictionary
        """
        # Truncate content if too long (to save tokens)
        max_content_length = 2000
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "... [truncated]"

        # Create prompt
        prompt = self.QUALITY_PROMPT_TEMPLATE.format(
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
                        {"role": "system", "content": "You are a quality assessment expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                # Parse response
                response_text = response.choices[0].message.content.strip()

                # Extract JSON (handle potential markdown code blocks)
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                result = json.loads(response_text)

                # Validate required fields
                required_fields = ['quality_score', 'is_rage_bait']
                if all(field in result for field in required_fields):
                    logger.debug(f"Quality assessment: score={result['quality_score']}, rage_bait={result['is_rage_bait']}")
                    return result
                else:
                    logger.warning(f"Missing required fields in LLM response: {result}")
                    raise ValueError("Incomplete response from LLM")

            except Exception as e:
                logger.warning(f"LLM filtering attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        raise Exception("All LLM filtering attempts failed")

    def _fallback_filter(self, content: str, title: str) -> Dict[str, Any]:
        """
        Fallback heuristic-based quality assessment.

        Args:
            content: Opinion content
            title: Opinion title

        Returns:
            Quality assessment dictionary
        """
        logger.debug("Using fallback heuristic quality filter")

        # Simple heuristics
        content_length = len(content)
        title_length = len(title)

        # Base score from length (longer = more substantial)
        length_score = min(10.0, (content_length / 500) * 5)

        # Check for evidence markers
        evidence_markers = ['study', 'research', 'data', 'evidence', 'source', 'according to']
        evidence_count = sum(1 for marker in evidence_markers if marker.lower() in content.lower())
        evidence_score = min(10.0, evidence_count * 2)

        # Check for rage bait indicators
        rage_indicators = ['hate', 'stupid', 'idiots', 'worst', 'terrible', 'destroyed', 'owned']
        rage_count = sum(1 for indicator in rage_indicators if indicator.lower() in content.lower())
        is_rage_bait = rage_count >= 3

        # Calculate overall quality score
        quality_score = (length_score + evidence_score) / 2

        # Penalize very short content
        if content_length < 200:
            quality_score *= 0.7

        # Penalize rage bait
        if is_rage_bait:
            quality_score *= 0.5

        return {
            'quality_score': round(quality_score, 2),
            'argument_quality': round(length_score, 2),
            'evidence': round(evidence_score, 2),
            'coherence': 7.0,  # Neutral assumption
            'is_rage_bait': is_rage_bait,
            'intellectual_value': round(quality_score, 2),
            'reasoning': 'Heuristic-based assessment (LLM unavailable)'
        }

    def filter_batch(self, opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a batch of opinions.

        Args:
            opinions: List of opinion dictionaries

        Returns:
            List of opinions with quality assessments added
        """
        filtered_opinions = []

        for opinion in opinions:
            try:
                # Get quality assessment
                assessment = self.filter_opinion(
                    opinion.get('content', ''),
                    opinion.get('title', '')
                )

                # Add assessment to opinion
                opinion['quality_score'] = assessment['quality_score']
                opinion['is_rage_bait'] = assessment['is_rage_bait']

                filtered_opinions.append(opinion)

                logger.info(f"Filtered: {opinion.get('title', '')[:50]}... " +
                          f"(score={assessment['quality_score']:.2f}, " +
                          f"rage_bait={assessment['is_rage_bait']})")

            except Exception as e:
                logger.error(f"Error filtering opinion: {e}")
                continue

        return filtered_opinions

    def passes_threshold(self, quality_score: float, is_rage_bait: bool) -> bool:
        """
        Check if an opinion passes quality threshold.

        Args:
            quality_score: Quality score (0-10)
            is_rage_bait: Whether it's rage bait

        Returns:
            True if passes threshold
        """
        return quality_score >= self.quality_threshold and not is_rage_bait


# Test/CLI interface
if __name__ == "__main__":
    filter = QualityFilter()

    # Test with sample opinion
    test_opinion = {
        'title': 'Universal Basic Income is not economically viable',
        'content': '''Recent economic studies have shown that implementing a true Universal Basic Income
        would require a significant restructuring of the tax system. The data from pilot programs in
        Finland and Kenya suggest that while UBI can reduce poverty, the inflationary pressures and
        labor market effects are more complex than advocates suggest. We need more rigorous
        evidence-based analysis before implementing such sweeping changes.'''
    }

    result = filter.filter_opinion(test_opinion['content'], test_opinion['title'])

    print("\n=== Quality Filter Test ===")
    print(f"Title: {test_opinion['title']}")
    print(f"\nAssessment:")
    print(f"  Quality Score: {result['quality_score']:.2f}")
    print(f"  Is Rage Bait: {result['is_rage_bait']}")
    print(f"  Passes Threshold: {filter.passes_threshold(result['quality_score'], result['is_rage_bait'])}")
    if 'reasoning' in result:
        print(f"  Reasoning: {result['reasoning']}")
