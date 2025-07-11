"""
Social Media Research Agent
A comprehensive research agent for social media content creation using LangChain tools.
Follows SOLID, DRY, and KISS principles with Pydantic data validation.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field
from pydantic.types import PositiveInt, confloat

# LangChain Core Imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


# Free markdown conversion libraries
try:
    import html2text
    import markdownify
    MARKDOWN_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Free markdown libraries not installed: {e}")
    print("Install with: pip install html2text markdownify")
    MARKDOWN_LIBS_AVAILABLE = False

# LangChain tools
try:
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain dependencies not installed: {e}")
    print("Install with: pip install langchain langchain-community playwright")
    LANGCHAIN_AVAILABLE = False

class ResearchDepth(str, Enum):
    """
    Enumeration for research depth levels.
    
    Defines the intensity and scope of research to be performed.
    """
    QUICK = "quick"
    DEEP = "deep"


class ResearchQuery(BaseModel):
    """
    Input model for research queries with validation.
    
    This model validates and structures the input for research requests,
    ensuring all necessary parameters are provided and valid.
    """
    
    idea: str = Field(
        ..., 
        min_length=3, 
        max_length=500,
        description="The main topic or idea to research. Should be specific enough to yield meaningful results but not overly narrow."
    )
    
    research_depth: ResearchDepth = Field(
        default=ResearchDepth.QUICK,
        description="The depth of research to perform. 'quick' for basic search results, 'deep' for comprehensive analysis including web scraping."
    )
    
    max_sources: PositiveInt = Field(
        default=5,
        le=20,
        description="Maximum number of sources to analyze. Limited to 20 to prevent excessive processing time."
    )
    
    target_audience: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Specific audience for the content (e.g., 'business professionals', 'tech enthusiasts', 'general public')."
    )
    
    content_type: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Type of content to create (e.g., 'educational', 'promotional', 'news', 'opinion')."
    )
    
    class Config:
        """Pydantic configuration for the ResearchQuery model."""
        use_enum_values = True
        schema_extra = {
            "example": {
                "idea": "artificial intelligence impact on healthcare",
                "research_depth": "deep",
                "max_sources": 10,
                "target_audience": "healthcare professionals",
                "content_type": "educational"
            }
        }


class Citation(BaseModel):
    """
    Model for source citations with detailed metadata.
    
    Tracks the source of information for fact verification and attribution.
    """
    
    url: str = Field(
        ...,
        description="The complete URL of the source. Must be a valid HTTP/HTTPS URL."
    )
    
    title: str = Field(
        ...,
        max_length=200,
        description="The title of the source article or page. Used for human-readable source identification."
    )
    
    domain: str = Field(
        ...,
        max_length=100,
        description="The domain name of the source (e.g., 'cnn.com', 'nature.com'). Used for source credibility assessment."
    )
    
    access_date: datetime = Field(
        default_factory=datetime.now,
        description="The date and time when the source was accessed. Important for tracking information freshness."
    )
    
    credibility_score: confloat(ge=0.0, le=1.0) = Field( # type: ignore
        default=0.5,
        description="Automated credibility score from 0.0 to 1.0, where 1.0 is most credible. Based on domain authority and source type."
    )
    
    class Config:
        """Pydantic configuration for the Citation model."""
        schema_extra = {
            "example": {
                "url": "https://www.nature.com/articles/s41586-021-03819-2",
                "title": "AI breakthrough in protein folding prediction",
                "domain": "nature.com",
                "credibility_score": 0.95
            }
        }


class Fact(BaseModel):
    """
    Model for individual factual information with verification.
    
    Represents a single piece of verifiable information extracted from research.
    """
    
    statement: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The factual statement or claim. Should be specific, verifiable, and clearly articulated."
    )
    
    fact_type: Literal["statistic", "definition", "quote", "date", "event", "trend"] = Field(
        ...,
        description="Type of fact to help categorize and present information appropriately."
    )
    
    confidence_score: confloat(ge=0.0, le=1.0) = Field( # type: ignore
        ...,
        description="Confidence level in the accuracy of this fact, from 0.0 to 1.0. Based on source credibility and cross-referencing."
    )
    
    source: Citation = Field(
        ...,
        description="The source citation for this fact. Essential for verification and attribution."
    )
    
    extracted_date: datetime = Field(
        default_factory=datetime.now,
        description="When this fact was extracted from the source. Helps track information freshness."
    )
    
    supporting_evidence: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Additional context or supporting information that validates this fact."
    )
    
    contradictory_info: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Any contradictory information found from other sources. Important for fact-checking."
    )
    
    class Config:
        """Pydantic configuration for the Fact model."""
        schema_extra = {
            "example": {
                "statement": "AI models like GPT-4 can process over 25,000 words of context",
                "fact_type": "statistic",
                "confidence_score": 0.9,
                "supporting_evidence": "Based on official OpenAI documentation",
                "source": {
                    "url": "https://openai.com/research/gpt-4",
                    "title": "GPT-4 Technical Report",
                    "domain": "openai.com",
                    "credibility_score": 0.95
                }
            }
        }


class StatisticalData(BaseModel):
    """
    Model for numerical and statistical information.
    
    Specialized model for handling quantitative data with proper validation.
    """
    
    metric_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Name of the metric or statistic (e.g., 'Market Size', 'Growth Rate', 'User Count')."
    )
    
    value: Union[int, float] = Field(
        ...,
        description="The numerical value of the statistic. Can be integer or decimal."
    )
    
    unit: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Unit of measurement (e.g., 'million', 'percent', 'billion USD', 'users')."
    )
    
    time_period: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Time period for this statistic (e.g., '2024', 'Q1 2025', 'annually')."
    )
    
    source: Citation = Field(
        ...,
        description="Source citation for this statistical data. Critical for data verification."
    )
    
    context: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Additional context that helps interpret this statistic correctly."
    )
    
    def validate_metric_name(cls, v):
        """Ensure metric name is descriptive."""
        if v.lower() in ['number', 'value', 'amount']:
            raise ValueError('Metric name must be more descriptive than generic terms')
        return v.strip()
    
    class Config:
        """Pydantic configuration for the StatisticalData model."""
        schema_extra = {
            "example": {
                "metric_name": "Global AI Market Size",
                "value": 428.0,
                "unit": "billion USD",
                "time_period": "2024",
                "context": "Expected to reach $2.02 trillion by 2030",
                "source": {
                    "url": "https://www.marketresearch.com/ai-report",
                    "title": "AI Market Analysis 2024",
                    "domain": "marketresearch.com",
                    "credibility_score": 0.8
                }
            }
        }


class TrendingTopic(BaseModel):
    """
    Model for trending topics and hashtags.
    
    Represents current trends related to the research topic.
    """
    
    topic: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="The trending topic or hashtag. Should be relevant to the research query."
    )
    
    relevance_score: confloat(ge=0.0, le=1.0) = Field( # type: ignore
        ...,
        description="How relevant this topic is to the research query, from 0.0 to 1.0."
    )
    
    trend_direction: Literal["rising", "stable", "declining"] = Field(
        ...,
        description="Direction of the trend - whether it's gaining, maintaining, or losing momentum."
    )
    
    associated_hashtags: List[str] = Field(
        default_factory=list,
        max_items=10,
        description="Related hashtags for social media use. Each hashtag should start with #."
    )
    
    def validate_hashtags(cls, v):
        """Ensure hashtags are properly formatted."""
        validated_hashtags = []
        for hashtag in v:
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"
            # Remove spaces and special characters except #
            hashtag = ''.join(c for c in hashtag if c.isalnum() or c == '#')
            validated_hashtags.append(hashtag)
        return validated_hashtags
    
    class Config:
        """Pydantic configuration for the TrendingTopic model."""
        schema_extra = {
            "example": {
                "topic": "Artificial Intelligence Ethics",
                "relevance_score": 0.9,
                "trend_direction": "rising",
                "associated_hashtags": ["#AIEthics", "#ResponsibleAI", "#TechEthics"]
            }
        }


class ContentAngle(BaseModel):
    """
    Model for content angles and perspectives.
    
    Represents different angles or perspectives for creating content.
    """
    
    angle_title: str = Field(
        ...,
        min_length=5,
        max_length=100,
        description="Compelling title for this content angle. Should be attention-grabbing and specific."
    )
    
    description: str = Field(
        ...,
        min_length=20,
        max_length=300,
        description="Detailed description of this content angle, including key points to cover."
    )
    
    target_platform: List[Literal["twitter", "linkedin", "instagram", "facebook", "tiktok"]] = Field(
        default_factory=list,
        description="Social media platforms where this angle would be most effective."
    )
    
    engagement_potential: Literal["high", "medium", "low"] = Field(
        ...,
        description="Predicted engagement potential based on topic relevance and current trends."
    )
    
    supporting_facts: List[str] = Field(
        default_factory=list,
        max_items=5,
        description="Key facts or statistics that support this content angle."
    )
    
    class Config:
        """Pydantic configuration for the ContentAngle model."""
        schema_extra = {
            "example": {
                "angle_title": "Why AI is Revolutionizing Healthcare Diagnosis",
                "description": "Focus on specific AI applications in medical diagnosis, including success rates and real-world examples.",
                "target_platform": ["linkedin", "twitter"],
                "engagement_potential": "high",
                "supporting_facts": ["AI diagnostic accuracy exceeds 90% in certain conditions", "Reduced diagnosis time by 50%"]
            }
        }


class ResearchMetadata(BaseModel):
    """
    Model for research metadata and processing information.
    
    Tracks the research process and provides transparency about data collection.
    """
    
    research_date: datetime = Field(
        default_factory=datetime.now,
        description="When the research was conducted. Important for data freshness assessment."
    )
    
    processing_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time taken to complete the research in seconds. Useful for performance monitoring."
    )
    
    sources_processed: PositiveInt = Field(
        ...,
        description="Number of sources that were successfully processed during research."
    )
    
    sources_failed: int = Field(
        default=0,
        ge=0,
        description="Number of sources that failed to process. Important for assessing data completeness."
    )
    
    search_queries_used: List[str] = Field(
        default_factory=list,
        description="List of search queries used during research. Provides transparency about search strategy."
    )
    
    confidence_level: confloat(ge=0.0, le=1.0) = Field( # type: ignore
        ...,
        description="Overall confidence level in the research results based on source quality and data consistency."
    )
    
    limitations: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Any limitations or caveats about the research results that users should be aware of."
    )
    
    class Config:
        """Pydantic configuration for the ResearchMetadata model."""
        schema_extra = {
            "example": {
                "processing_time": 15.5,
                "sources_processed": 8,
                "sources_failed": 2,
                "search_queries_used": ["AI healthcare", "medical AI diagnosis", "healthcare AI statistics"],
                "confidence_level": 0.85,
                "limitations": "Limited to English-language sources published in the last 2 years"
            }
        }


class ResearchOutput(BaseModel):
    """
    Comprehensive output model for research results.
    
    This is the main output model that contains all research findings,
    structured for easy consumption and further processing.
    """
    
    query: ResearchQuery = Field(
        ...,
        description="The original research query that generated these results."
    )
    
    summary: str = Field(
        ...,
        min_length=50,
        max_length=1000,
        description="Executive summary of the research findings. Should provide a clear overview of key discoveries."
    )
    
    facts: List[Fact] = Field(
        default_factory=list,
        description="List of verified facts extracted from research. Each fact includes source attribution and confidence scores."
    )
    
    statistics: List[StatisticalData] = Field(
        default_factory=list,
        description="Numerical data and statistics found during research. Includes context and source validation."
    )
    
    trending_topics: List[TrendingTopic] = Field(
        default_factory=list,
        description="Current trending topics related to the research query. Useful for social media content strategy."
    )
    
    content_angles: List[ContentAngle] = Field(
        default_factory=list,
        description="Different perspectives and angles for creating content based on the research findings."
    )
    
    key_insights: List[str] = Field(
        default_factory=list,
        max_items=10,
        description="Major insights and takeaways from the research. Should be actionable and significant."
    )
    
    sources: List[Citation] = Field(
        default_factory=list,
        description="All sources used in the research with credibility scores and access information."
    )
    
    metadata: ResearchMetadata = Field(
        ...,
        description="Information about the research process, including timing, success rates, and limitations."
    )
    
    def validate_facts_quality(cls, v):
        """Ensure facts meet minimum quality standards."""
        if len(v) > 0:
            # Check that at least some facts have high confidence
            high_confidence_facts = [f for f in v if f.confidence_score >= 0.7]
            if len(high_confidence_facts) == 0:
                raise ValueError('At least one fact should have confidence score >= 0.7')
        return v
    
    def validate_research_consistency(cls, values):
        """Validate that the research output is internally consistent."""
        facts = values.get('facts', [])
        statistics = values.get('statistics', [])
        sources = values.get('sources', [])
        
        # Ensure we have some substantial content
        if len(facts) == 0 and len(statistics) == 0:
            raise ValueError('Research output must contain at least one fact or statistic')
        
        # Ensure sources are provided for facts
        if len(facts) > 0 and len(sources) == 0:
            raise ValueError('Sources must be provided when facts are present')
        
        return values
    
    def to_dict(self) -> Dict:
        """Convert the research output to a dictionary for serialization."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convert the research output to JSON string."""
        return self.json(indent=2)
    
    class Config:
        """Pydantic configuration for the ResearchOutput model."""
        schema_extra = {
            "example": {
                "query": {
                    "idea": "artificial intelligence in healthcare",
                    "research_depth": "deep",
                    "max_sources": 10
                },
                "summary": "AI is transforming healthcare through improved diagnostics, personalized treatment, and operational efficiency.",
                "facts": [
                    {
                        "statement": "AI diagnostic systems achieve 94% accuracy in detecting skin cancer",
                        "fact_type": "statistic",
                        "confidence_score": 0.9,
                        "source": {
                            "url": "https://www.nature.com/articles/nature21056",
                            "title": "Dermatologist-level classification of skin cancer",
                            "domain": "nature.com",
                            "credibility_score": 0.95
                        }
                    }
                ],
                "key_insights": [
                    "AI is most effective in pattern recognition tasks like medical imaging",
                    "Healthcare AI adoption is accelerating due to proven ROI"
                ]
            }
        }


class ErrorResponse(BaseModel):
    """
    Model for error responses with detailed information.
    
    Provides structured error information for debugging and user feedback.
    """
    
    error_type: str = Field(
        ...,
        description="Type of error that occurred (e.g., 'ValidationError', 'NetworkError', 'ProcessingError')."
    )
    
    error_message: str = Field(
        ...,
        description="Human-readable error message explaining what went wrong."
    )
    
    error_details: Optional[Dict] = Field(
        default=None,
        description="Additional technical details about the error for debugging purposes."
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the error occurred."
    )
    
    query_context: Optional[str] = Field(
        default=None,
        description="The research query or context that caused the error."
    )
    
    class Config:
        """Pydantic configuration for the ErrorResponse model."""
        schema_extra = {
            "example": {
                "error_type": "ValidationError",
                "error_message": "Research query must be at least 3 characters long",
                "error_details": {"field": "idea", "rejected_value": "AI"},
                "query_context": "AI"
            }
        }




class ContentAnalyzer:
    """Analyze content for social media insights - Single Responsibility Principle"""
    
    def extract_trending_topics(self, content: str) -> List[str]:
        """Extract trending topics from content"""
        # Simple keyword extraction - in production, use NLP libraries
        keywords = ["AI", "technology", "innovation", "digital", "future", "trends"]
        found_topics = []
        
        content_lower = content.lower()
        for keyword in keywords:
            if keyword.lower() in content_lower:
                found_topics.append(f"#{keyword}")
        
        return found_topics[:5]  # Return top 5
    
    def generate_content_angles(self, topic: str, content: str) -> List[str]:
        """Generate content angles for social media"""
        angles = [
            f"Breaking: Latest developments in {topic}",
            f"Why {topic} matters for your business",
            f"The future of {topic} - what experts say",
            f"Top 5 things to know about {topic}",
            f"How {topic} is changing the industry"
        ]
        return angles[:3]  # Return top 3
    
    def extract_statistics(self, content: str) -> List[str]:
        """Extract key statistics from content"""
        # Simple pattern matching for numbers/percentages
        import re
        stats_pattern = r'\b\d+(?:\.\d+)?%?\b'
        matches = re.findall(stats_pattern, content)
        
        # Format as statistics
        stats = []
        for match in matches[:3]:  # Top 3 stats
            stats.append(f"Key stat: {match}")
        
        return stats

