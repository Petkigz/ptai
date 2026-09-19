"""
Source Quality Engine - assesses source credibility, novelty, corroboration
"""
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SourceAssessment:
    url: str
    credibility: float
    novelty: float
    corroborated: bool
    recency: float
    overall_quality: float


class SourceQualityEngine:
    def __init__(self):
        self.credibility_map = {
            "reuters": 0.95,
            "apnews": 0.95,
            "bbc": 0.9,
            "bloomberg": 0.9,
            "official": 0.95,
            "gov": 0.9,
            "twitter": 0.5,
            "x.com": 0.5,
            "random_blog": 0.3
        }

    def assess(self, url: str, content: str, other_sources: List[str] = None) -> SourceAssessment:
        credibility = 0.5
        for domain, cred in self.credibility_map.items():
            if domain in url.lower():
                credibility = cred
                break
        
        # Novelty - is content new?
        novelty = 0.6  # placeholder
        
        # Corroborated?
        corroborated = False
        if other_sources:
            # Check if other sources mention same info
            corroborated = len(other_sources) >= 2
        
        # Recency
        recency = 0.7
        
        overall = (credibility * 0.4 + novelty * 0.2 + (1.0 if corroborated else 0.3) * 0.2 + recency * 0.2)
        
        return SourceAssessment(
            url=url,
            credibility=credibility,
            novelty=novelty,
            corroborated=corroborated,
            recency=recency,
            overall_quality=overall
        )
