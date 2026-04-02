from typing import List, Optional

from app.agents.creative_agent import (
    CandidateWall,
    CreativeAgent,
    CreativeIntentPlan,
    ExpandedQuery,
    ResourceRecommendation,
)
from app.repositories.search_repository import SearchRepository


class AgentToolsService:
    def __init__(self, creative_agent: CreativeAgent, search_repo: SearchRepository):
        self.creative_agent = creative_agent
        self.search_repo = search_repo

    def analyze_intent(
        self, user_intent: str, clarification_closed: bool = False
    ) -> CreativeIntentPlan:
        return self.creative_agent.analyze_intent(
            user_intent=user_intent,
            clarification_closed=clarification_closed,
        )

    def recommend_resources(self, plan: CreativeIntentPlan) -> ResourceRecommendation:
        resources = self.creative_agent.load_resources()
        return self.creative_agent.recommend_resources(plan=plan, resources=resources)

    def build_axis_expansions(
        self,
        user_intent: str,
        plan: CreativeIntentPlan,
        recommendation: Optional[ResourceRecommendation] = None,
        previous_expansions: Optional[List[ExpandedQuery]] = None,
        force_refresh: bool = False,
    ) -> List[ExpandedQuery]:
        resources = self.creative_agent.load_resources()
        return self.creative_agent.build_axis_expansions(
            user_intent=user_intent,
            plan=plan,
            resources=resources,
            recommendation=recommendation,
            previous_expansions=previous_expansions,
            force_refresh=force_refresh,
        )

    def build_candidate_wall(
        self,
        expansions: List[ExpandedQuery],
        per_query_k: int = 2,
        top_k: int = 12,
    ) -> CandidateWall:
        return self.creative_agent.build_candidate_wall(
            search_engine=self.search_repo.search_engine,
            expansions=expansions,
            per_query_k=per_query_k,
            top_k=top_k,
        )

    def describe_wall(self, wall: CandidateWall) -> str:
        return self.creative_agent.describe_wall(wall)
