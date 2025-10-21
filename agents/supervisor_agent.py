"""
Supervisor Agent (Orchestrator) - Routes queries between research and writing teams
"""
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from agents.base_agent import BaseAgent
from models import ReportMode, UserQuestion


class QueryType(str, Enum):
    """Types of queries the supervisor can route"""
    ANOMALY_INVESTIGATION = "anomaly_investigation"
    LEADERSHIP_SUMMARY = "leadership_summary"
    PATTERN_ANALYSIS = "pattern_analysis"
    INCIDENT_ANALYSIS = "incident_analysis"
    DAILY_SUMMARY = "daily_summary"
    DEEP_RESEARCH = "deep_research"


class TeamType(str, Enum):
    """Types of teams in the system"""
    RESEARCH = "research"
    WRITING = "writing"
    EDITING = "editing"
    EVALUATION = "evaluation"


class SupervisorAgent(BaseAgent):
    """Supervisor Agent that orchestrates the entire analysis workflow"""
    
    def __init__(self):
        super().__init__("supervisor")
        self.logger = logging.getLogger("traffix.supervisor")
        self.team_assignments = {}
        self.workflow_state = {}
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute supervisor orchestration logic"""
        user_query = input_data.get("user_question", "")
        location = input_data.get("location", "Unknown")
        mode = input_data.get("mode", "quick")
        
        self.logger.info(f"Supervisor orchestrating analysis for {location}")
        
        try:
            # Step 1: Analyze and classify the query
            query_analysis = await self._analyze_query(user_query, location, mode)
            
            # Step 2: Create work plan
            work_plan = await self._create_work_plan(query_analysis)
            
            # Step 3: Assign tasks to teams
            team_assignments = await self._assign_tasks_to_teams(work_plan)
            
            # Step 4: Monitor execution
            execution_results = await self._monitor_execution(team_assignments)
            
            # Step 5: Synthesize final results
            final_results = await self._synthesize_results(execution_results)
            
            return {
                "supervisor_analysis": query_analysis,
                "work_plan": work_plan,
                "team_assignments": team_assignments,
                "execution_results": execution_results,
                "final_results": final_results,
                "orchestration_metadata": {
                    "location": location,
                    "mode": mode,
                    "query_type": query_analysis.get("query_type"),
                    "complexity": query_analysis.get("complexity"),
                    "estimated_duration": work_plan.get("estimated_duration"),
                    "teams_involved": list(team_assignments.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Supervisor orchestration failed: {e}")
            raise
    
    async def _analyze_query(self, user_query: str, location: str, mode: str) -> Dict[str, Any]:
        """Analyze the user query to determine requirements and complexity"""
        
        # Determine query type based on content and mode
        query_type = self._classify_query_type(user_query, mode)
        
        # Assess complexity
        complexity = self._assess_complexity(user_query, mode)
        
        # Determine required data sources
        required_sources = self._determine_required_sources(query_type, complexity)
        
        # Estimate processing requirements
        processing_requirements = self._estimate_processing_requirements(query_type, complexity)
        
        return {
            "query_type": query_type,
            "complexity": complexity,
            "required_sources": required_sources,
            "processing_requirements": processing_requirements,
            "user_intent": self._extract_user_intent(user_query),
            "location": location,
            "mode": mode
        }
    
    async def _create_work_plan(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed work plan based on query analysis"""
        
        query_type = query_analysis["query_type"]
        complexity = query_analysis["complexity"]
        
        # Define work phases
        phases = []
        
        # Phase 1: Research
        research_phase = {
            "phase": "research",
            "team": TeamType.RESEARCH,
            "tasks": self._define_research_tasks(query_type, complexity),
            "dependencies": [],
            "estimated_duration": self._estimate_research_duration(query_type, complexity)
        }
        phases.append(research_phase)
        
        # Phase 2: Writing
        writing_phase = {
            "phase": "writing",
            "team": TeamType.WRITING,
            "tasks": self._define_writing_tasks(query_type, complexity),
            "dependencies": ["research"],
            "estimated_duration": self._estimate_writing_duration(query_type, complexity)
        }
        phases.append(writing_phase)
        
        # Phase 3: Editing (for complex queries)
        if complexity in ["high", "critical"]:
            editing_phase = {
                "phase": "editing",
                "team": TeamType.EDITING,
                "tasks": self._define_editing_tasks(query_type, complexity),
                "dependencies": ["writing"],
                "estimated_duration": self._estimate_editing_duration(query_type, complexity)
            }
            phases.append(editing_phase)
        
        # Phase 4: Evaluation (for all queries)
        evaluation_phase = {
            "phase": "evaluation",
            "team": TeamType.EVALUATION,
            "tasks": self._define_evaluation_tasks(query_type, complexity),
            "dependencies": ["writing", "editing"] if complexity in ["high", "critical"] else ["writing"],
            "estimated_duration": self._estimate_evaluation_duration(query_type, complexity)
        }
        phases.append(evaluation_phase)
        
        # Calculate total duration
        total_duration = sum(phase["estimated_duration"] for phase in phases)
        
        return {
            "phases": phases,
            "total_phases": len(phases),
            "estimated_duration": total_duration,
            "parallel_execution": self._can_parallelize(phases),
            "critical_path": self._identify_critical_path(phases)
        }
    
    async def _assign_tasks_to_teams(self, work_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assign specific tasks to teams based on work plan"""
        
        team_assignments = {}
        
        for phase in work_plan["phases"]:
            team = phase["team"]
            tasks = phase["tasks"]
            
            if team not in team_assignments:
                team_assignments[team] = {
                    "team_type": team,
                    "assigned_tasks": [],
                    "dependencies": phase["dependencies"],
                    "estimated_duration": phase["estimated_duration"],
                    "priority": self._calculate_team_priority(team, work_plan)
                }
            
            team_assignments[team]["assigned_tasks"].extend(tasks)
        
        return team_assignments
    
    async def _monitor_execution(self, team_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the execution of team assignments"""
        
        execution_results = {}
        
        for team, assignment in team_assignments.items():
            self.logger.info(f"Monitoring {team} team execution")
            
            # Simulate team execution monitoring
            execution_results[team] = {
                "status": "completed",  # In real implementation, this would be dynamic
                "tasks_completed": len(assignment["assigned_tasks"]),
                "execution_time": assignment["estimated_duration"],
                "quality_score": self._calculate_team_quality_score(team),
                "outputs": self._generate_team_outputs(team, assignment)
            }
        
        return execution_results
    
    async def _synthesize_results(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from all teams into final output"""
        
        # Collect outputs from all teams
        research_output = execution_results.get(TeamType.RESEARCH, {}).get("outputs", {})
        writing_output = execution_results.get(TeamType.WRITING, {}).get("outputs", {})
        editing_output = execution_results.get(TeamType.EDITING, {}).get("outputs", {})
        evaluation_output = execution_results.get(TeamType.EVALUATION, {}).get("outputs", {})
        
        # Synthesize final results
        final_results = {
            "research_findings": research_output,
            "narrative_content": writing_output,
            "edited_content": editing_output if editing_output else writing_output,
            "quality_assessment": evaluation_output,
            "synthesis_metadata": {
                "teams_involved": list(execution_results.keys()),
                "overall_quality": self._calculate_overall_quality(execution_results),
                "synthesis_timestamp": datetime.now().isoformat()
            }
        }
        
        return final_results
    
    def _classify_query_type(self, user_query: str, mode: str) -> QueryType:
        """Classify the type of query based on content and mode"""
        
        query_lower = user_query.lower()
        
        if "why" in query_lower and ("congestion" in query_lower or "traffic" in query_lower):
            return QueryType.ANOMALY_INVESTIGATION
        elif "summarize" in query_lower or "highlights" in query_lower:
            return QueryType.LEADERSHIP_SUMMARY
        elif "pattern" in query_lower or "recurring" in query_lower:
            return QueryType.PATTERN_ANALYSIS
        elif "incident" in query_lower or "accident" in query_lower:
            return QueryType.INCIDENT_ANALYSIS
        elif mode == "deep":
            return QueryType.DEEP_RESEARCH
        else:
            return QueryType.DAILY_SUMMARY
    
    def _assess_complexity(self, user_query: str, mode: str) -> str:
        """Assess the complexity of the query"""
        
        complexity_indicators = {
            "low": ["simple", "quick", "basic", "overview"],
            "medium": ["detailed", "analysis", "investigate", "explain"],
            "high": ["comprehensive", "thorough", "deep", "complex", "multiple"],
            "critical": ["urgent", "critical", "emergency", "immediate"]
        }
        
        query_lower = user_query.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return complexity
        
        # Default based on mode
        if mode == "deep":
            return "high"
        elif mode == "quick":
            return "low"
        else:
            return "medium"
    
    def _determine_required_sources(self, query_type: QueryType, complexity: str) -> List[str]:
        """Determine required data sources based on query type and complexity"""
        
        base_sources = ["ritis", "news"]
        
        if query_type in [QueryType.ANOMALY_INVESTIGATION, QueryType.INCIDENT_ANALYSIS]:
            base_sources.extend(["incidents", "weather"])
        
        if query_type == QueryType.PATTERN_ANALYSIS:
            base_sources.extend(["incidents", "weather", "social"])
        
        if complexity in ["high", "critical"]:
            base_sources.extend(["weather", "social", "historical"])
        
        return list(set(base_sources))
    
    def _estimate_processing_requirements(self, query_type: QueryType, complexity: str) -> Dict[str, Any]:
        """Estimate processing requirements"""
        
        base_time = {
            QueryType.DAILY_SUMMARY: 30,
            QueryType.ANOMALY_INVESTIGATION: 60,
            QueryType.LEADERSHIP_SUMMARY: 90,
            QueryType.PATTERN_ANALYSIS: 120,
            QueryType.INCIDENT_ANALYSIS: 45,
            QueryType.DEEP_RESEARCH: 180
        }
        
        complexity_multiplier = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.0,
            "critical": 2.5
        }
        
        estimated_time = base_time.get(query_type, 60) * complexity_multiplier.get(complexity, 1.0)
        
        return {
            "estimated_seconds": estimated_time,
            "memory_requirements": "medium" if complexity in ["low", "medium"] else "high",
            "cpu_intensity": "low" if complexity == "low" else "high"
        }
    
    def _extract_user_intent(self, user_query: str) -> str:
        """Extract the user's intent from the query"""
        
        intent_patterns = {
            "investigate": ["why", "what caused", "investigate", "analyze"],
            "summarize": ["summarize", "overview", "highlights", "summary"],
            "explain": ["explain", "how", "describe", "tell me about"],
            "predict": ["predict", "forecast", "what will happen", "trends"],
            "recommend": ["recommend", "suggest", "what should", "advice"]
        }
        
        query_lower = user_query.lower()
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return "investigate"  # Default intent
    
    def _define_research_tasks(self, query_type: QueryType, complexity: str) -> List[Dict[str, Any]]:
        """Define tasks for the research team"""
        
        base_tasks = [
            {"task": "collect_traffic_data", "description": "Gather RITIS traffic data"},
            {"task": "collect_news_data", "description": "Gather relevant news articles"},
            {"task": "identify_incidents", "description": "Extract key incidents and events"}
        ]
        
        if query_type in [QueryType.ANOMALY_INVESTIGATION, QueryType.INCIDENT_ANALYSIS]:
            base_tasks.append({"task": "analyze_causes", "description": "Identify root causes of issues"})
        
        if query_type == QueryType.PATTERN_ANALYSIS:
            base_tasks.append({"task": "pattern_recognition", "description": "Identify recurring patterns"})
        
        if complexity in ["high", "critical"]:
            base_tasks.extend([
                {"task": "historical_analysis", "description": "Analyze historical trends"},
                {"task": "cross_reference", "description": "Cross-reference multiple data sources"}
            ])
        
        return base_tasks
    
    def _define_writing_tasks(self, query_type: QueryType, complexity: str) -> List[Dict[str, Any]]:
        """Define tasks for the writing team"""
        
        base_tasks = [
            {"task": "create_narrative", "description": "Synthesize data into coherent narrative"},
            {"task": "write_summary", "description": "Create executive summary"}
        ]
        
        if query_type == QueryType.LEADERSHIP_SUMMARY:
            base_tasks.append({"task": "format_for_leadership", "description": "Format content for executive consumption"})
        
        if complexity in ["high", "critical"]:
            base_tasks.extend([
                {"task": "detailed_analysis", "description": "Create detailed analysis sections"},
                {"task": "supporting_evidence", "description": "Include comprehensive supporting evidence"}
            ])
        
        return base_tasks
    
    def _define_editing_tasks(self, query_type: QueryType, complexity: str) -> List[Dict[str, Any]]:
        """Define tasks for the editing team"""
        
        return [
            {"task": "fact_check", "description": "Verify factual accuracy"},
            {"task": "improve_readability", "description": "Enhance readability and flow"},
            {"task": "tone_adjustment", "description": "Ensure empathetic and professional tone"},
            {"task": "consistency_check", "description": "Check for consistency and coherence"}
        ]
    
    def _define_evaluation_tasks(self, query_type: QueryType, complexity: str) -> List[Dict[str, Any]]:
        """Define tasks for the evaluation team"""
        
        return [
            {"task": "quality_assessment", "description": "Assess overall quality using RAGAS metrics"},
            {"task": "relevancy_check", "description": "Verify content relevancy"},
            {"task": "faithfulness_check", "description": "Check faithfulness to source data"},
            {"task": "final_review", "description": "Conduct final quality review"}
        ]
    
    def _estimate_research_duration(self, query_type: QueryType, complexity: str) -> float:
        """Estimate research phase duration in seconds"""
        base_times = {
            QueryType.DAILY_SUMMARY: 20,
            QueryType.ANOMALY_INVESTIGATION: 40,
            QueryType.LEADERSHIP_SUMMARY: 30,
            QueryType.PATTERN_ANALYSIS: 60,
            QueryType.INCIDENT_ANALYSIS: 25,
            QueryType.DEEP_RESEARCH: 90
        }
        
        complexity_multiplier = {"low": 1.0, "medium": 1.2, "high": 1.5, "critical": 2.0}
        return base_times.get(query_type, 30) * complexity_multiplier.get(complexity, 1.0)
    
    def _estimate_writing_duration(self, query_type: QueryType, complexity: str) -> float:
        """Estimate writing phase duration in seconds"""
        base_times = {
            QueryType.DAILY_SUMMARY: 15,
            QueryType.ANOMALY_INVESTIGATION: 25,
            QueryType.LEADERSHIP_SUMMARY: 20,
            QueryType.PATTERN_ANALYSIS: 35,
            QueryType.INCIDENT_ANALYSIS: 20,
            QueryType.DEEP_RESEARCH: 60
        }
        
        complexity_multiplier = {"low": 1.0, "medium": 1.3, "high": 1.8, "critical": 2.2}
        return base_times.get(query_type, 20) * complexity_multiplier.get(complexity, 1.0)
    
    def _estimate_editing_duration(self, query_type: QueryType, complexity: str) -> float:
        """Estimate editing phase duration in seconds"""
        return 15 * (2 if complexity == "critical" else 1.5 if complexity == "high" else 1.0)
    
    def _estimate_evaluation_duration(self, query_type: QueryType, complexity: str) -> float:
        """Estimate evaluation phase duration in seconds"""
        return 10 * (1.5 if complexity in ["high", "critical"] else 1.0)
    
    def _can_parallelize(self, phases: List[Dict[str, Any]]) -> bool:
        """Determine if phases can be parallelized"""
        # Research and some evaluation tasks can run in parallel
        return True
    
    def _identify_critical_path(self, phases: List[Dict[str, Any]]) -> List[str]:
        """Identify the critical path through phases"""
        return [phase["phase"] for phase in phases]
    
    def _calculate_team_priority(self, team: TeamType, work_plan: Dict[str, Any]) -> str:
        """Calculate priority for team assignments"""
        if team == TeamType.RESEARCH:
            return "high"
        elif team == TeamType.WRITING:
            return "high"
        elif team == TeamType.EDITING:
            return "medium"
        else:  # EVALUATION
            return "low"
    
    def _calculate_team_quality_score(self, team: TeamType) -> float:
        """Calculate quality score for team execution"""
        # In real implementation, this would be based on actual performance metrics
        base_scores = {
            TeamType.RESEARCH: 0.85,
            TeamType.WRITING: 0.90,
            TeamType.EDITING: 0.88,
            TeamType.EVALUATION: 0.92
        }
        return base_scores.get(team, 0.80)
    
    def _generate_team_outputs(self, team: TeamType, assignment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock outputs for team execution"""
        # In real implementation, this would return actual team outputs
        return {
            "team": team,
            "tasks_completed": len(assignment["assigned_tasks"]),
            "output_quality": self._calculate_team_quality_score(team),
            "output_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_overall_quality(self, execution_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from all team results"""
        quality_scores = [result.get("quality_score", 0.8) for result in execution_results.values()]
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.8
