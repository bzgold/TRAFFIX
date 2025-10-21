"""
LangGraph Workflow for Traffix Multi-Agent System
Consolidated and improved version
"""
import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from agents import SupervisorAgent, ResearchAgent, WriterAgent, EditorAgent, EvaluatorAgent
from models import ReportMode, AgentStatus


class TraffixState(TypedDict):
    """State schema for Traffix workflow"""
    messages: List[Any]
    user_query: str
    location: str
    mode: str
    query_analysis: Dict[str, Any]
    work_plan: Dict[str, Any]
    research_data: Dict[str, Any]
    narrative_content: Dict[str, Any]
    edited_content: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    final_output: Dict[str, Any]
    current_agent: str
    workflow_state: str
    errors: List[str]


class TraffixWorkflow:
    """LangGraph workflow orchestrating the Traffix multi-agent system"""
    
    def __init__(self, vector_service=None):
        self.logger = logging.getLogger("traffix.workflow")
        self.vector_service = vector_service
        self.supervisor = SupervisorAgent()
        self.research = ResearchAgent(vector_service=vector_service)
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.evaluator = EvaluatorAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the state schema using TypedDict
        workflow = StateGraph(TraffixState)
        
        # Add nodes for each agent
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("editor", self._editor_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define the workflow edges
        workflow.set_entry_point("supervisor")
        
        # Supervisor routes to research
        workflow.add_edge("supervisor", "research")
        
        # Research routes to writer
        workflow.add_edge("research", "writer")
        
        # Writer routes to editor (for complex queries) or evaluator
        workflow.add_conditional_edges(
            "writer",
            self._should_edit,
            {
                "edit": "editor",
                "evaluate": "evaluator"
            }
        )
        
        # Editor routes to evaluator
        workflow.add_edge("editor", "evaluator")
        
        # Evaluator routes to finalize
        workflow.add_edge("evaluator", "finalize")
        
        # Finalize ends the workflow
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Supervisor agent node"""
        
        self.logger.info("Executing supervisor node")
        
        try:
            # Extract user query and context
            user_query = state.get("user_query", "")
            location = state.get("location", "Unknown")
            mode = state.get("mode", "quick")
            
            # Prepare input for supervisor
            supervisor_input = {
                "user_question": user_query,
                "location": location,
                "mode": mode
            }
            
            # Execute supervisor task
            supervisor_task = await self.supervisor.execute_task(supervisor_input)
            
            if supervisor_task.status != AgentStatus.COMPLETED:
                raise Exception(f"Supervisor task failed: {supervisor_task.error_message}")
            
            # Extract results
            supervisor_output = supervisor_task.output_data
            query_analysis = supervisor_output.get("supervisor_analysis", {})
            work_plan = supervisor_output.get("work_plan", {})
            
            # Update state
            new_state = {
                "query_analysis": query_analysis,
                "work_plan": work_plan,
                "current_agent": "supervisor",
                "workflow_state": "supervisor_completed"
            }
            
            # Add message
            new_state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Supervisor analysis completed for {location}")
            ]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Supervisor node failed: {e}")
            return {
                "errors": state.get("errors", []) + [str(e)],
                "workflow_state": "error"
            }
    
    async def _research_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Research agent node"""
        
        self.logger.info("Executing research node")
        
        try:
            # Extract context from supervisor
            query_analysis = state.get("query_analysis", {})
            location = state.get("location", "Unknown")
            
            # Prepare input for research agent
            research_input = {
                "location": location,
                "query_type": query_analysis.get("query_type", "daily_summary"),
                "complexity": query_analysis.get("complexity", "medium"),
                "required_sources": query_analysis.get("required_sources", ["ritis", "news"])
            }
            
            # Execute research task
            research_task = await self.research.execute_task(research_input)
            
            if research_task.status != AgentStatus.COMPLETED:
                raise Exception(f"Research task failed: {research_task.error_message}")
            
            # Extract results
            research_output = research_task.output_data
            research_data = {
                "collected_data": research_output.get("collected_data", {}),
                "key_incidents": research_output.get("key_incidents", []),
                "cause_analysis": research_output.get("cause_analysis", {}),
                "pattern_analysis": research_output.get("pattern_analysis", {}),
                "research_insights": research_output.get("research_insights", {})
            }
            
            # Update state
            new_state = {
                "research_data": research_data,
                "current_agent": "research",
                "workflow_state": "research_completed"
            }
            
            # Add message
            new_state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Research completed for {location} - {len(research_data.get('key_incidents', []))} incidents found")
            ]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Research node failed: {e}")
            return {
                "errors": state.get("errors", []) + [str(e)],
                "workflow_state": "error"
            }
    
    async def _writer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Writer agent node"""
        
        self.logger.info("Executing writer node")
        
        try:
            # Extract context
            research_data = state.get("research_data", {})
            query_analysis = state.get("query_analysis", {})
            location = state.get("location", "Unknown")
            
            # Prepare input for writer agent
            writer_input = {
                "research_data": research_data,
                "location": location,
                "query_type": query_analysis.get("query_type", "daily_summary"),
                "narrative_style": "professional",
                "audience": "analysts"
            }
            
            # Execute writer task
            writer_task = await self.writer.execute_task(writer_input)
            
            if writer_task.status != AgentStatus.COMPLETED:
                raise Exception(f"Writer task failed: {writer_task.error_message}")
            
            # Extract results
            writer_output = writer_task.output_data
            narrative_content = {
                "executive_summary": writer_output.get("executive_summary", ""),
                "detailed_narrative": writer_output.get("detailed_narrative", ""),
                "specialized_narratives": writer_output.get("specialized_narratives", {}),
                "story_elements": writer_output.get("story_elements", [])
            }
            
            # Update state
            new_state = {
                "narrative_content": narrative_content,
                "current_agent": "writer",
                "workflow_state": "writer_completed"
            }
            
            # Add message
            new_state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Writing completed for {location} - narrative generated")
            ]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Writer node failed: {e}")
            return {
                "errors": state.get("errors", []) + [str(e)],
                "workflow_state": "error"
            }
    
    async def _editor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Editor agent node"""
        
        self.logger.info("Executing editor node")
        
        try:
            # Extract context
            narrative_content = state.get("narrative_content", {})
            research_data = state.get("research_data", {})
            query_analysis = state.get("query_analysis", {})
            location = state.get("location", "Unknown")
            
            # Prepare content for editing
            content_to_edit = narrative_content.get("detailed_narrative", "")
            
            # Prepare input for editor agent
            editor_input = {
                "content": content_to_edit,
                "source_data": research_data,
                "location": location,
                "audience": "analysts",
                "purpose": "analysis",
                "context": "traffic analysis"
            }
            
            # Execute editor task
            editor_task = await self.editor.execute_task(editor_input)
            
            if editor_task.status != AgentStatus.COMPLETED:
                raise Exception(f"Editor task failed: {editor_task.error_message}")
            
            # Extract results
            editor_output = editor_task.output_data
            edited_content = {
                "original_content": editor_output.get("original_content", ""),
                "corrected_content": editor_output.get("corrected_content", ""),
                "editing_summary": editor_output.get("editing_summary", {}),
                "quality_improvements": editor_output.get("editor_metadata", {}).get("overall_quality_score", 0)
            }
            
            # Update state
            new_state = {
                "edited_content": edited_content,
                "current_agent": "editor",
                "workflow_state": "editor_completed"
            }
            
            # Add message
            new_state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Editing completed for {location} - quality improvements applied")
            ]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Editor node failed: {e}")
            return {
                "errors": state.get("errors", []) + [str(e)],
                "workflow_state": "error"
            }
    
    async def _evaluator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluator agent node"""
        
        self.logger.info("Executing evaluator node")
        
        try:
            # Extract context
            narrative_content = state.get("narrative_content", {})
            edited_content = state.get("edited_content", {})
            research_data = state.get("research_data", {})
            query_analysis = state.get("query_analysis", {})
            
            # Use edited content if available, otherwise use original narrative
            content_to_evaluate = edited_content.get("corrected_content", "") if edited_content else narrative_content.get("detailed_narrative", "")
            
            # Prepare input for evaluator agent
            evaluator_input = {
                "response": content_to_evaluate,
                "context": research_data,
                "question": state.get("user_query", ""),
                "ground_truth": {
                    "expected_findings": research_data.get("research_insights", {}).get("key_findings", []),
                    "expected_causes": research_data.get("cause_analysis", {}).get("primary_causes", []),
                    "expected_recommendations": research_data.get("research_insights", {}).get("recommendations", [])
                }
            }
            
            # Execute evaluator task
            evaluator_task = await self.evaluator.execute_task(evaluator_input)
            
            if evaluator_task.status != AgentStatus.COMPLETED:
                raise Exception(f"Evaluator task failed: {evaluator_task.error_message}")
            
            # Extract results
            evaluator_output = evaluator_task.output_data
            evaluation_results = {
                "evaluation_results": evaluator_output.get("evaluation_results", {}),
                "composite_scores": evaluator_output.get("composite_scores", {}),
                "quality_issues": evaluator_output.get("quality_issues", []),
                "recommendations": evaluator_output.get("recommendations", []),
                "overall_score": evaluator_output.get("overall_score", 0.5),
                "evaluation_report": evaluator_output.get("evaluation_report", {})
            }
            
            # Update state
            new_state = {
                "evaluation_results": evaluation_results,
                "current_agent": "evaluator",
                "workflow_state": "evaluator_completed"
            }
            
            # Add message
            new_state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"Evaluation completed - overall score: {evaluation_results.get('overall_score', 0):.2f}")
            ]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Evaluator node failed: {e}")
            return {
                "errors": state.get("errors", []) + [str(e)],
                "workflow_state": "error"
            }
    
    async def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize node - compile final output"""
        
        self.logger.info("Executing finalize node")
        
        try:
            # Compile final output
            final_output = {
                "user_query": state.get("user_query", ""),
                "location": state.get("location", "Unknown"),
                "mode": state.get("mode", "quick"),
                "query_analysis": state.get("query_analysis", {}),
                "research_data": state.get("research_data", {}),
                "narrative_content": state.get("narrative_content", {}),
                "edited_content": state.get("edited_content", {}),
                "evaluation_results": state.get("evaluation_results", {}),
                "workflow_metadata": {
                    "workflow_state": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "agents_executed": self._get_executed_agents(state),
                    "total_errors": len(state.get("errors", []))
                }
            }
            
            # Update state
            new_state = {
                "final_output": final_output,
                "current_agent": "finalize",
                "workflow_state": "completed"
            }
            
            # Add final message
            new_state["messages"] = state.get("messages", []) + [
                AIMessage(content="Traffix analysis completed successfully")
            ]
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Finalize node failed: {e}")
            return {
                "errors": state.get("errors", []) + [str(e)],
                "workflow_state": "error"
            }
    
    def _should_edit(self, state: Dict[str, Any]) -> str:
        """Determine if content should be edited based on complexity"""
        
        query_analysis = state.get("query_analysis", {})
        complexity = query_analysis.get("complexity", "medium")
        
        # Edit for high and critical complexity queries
        if complexity in ["high", "critical"]:
            return "edit"
        else:
            return "evaluate"
    
    def _get_executed_agents(self, state: Dict[str, Any]) -> List[str]:
        """Get list of agents that were executed"""
        
        agents = []
        
        if state.get("query_analysis"):
            agents.append("supervisor")
        if state.get("research_data"):
            agents.append("research")
        if state.get("narrative_content"):
            agents.append("writer")
        if state.get("edited_content"):
            agents.append("editor")
        if state.get("evaluation_results"):
            agents.append("evaluator")
        
        return agents
    
    async def run_workflow(self, user_query: str, location: str, mode: str = "quick") -> Dict[str, Any]:
        """Run the complete workflow"""
        
        self.logger.info(f"Starting Traffix workflow for {location}")
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "location": location,
            "mode": mode,
            "query_analysis": {},
            "work_plan": {},
            "research_data": {},
            "narrative_content": {},
            "edited_content": {},
            "evaluation_results": {},
            "final_output": {},
            "current_agent": "",
            "workflow_state": "started",
            "errors": []
        }
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            self.logger.info("Traffix workflow completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "error": str(e),
                "workflow_state": "failed",
                "final_output": {}
            }