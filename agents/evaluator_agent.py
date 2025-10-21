"""
Evaluator Agent (QA) - Uses RAGAS-style heuristics to improve pipeline quality
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tech_config import tech_settings


class EvaluatorAgent(BaseAgent):
    """Evaluator Agent responsible for quality assurance using RAGAS-style metrics"""
    
    def __init__(self):
        super().__init__("evaluator")
        self.logger = logging.getLogger("traffix.evaluator")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=tech_settings.openai_api_key,
            temperature=0.1,  # Very low temperature for consistent evaluation
            max_tokens=1000
        )
        self._setup_prompts()
        self._setup_ragas_metrics()
    
    def _setup_prompts(self):
        """Setup LLM prompts for different evaluation tasks"""
        
        # Faithfulness Prompt (RAGAS metric)
        self.faithfulness_prompt = PromptTemplate(
            input_variables=["response", "context"],
            template="""
            Evaluate the faithfulness of the response to the given context.
            
            Context (source data):
            {context}
            
            Response to evaluate:
            {response}
            
            Rate the faithfulness on a scale of 0-1 where:
            0 = Response contains claims not supported by context
            0.5 = Response is partially supported by context
            1 = Response is fully supported by context
            
            Consider:
            - Are all claims in the response backed by the context?
            - Are there any hallucinations or unsupported statements?
            - Are the data interpretations accurate?
            - Are the conclusions justified by the evidence?
            
            Provide your score and brief explanation.
            """
        )
        
        # Answer Relevancy Prompt (RAGAS metric)
        self.relevancy_prompt = PromptTemplate(
            input_variables=["response", "question"],
            template="""
            Evaluate the relevancy of the response to the given question.
            
            Question:
            {question}
            
            Response to evaluate:
            {response}
            
            Rate the relevancy on a scale of 0-1 where:
            0 = Response is completely irrelevant to the question
            0.5 = Response is partially relevant
            1 = Response is highly relevant and directly answers the question
            
            Consider:
            - Does the response address the specific question asked?
            - Is the information provided useful for the question?
            - Are there irrelevant details that don't help answer the question?
            - Does the response stay focused on the topic?
            
            Provide your score and brief explanation.
            """
        )
        
        # Context Precision Prompt (RAGAS metric)
        self.context_precision_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Evaluate the precision of the context in answering the question.
            
            Question:
            {question}
            
            Context provided:
            {context}
            
            Rate the context precision on a scale of 0-1 where:
            0 = Context contains mostly irrelevant information
            0.5 = Context is partially relevant
            1 = Context is highly precise and directly relevant
            
            Consider:
            - How much of the context is actually relevant to the question?
            - Are there unnecessary details that don't help answer the question?
            - Is the context focused and targeted?
            - Does the context provide the right level of detail?
            
            Provide your score and brief explanation.
            """
        )
        
        # Context Recall Prompt (RAGAS metric)
        self.context_recall_prompt = PromptTemplate(
            input_variables=["context", "ground_truth"],
            template="""
            Evaluate the recall of the context compared to ground truth.
            
            Ground truth (expected information):
            {ground_truth}
            
            Context provided:
            {context}
            
            Rate the context recall on a scale of 0-1 where:
            0 = Context misses most important information
            0.5 = Context captures some important information
            1 = Context captures all important information
            
            Consider:
            - How much of the ground truth information is present in the context?
            - Are there important details missing from the context?
            - Is the context comprehensive enough?
            - Does the context cover all necessary aspects?
            
            Provide your score and brief explanation.
            """
        )
        
        # Answer Correctness Prompt (RAGAS metric)
        self.correctness_prompt = PromptTemplate(
            input_variables=["response", "ground_truth"],
            template="""
            Evaluate the correctness of the response against ground truth.
            
            Ground truth:
            {ground_truth}
            
            Response to evaluate:
            {response}
            
            Rate the correctness on a scale of 0-1 where:
            0 = Response is completely incorrect
            0.5 = Response is partially correct
            1 = Response is completely correct
            
            Consider:
            - Are the facts and figures accurate?
            - Are the interpretations correct?
            - Are the conclusions valid?
            - Are there any errors in the response?
            
            Provide your score and brief explanation.
            """
        )
        
        # Answer Similarity Prompt (RAGAS metric)
        self.similarity_prompt = PromptTemplate(
            input_variables=["response", "ground_truth"],
            template="""
            Evaluate the semantic similarity between response and ground truth.
            
            Ground truth:
            {ground_truth}
            
            Response to evaluate:
            {response}
            
            Rate the similarity on a scale of 0-1 where:
            0 = Response is completely different from ground truth
            0.5 = Response is somewhat similar
            1 = Response is very similar in meaning
            
            Consider:
            - Do the responses convey the same information?
            - Are the key points the same?
            - Is the overall message consistent?
            - Are there differences in wording but same meaning?
            
            Provide your score and brief explanation.
            """
        )
    
    def _setup_ragas_metrics(self):
        """Setup RAGAS-style evaluation metrics"""
        
        self.ragas_metrics = {
            "faithfulness": {
                "weight": 0.25,
                "description": "Measures how grounded the response is in the given context",
                "threshold": 0.7
            },
            "answer_relevancy": {
                "weight": 0.20,
                "description": "Measures how relevant the response is to the given question",
                "threshold": 0.8
            },
            "context_precision": {
                "weight": 0.15,
                "description": "Measures how precise the context is in answering the question",
                "threshold": 0.6
            },
            "context_recall": {
                "weight": 0.15,
                "description": "Measures how well the context captures the ground truth",
                "threshold": 0.7
            },
            "answer_correctness": {
                "weight": 0.15,
                "description": "Measures the correctness of the response against ground truth",
                "threshold": 0.8
            },
            "answer_similarity": {
                "weight": 0.10,
                "description": "Measures semantic similarity between response and ground truth",
                "threshold": 0.7
            }
        }
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluation tasks using RAGAS-style metrics"""
        
        response = input_data.get("response", "")
        context = input_data.get("context", {})
        question = input_data.get("question", "")
        ground_truth = input_data.get("ground_truth", {})
        
        self.logger.info("Evaluator agent performing RAGAS-style evaluation")
        
        try:
            # Step 1: Evaluate individual RAGAS metrics
            evaluation_results = await self._evaluate_ragas_metrics(
                response, context, question, ground_truth
            )
            
            # Step 2: Calculate composite scores
            composite_scores = self._calculate_composite_scores(evaluation_results)
            
            # Step 3: Identify quality issues
            quality_issues = self._identify_quality_issues(evaluation_results)
            
            # Step 4: Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(
                evaluation_results, quality_issues
            )
            
            # Step 5: Calculate overall quality score
            overall_score = self._calculate_overall_quality_score(evaluation_results)
            
            # Step 6: Generate evaluation report
            evaluation_report = self._generate_evaluation_report(
                evaluation_results, composite_scores, quality_issues, recommendations
            )
            
            return {
                "evaluation_results": evaluation_results,
                "composite_scores": composite_scores,
                "quality_issues": quality_issues,
                "recommendations": recommendations,
                "overall_score": overall_score,
                "evaluation_report": evaluation_report,
                "evaluator_metadata": {
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "metrics_evaluated": list(self.ragas_metrics.keys()),
                    "evaluation_method": "RAGAS-style",
                    "quality_grade": self._assign_quality_grade(overall_score)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    async def _evaluate_ragas_metrics(self, response: str, context: Dict[str, Any], 
                                    question: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all RAGAS metrics"""
        
        evaluation_results = {}
        
        # Prepare context and ground truth for evaluation
        formatted_context = self._format_context_for_evaluation(context)
        formatted_ground_truth = self._format_ground_truth_for_evaluation(ground_truth)
        
        # Evaluate each metric
        for metric_name, metric_config in self.ragas_metrics.items():
            try:
                score, explanation = await self._evaluate_single_metric(
                    metric_name, response, formatted_context, question, formatted_ground_truth
                )
                
                evaluation_results[metric_name] = {
                    "score": score,
                    "explanation": explanation,
                    "threshold": metric_config["threshold"],
                    "passed": score >= metric_config["threshold"],
                    "weight": metric_config["weight"]
                }
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {metric_name}: {e}")
                evaluation_results[metric_name] = {
                    "score": 0.5,
                    "explanation": f"Evaluation failed: {str(e)}",
                    "threshold": metric_config["threshold"],
                    "passed": False,
                    "weight": metric_config["weight"]
                }
        
        return evaluation_results
    
    async def _evaluate_single_metric(self, metric_name: str, response: str, 
                                    context: str, question: str, 
                                    ground_truth: str) -> Tuple[float, str]:
        """Evaluate a single RAGAS metric"""
        
        # Get the appropriate prompt
        prompt_map = {
            "faithfulness": self.faithfulness_prompt,
            "answer_relevancy": self.relevancy_prompt,
            "context_precision": self.context_precision_prompt,
            "context_recall": self.context_recall_prompt,
            "answer_correctness": self.correctness_prompt,
            "answer_similarity": self.similarity_prompt
        }
        
        prompt = prompt_map.get(metric_name)
        if not prompt:
            return 0.5, "Unknown metric"
        
        try:
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate evaluation
            if metric_name in ["faithfulness", "answer_relevancy"]:
                evaluation = await chain.arun(response=response, context=context)
            elif metric_name in ["context_precision", "context_recall"]:
                evaluation = await chain.arun(context=context, question=question)
            else:  # answer_correctness, answer_similarity
                evaluation = await chain.arun(response=response, ground_truth=ground_truth)
            
            # Parse score and explanation
            score, explanation = self._parse_evaluation_response(evaluation)
            
            return score, explanation
            
        except Exception as e:
            self.logger.error(f"Error evaluating {metric_name}: {e}")
            return 0.5, f"Evaluation error: {str(e)}"
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, str]:
        """Parse evaluation response to extract score and explanation"""
        
        try:
            # Look for score in the response
            lines = response.split('\n')
            score = 0.5  # Default score
            explanation = response
            
            for line in lines:
                line = line.strip().lower()
                if 'score:' in line:
                    # Extract score
                    score_part = line.split('score:')[1].strip()
                    try:
                        score = float(score_part.split()[0])
                        score = max(0, min(1, score))  # Clamp to 0-1
                    except:
                        pass
                elif 'rating:' in line:
                    # Extract score from rating
                    rating_part = line.split('rating:')[1].strip()
                    try:
                        score = float(rating_part.split()[0])
                        score = max(0, min(1, score))  # Clamp to 0-1
                    except:
                        pass
            
            return score, explanation
            
        except Exception as e:
            self.logger.error(f"Error parsing evaluation response: {e}")
            return 0.5, response
    
    def _calculate_composite_scores(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite scores from individual metrics"""
        
        # Weighted average of all metrics
        weighted_sum = 0
        total_weight = 0
        
        for metric_name, results in evaluation_results.items():
            weight = results.get("weight", 0)
            score = results.get("score", 0)
            weighted_sum += score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Calculate category scores
        faithfulness_score = evaluation_results.get("faithfulness", {}).get("score", 0.5)
        relevancy_score = evaluation_results.get("answer_relevancy", {}).get("score", 0.5)
        context_score = (
            evaluation_results.get("context_precision", {}).get("score", 0.5) +
            evaluation_results.get("context_recall", {}).get("score", 0.5)
        ) / 2
        
        correctness_score = (
            evaluation_results.get("answer_correctness", {}).get("score", 0.5) +
            evaluation_results.get("answer_similarity", {}).get("score", 0.5)
        ) / 2
        
        return {
            "overall_score": overall_score,
            "faithfulness_score": faithfulness_score,
            "relevancy_score": relevancy_score,
            "context_score": context_score,
            "correctness_score": correctness_score,
            "metrics_passed": sum(1 for results in evaluation_results.values() 
                                if results.get("passed", False)),
            "total_metrics": len(evaluation_results)
        }
    
    def _identify_quality_issues(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify quality issues based on evaluation results"""
        
        issues = []
        
        for metric_name, results in evaluation_results.items():
            if not results.get("passed", False):
                issue = {
                    "metric": metric_name,
                    "score": results.get("score", 0),
                    "threshold": results.get("threshold", 0.5),
                    "severity": self._calculate_issue_severity(
                        results.get("score", 0), results.get("threshold", 0.5)
                    ),
                    "description": self._get_issue_description(metric_name, results),
                    "explanation": results.get("explanation", "")
                }
                issues.append(issue)
        
        # Sort by severity
        issues.sort(key=lambda x: x["severity"], reverse=True)
        
        return issues
    
    def _calculate_issue_severity(self, score: float, threshold: float) -> str:
        """Calculate issue severity based on score vs threshold"""
        
        gap = threshold - score
        
        if gap > 0.3:
            return "critical"
        elif gap > 0.2:
            return "high"
        elif gap > 0.1:
            return "medium"
        else:
            return "low"
    
    def _get_issue_description(self, metric_name: str, results: Dict[str, Any]) -> str:
        """Get human-readable issue description"""
        
        descriptions = {
            "faithfulness": "Response contains claims not supported by source data",
            "answer_relevancy": "Response is not relevant to the question asked",
            "context_precision": "Context contains too much irrelevant information",
            "context_recall": "Context is missing important information",
            "answer_correctness": "Response contains factual errors",
            "answer_similarity": "Response differs significantly from expected answer"
        }
        
        return descriptions.get(metric_name, f"Quality issue with {metric_name}")
    
    def _generate_improvement_recommendations(self, evaluation_results: Dict[str, Any], 
                                            quality_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on evaluation results"""
        
        recommendations = []
        
        # Generate recommendations for each failed metric
        for issue in quality_issues:
            metric_name = issue["metric"]
            severity = issue["severity"]
            
            recommendation = {
                "metric": metric_name,
                "priority": severity,
                "recommendation": self._get_improvement_recommendation(metric_name, issue),
                "actionable_steps": self._get_actionable_steps(metric_name, issue)
            }
            recommendations.append(recommendation)
        
        # Add general recommendations
        if len(quality_issues) > 3:
            recommendations.append({
                "metric": "overall",
                "priority": "high",
                "recommendation": "Multiple quality issues detected - comprehensive review needed",
                "actionable_steps": [
                    "Review all source data for accuracy",
                    "Ensure response directly addresses the question",
                    "Verify all claims are supported by evidence",
                    "Improve context selection and filtering"
                ]
            })
        
        return recommendations
    
    def _get_improvement_recommendation(self, metric_name: str, issue: Dict[str, Any]) -> str:
        """Get specific improvement recommendation for a metric"""
        
        recommendations = {
            "faithfulness": "Ensure all claims in the response are directly supported by the source data. Remove any unsupported statements or interpretations.",
            "answer_relevancy": "Focus the response on directly answering the specific question asked. Remove irrelevant details and stay on topic.",
            "context_precision": "Improve context selection to include only information directly relevant to the question. Filter out unnecessary details.",
            "context_recall": "Ensure the context includes all necessary information to answer the question comprehensively. Add missing relevant details.",
            "answer_correctness": "Verify all facts, figures, and interpretations against reliable sources. Correct any errors or inaccuracies.",
            "answer_similarity": "Align the response more closely with the expected answer while maintaining accuracy and completeness."
        }
        
        return recommendations.get(metric_name, "Improve the quality of this metric")
    
    def _get_actionable_steps(self, metric_name: str, issue: Dict[str, Any]) -> List[str]:
        """Get actionable steps for improving a metric"""
        
        steps_map = {
            "faithfulness": [
                "Cross-reference every claim with source data",
                "Remove unsupported statements",
                "Add citations for key claims",
                "Verify data interpretations"
            ],
            "answer_relevancy": [
                "Identify the core question being asked",
                "Remove off-topic information",
                "Focus on direct answers",
                "Structure response around the question"
            ],
            "context_precision": [
                "Review context for relevance",
                "Remove unnecessary details",
                "Focus on key information",
                "Improve context filtering"
            ],
            "context_recall": [
                "Identify missing information",
                "Add relevant details",
                "Ensure comprehensive coverage",
                "Verify completeness"
            ],
            "answer_correctness": [
                "Fact-check all information",
                "Verify numerical data",
                "Check interpretations",
                "Correct errors"
            ],
            "answer_similarity": [
                "Compare with expected answer",
                "Align key points",
                "Maintain accuracy",
                "Improve consistency"
            ]
        }
        
        return steps_map.get(metric_name, ["Review and improve the response"])
    
    def _calculate_overall_quality_score(self, evaluation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        
        composite_scores = self._calculate_composite_scores(evaluation_results)
        return composite_scores.get("overall_score", 0.5)
    
    def _assign_quality_grade(self, overall_score: float) -> str:
        """Assign quality grade based on overall score"""
        
        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B"
        elif overall_score >= 0.6:
            return "C"
        elif overall_score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                  composite_scores: Dict[str, Any],
                                  quality_issues: List[Dict[str, Any]],
                                  recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        return {
            "summary": {
                "overall_score": composite_scores.get("overall_score", 0.5),
                "quality_grade": self._assign_quality_grade(composite_scores.get("overall_score", 0.5)),
                "metrics_passed": composite_scores.get("metrics_passed", 0),
                "total_metrics": composite_scores.get("total_metrics", 0),
                "issues_found": len(quality_issues),
                "critical_issues": len([i for i in quality_issues if i["severity"] == "critical"])
            },
            "detailed_scores": {
                metric: {
                    "score": results.get("score", 0),
                    "passed": results.get("passed", False),
                    "threshold": results.get("threshold", 0.5)
                }
                for metric, results in evaluation_results.items()
            },
            "quality_issues": quality_issues,
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(quality_issues, recommendations)
        }
    
    def _generate_next_steps(self, quality_issues: List[Dict[str, Any]], 
                           recommendations: List[Dict[str, Any]]) -> List[str]:
        """Generate next steps based on evaluation results"""
        
        next_steps = []
        
        if not quality_issues:
            next_steps.append("Quality evaluation passed - response is ready for use")
        else:
            critical_issues = [i for i in quality_issues if i["severity"] == "critical"]
            if critical_issues:
                next_steps.append("Address critical quality issues immediately")
                next_steps.append("Review and revise response based on recommendations")
            else:
                next_steps.append("Address quality issues to improve response")
                next_steps.append("Consider implementing recommended improvements")
        
        next_steps.append("Monitor quality metrics in future responses")
        next_steps.append("Update evaluation criteria based on feedback")
        
        return next_steps
    
    def _format_context_for_evaluation(self, context: Dict[str, Any]) -> str:
        """Format context data for evaluation"""
        
        formatted = "Context Data:\n\n"
        
        # Add key incidents
        key_incidents = context.get("key_incidents", [])
        if key_incidents:
            formatted += f"Key Incidents ({len(key_incidents)}):\n"
            for incident in key_incidents[:5]:
                formatted += f"- {incident.get('description', 'Unknown')}\n"
            formatted += "\n"
        
        # Add cause analysis
        cause_analysis = context.get("cause_analysis", {})
        if cause_analysis:
            formatted += "Cause Analysis:\n"
            formatted += f"Primary Causes: {cause_analysis.get('primary_causes', [])}\n"
            formatted += f"Contributing Factors: {cause_analysis.get('contributing_factors', [])}\n\n"
        
        # Add research insights
        insights = context.get("research_insights", {})
        if insights:
            formatted += "Research Insights:\n"
            formatted += f"Summary: {insights.get('summary', 'No summary')}\n"
            formatted += f"Key Findings: {insights.get('key_findings', [])}\n"
        
        return formatted
    
    def _format_ground_truth_for_evaluation(self, ground_truth: Dict[str, Any]) -> str:
        """Format ground truth data for evaluation"""
        
        formatted = "Ground Truth Data:\n\n"
        
        # Add expected findings
        expected_findings = ground_truth.get("expected_findings", [])
        if expected_findings:
            formatted += f"Expected Findings: {expected_findings}\n"
        
        # Add expected causes
        expected_causes = ground_truth.get("expected_causes", [])
        if expected_causes:
            formatted += f"Expected Causes: {expected_causes}\n"
        
        # Add expected recommendations
        expected_recommendations = ground_truth.get("expected_recommendations", [])
        if expected_recommendations:
            formatted += f"Expected Recommendations: {expected_recommendations}\n"
        
        return formatted
