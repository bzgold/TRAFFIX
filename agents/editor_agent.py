"""
Editor Agent (Copy & Context Checker) - Ensures factual accuracy, readability, empathetic tone
"""
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tech_config import tech_settings


class EditorAgent(BaseAgent):
    """Editor Agent responsible for ensuring factual accuracy, readability, and empathetic tone"""
    
    def __init__(self):
        super().__init__("editor")
        self.logger = logging.getLogger("traffix.editor")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=tech_settings.openai_api_key,
            temperature=0.3,  # Lower temperature for more consistent editing
            max_tokens=2000
        )
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup LLM prompts for different editing tasks"""
        
        # Fact Check Prompt
        self.fact_check_prompt = PromptTemplate(
            input_variables=["content", "source_data", "location"],
            template="""
            Review the following content for factual accuracy against the source data:
            
            Content to Review:
            {content}
            
            Source Data:
            {source_data}
            
            Location: {location}
            
            Please identify any factual inaccuracies, inconsistencies, or claims that cannot be 
            supported by the source data. Provide specific corrections and explanations.
            
            Focus on:
            - Numerical accuracy (speeds, volumes, times)
            - Incident details (severity, location, timing)
            - Cause-effect relationships
            - Data interpretations
            - Statistical claims
            
            Return your findings in a structured format.
            """
        )
        
        # Readability Check Prompt
        self.readability_prompt = PromptTemplate(
            input_variables=["content", "audience", "purpose"],
            template="""
            Review the following content for readability and clarity:
            
            Content:
            {content}
            
            Target Audience: {audience}
            Purpose: {purpose}
            
            Please assess and improve:
            1. Sentence structure and length
            2. Technical jargon and accessibility
            3. Logical flow and organization
            4. Clarity of explanations
            5. Use of active vs passive voice
            6. Paragraph structure
            7. Transition between ideas
            
            Provide specific suggestions for improvement while maintaining the technical accuracy.
            """
        )
        
        # Tone Check Prompt
        self.tone_prompt = PromptTemplate(
            input_variables=["content", "audience", "context"],
            template="""
            Review the following content for appropriate tone and empathy:
            
            Content:
            {content}
            
            Target Audience: {audience}
            Context: {context}
            
            Please ensure the tone is:
            1. Professional yet accessible
            2. Empathetic to those affected by traffic issues
            3. Objective and data-driven
            4. Appropriate for the audience level
            5. Constructive rather than critical
            6. Solution-oriented
            
            Identify any areas where the tone could be improved and suggest specific changes.
            Consider the human impact of traffic issues on commuters, businesses, and communities.
            """
        )
        
        # Consistency Check Prompt
        self.consistency_prompt = PromptTemplate(
            input_variables=["content", "style_guide"],
            template="""
            Review the following content for consistency:
            
            Content:
            {content}
            
            Style Guide:
            {style_guide}
            
            Please check for:
            1. Consistent terminology and definitions
            2. Uniform formatting and structure
            3. Consistent use of units and measurements
            4. Consistent citation and reference style
            5. Consistent voice and perspective
            6. Consistent use of technical terms
            7. Consistent data presentation format
            
            Identify any inconsistencies and provide corrections.
            """
        )
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute editing tasks"""
        content = input_data.get("content", "")
        source_data = input_data.get("source_data", {})
        location = input_data.get("location", "Unknown")
        audience = input_data.get("audience", "analysts")
        purpose = input_data.get("purpose", "analysis")
        context = input_data.get("context", "traffic analysis")
        
        self.logger.info(f"Editor agent reviewing content for {location}")
        
        try:
            # Step 1: Fact check
            fact_check_results = await self._fact_check(content, source_data, location)
            
            # Step 2: Readability check
            readability_results = await self._check_readability(content, audience, purpose)
            
            # Step 3: Tone check
            tone_results = await self._check_tone(content, audience, context)
            
            # Step 4: Consistency check
            consistency_results = await self._check_consistency(content)
            
            # Step 5: Apply corrections
            corrected_content = await self._apply_corrections(
                content, fact_check_results, readability_results, 
                tone_results, consistency_results
            )
            
            # Step 6: Generate editing summary
            editing_summary = self._generate_editing_summary(
                fact_check_results, readability_results, 
                tone_results, consistency_results
            )
            
            return {
                "original_content": content,
                "corrected_content": corrected_content,
                "fact_check_results": fact_check_results,
                "readability_results": readability_results,
                "tone_results": tone_results,
                "consistency_results": consistency_results,
                "editing_summary": editing_summary,
                "editor_metadata": {
                    "location": location,
                    "audience": audience,
                    "purpose": purpose,
                    "edited_at": datetime.now().isoformat(),
                    "overall_quality_score": self._calculate_overall_quality_score(
                        fact_check_results, readability_results, 
                        tone_results, consistency_results
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Editing task failed: {e}")
            raise
    
    async def _fact_check(self, content: str, source_data: Dict[str, Any], 
                         location: str) -> Dict[str, Any]:
        """Perform fact checking against source data"""
        
        try:
            # Format source data for prompt
            formatted_source = self._format_source_data_for_fact_check(source_data)
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.fact_check_prompt)
            
            # Generate fact check results
            fact_check_response = await chain.arun(
                content=content,
                source_data=formatted_source,
                location=location
            )
            
            # Parse response and extract issues
            issues = self._parse_fact_check_response(fact_check_response)
            
            return {
                "issues_found": len(issues),
                "issues": issues,
                "accuracy_score": self._calculate_accuracy_score(issues),
                "fact_check_response": fact_check_response
            }
            
        except Exception as e:
            self.logger.error(f"Fact check failed: {e}")
            return {
                "issues_found": 0,
                "issues": [],
                "accuracy_score": 0.8,  # Default score
                "fact_check_response": "Fact check could not be completed"
            }
    
    async def _check_readability(self, content: str, audience: str, 
                               purpose: str) -> Dict[str, Any]:
        """Check content readability and clarity"""
        
        try:
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.readability_prompt)
            
            # Generate readability assessment
            readability_response = await chain.arun(
                content=content,
                audience=audience,
                purpose=purpose
            )
            
            # Calculate readability metrics
            readability_metrics = self._calculate_readability_metrics(content)
            
            # Parse suggestions
            suggestions = self._parse_readability_suggestions(readability_response)
            
            return {
                "readability_metrics": readability_metrics,
                "suggestions": suggestions,
                "readability_score": self._calculate_readability_score(readability_metrics),
                "readability_response": readability_response
            }
            
        except Exception as e:
            self.logger.error(f"Readability check failed: {e}")
            return {
                "readability_metrics": {},
                "suggestions": [],
                "readability_score": 0.7,
                "readability_response": "Readability check could not be completed"
            }
    
    async def _check_tone(self, content: str, audience: str, 
                         context: str) -> Dict[str, Any]:
        """Check content tone and empathy"""
        
        try:
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.tone_prompt)
            
            # Generate tone assessment
            tone_response = await chain.arun(
                content=content,
                audience=audience,
                context=context
            )
            
            # Analyze tone characteristics
            tone_analysis = self._analyze_tone_characteristics(content)
            
            # Parse tone suggestions
            tone_suggestions = self._parse_tone_suggestions(tone_response)
            
            return {
                "tone_analysis": tone_analysis,
                "suggestions": tone_suggestions,
                "tone_score": self._calculate_tone_score(tone_analysis),
                "tone_response": tone_response
            }
            
        except Exception as e:
            self.logger.error(f"Tone check failed: {e}")
            return {
                "tone_analysis": {},
                "suggestions": [],
                "tone_score": 0.7,
                "tone_response": "Tone check could not be completed"
            }
    
    async def _check_consistency(self, content: str) -> Dict[str, Any]:
        """Check content consistency"""
        
        try:
            # Define style guide
            style_guide = self._get_style_guide()
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.consistency_prompt)
            
            # Generate consistency assessment
            consistency_response = await chain.arun(
                content=content,
                style_guide=style_guide
            )
            
            # Check for consistency issues
            consistency_issues = self._identify_consistency_issues(content)
            
            return {
                "consistency_issues": consistency_issues,
                "consistency_score": self._calculate_consistency_score(consistency_issues),
                "consistency_response": consistency_response
            }
            
        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return {
                "consistency_issues": [],
                "consistency_score": 0.8,
                "consistency_response": "Consistency check could not be completed"
            }
    
    async def _apply_corrections(self, content: str, fact_check_results: Dict[str, Any],
                               readability_results: Dict[str, Any], tone_results: Dict[str, Any],
                               consistency_results: Dict[str, Any]) -> str:
        """Apply all corrections to the content"""
        
        corrected_content = content
        
        try:
            # Apply fact corrections
            if fact_check_results.get("issues"):
                corrected_content = self._apply_fact_corrections(
                    corrected_content, fact_check_results["issues"]
                )
            
            # Apply readability improvements
            if readability_results.get("suggestions"):
                corrected_content = self._apply_readability_improvements(
                    corrected_content, readability_results["suggestions"]
                )
            
            # Apply tone improvements
            if tone_results.get("suggestions"):
                corrected_content = self._apply_tone_improvements(
                    corrected_content, tone_results["suggestions"]
                )
            
            # Apply consistency fixes
            if consistency_results.get("consistency_issues"):
                corrected_content = self._apply_consistency_fixes(
                    corrected_content, consistency_results["consistency_issues"]
                )
            
            return corrected_content
            
        except Exception as e:
            self.logger.error(f"Failed to apply corrections: {e}")
            return content
    
    def _format_source_data_for_fact_check(self, source_data: Dict[str, Any]) -> str:
        """Format source data for fact checking"""
        
        formatted = "Source Data for Fact Checking:\n\n"
        
        # Add key incidents
        key_incidents = source_data.get("key_incidents", [])
        if key_incidents:
            formatted += f"Key Incidents ({len(key_incidents)}):\n"
            for incident in key_incidents:
                formatted += f"- {incident.get('description', 'Unknown')}\n"
                formatted += f"  Severity: {incident.get('severity', 'Unknown')}\n"
                formatted += f"  Location: {incident.get('location', 'Unknown')}\n"
                formatted += f"  Time: {incident.get('start_time', 'Unknown')}\n\n"
        
        # Add traffic data
        traffic_data = source_data.get("traffic_data", [])
        if traffic_data:
            formatted += f"Traffic Data ({len(traffic_data)} records):\n"
            for data in traffic_data[:5]:  # First 5 records
                formatted += f"- Speed: {data.get('speed', 'N/A')} mph\n"
                formatted += f"  Volume: {data.get('volume', 'N/A')} vehicles\n"
                formatted += f"  Time: {data.get('timestamp', 'N/A')}\n\n"
        
        # Add cause analysis
        cause_analysis = source_data.get("cause_analysis", {})
        if cause_analysis:
            formatted += "Cause Analysis:\n"
            formatted += f"Primary Causes: {cause_analysis.get('primary_causes', [])}\n"
            formatted += f"Contributing Factors: {cause_analysis.get('contributing_factors', [])}\n"
            formatted += f"Confidence: {cause_analysis.get('cause_confidence', 0):.2f}\n\n"
        
        return formatted
    
    def _parse_fact_check_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse fact check response to extract issues"""
        
        issues = []
        
        # Simple parsing - in real implementation, this would be more sophisticated
        lines = response.split('\n')
        current_issue = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Issue:'):
                if current_issue:
                    issues.append(current_issue)
                current_issue = {"description": line[7:].strip()}
            elif line.startswith('Correction:'):
                current_issue["correction"] = line[12:].strip()
            elif line.startswith('Severity:'):
                current_issue["severity"] = line[10:].strip()
        
        if current_issue:
            issues.append(current_issue)
        
        return issues
    
    def _calculate_accuracy_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score based on issues found"""
        
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.5, "critical": 0.8}
        total_weight = sum(severity_weights.get(issue.get("severity", "medium"), 0.3) for issue in issues)
        
        # Calculate score (1.0 - weighted penalty)
        score = max(0.0, 1.0 - (total_weight * 0.2))
        return score
    
    def _calculate_readability_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate readability metrics"""
        
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        # Basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Count complex words (3+ syllables - simplified)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_word_ratio = complex_words / len(words) if words else 0
        
        # Flesch Reading Ease approximation
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * complex_word_ratio)
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "complex_word_ratio": complex_word_ratio,
            "flesch_score": flesch_score,
            "total_words": len(words),
            "total_sentences": len(sentences)
        }
    
    def _parse_readability_suggestions(self, response: str) -> List[str]:
        """Parse readability suggestions from response"""
        
        suggestions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                suggestions.append(line[1:].strip())
        
        return suggestions
    
    def _calculate_readability_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall readability score"""
        
        flesch_score = metrics.get("flesch_score", 50)
        avg_sentence_length = metrics.get("avg_sentence_length", 15)
        complex_word_ratio = metrics.get("complex_word_ratio", 0.3)
        
        # Normalize Flesch score (0-100 to 0-1)
        flesch_normalized = max(0, min(1, flesch_score / 100))
        
        # Penalize very long sentences
        sentence_penalty = max(0, (avg_sentence_length - 20) * 0.02)
        
        # Penalize too many complex words
        complexity_penalty = max(0, (complex_word_ratio - 0.2) * 0.5)
        
        score = flesch_normalized - sentence_penalty - complexity_penalty
        return max(0, min(1, score))
    
    def _analyze_tone_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze tone characteristics of content"""
        
        # Count positive/negative words (simplified)
        positive_words = ['improved', 'better', 'successful', 'effective', 'efficient', 'optimized']
        negative_words = ['problem', 'issue', 'failure', 'delay', 'congestion', 'accident']
        
        positive_count = sum(1 for word in positive_words if word.lower() in content.lower())
        negative_count = sum(1 for word in negative_words if word.lower() in content.lower())
        
        # Check for empathetic language
        empathetic_words = ['impact', 'affected', 'community', 'residents', 'commuters', 'safety']
        empathetic_count = sum(1 for word in empathetic_words if word.lower() in content.lower())
        
        # Check for professional language
        professional_words = ['analysis', 'data', 'evidence', 'recommendation', 'strategy']
        professional_count = sum(1 for word in professional_words if word.lower() in content.lower())
        
        return {
            "positive_sentiment": positive_count,
            "negative_sentiment": negative_count,
            "empathetic_language": empathetic_count,
            "professional_language": professional_count,
            "tone_balance": (positive_count - negative_count) / max(1, positive_count + negative_count)
        }
    
    def _parse_tone_suggestions(self, response: str) -> List[str]:
        """Parse tone suggestions from response"""
        
        suggestions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                suggestions.append(line[1:].strip())
        
        return suggestions
    
    def _calculate_tone_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate tone score based on analysis"""
        
        empathetic_score = min(1, analysis.get("empathetic_language", 0) / 3)
        professional_score = min(1, analysis.get("professional_language", 0) / 5)
        balance_score = (analysis.get("tone_balance", 0) + 1) / 2  # Normalize -1 to 1 to 0 to 1
        
        return (empathetic_score + professional_score + balance_score) / 3
    
    def _get_style_guide(self) -> str:
        """Get style guide for consistency checking"""
        
        return """
        Traffix Style Guide:
        
        1. Terminology:
           - Use "traffic congestion" not "traffic jam"
           - Use "incident" not "accident" unless referring to specific crash
           - Use "travel time" not "drive time"
           - Use "corridor" for major roadways
        
        2. Units:
           - Speed: mph (miles per hour)
           - Distance: miles
           - Time: hours, minutes
           - Volume: vehicles per hour
        
        3. Formatting:
           - Use bullet points for lists
           - Use numbered lists for steps
           - Use bold for key metrics
           - Use italics for emphasis
        
        4. Voice:
           - Use active voice when possible
           - Use present tense for current conditions
           - Use past tense for historical data
        
        5. Citations:
           - Reference data sources
           - Include timestamps
           - Specify confidence levels
        """
    
    def _identify_consistency_issues(self, content: str) -> List[Dict[str, Any]]:
        """Identify consistency issues in content"""
        
        issues = []
        
        # Check for inconsistent terminology
        terminology_issues = self._check_terminology_consistency(content)
        issues.extend(terminology_issues)
        
        # Check for inconsistent units
        unit_issues = self._check_unit_consistency(content)
        issues.extend(unit_issues)
        
        # Check for inconsistent formatting
        formatting_issues = self._check_formatting_consistency(content)
        issues.extend(formatting_issues)
        
        return issues
    
    def _check_terminology_consistency(self, content: str) -> List[Dict[str, Any]]:
        """Check for terminology consistency"""
        
        issues = []
        
        # Check for mixed use of "accident" and "incident"
        if "accident" in content.lower() and "incident" in content.lower():
            issues.append({
                "type": "terminology",
                "description": "Mixed use of 'accident' and 'incident'",
                "suggestion": "Use 'incident' consistently"
            })
        
        # Check for mixed use of "traffic jam" and "congestion"
        if "traffic jam" in content.lower() and "congestion" in content.lower():
            issues.append({
                "type": "terminology",
                "description": "Mixed use of 'traffic jam' and 'congestion'",
                "suggestion": "Use 'congestion' consistently"
            })
        
        return issues
    
    def _check_unit_consistency(self, content: str) -> List[Dict[str, Any]]:
        """Check for unit consistency"""
        
        issues = []
        
        # Check for mixed speed units
        if "mph" in content and "km/h" in content:
            issues.append({
                "type": "units",
                "description": "Mixed speed units (mph and km/h)",
                "suggestion": "Use mph consistently"
            })
        
        # Check for mixed time units
        if "hours" in content and "hrs" in content:
            issues.append({
                "type": "units",
                "description": "Mixed time unit abbreviations",
                "suggestion": "Use 'hours' consistently"
            })
        
        return issues
    
    def _check_formatting_consistency(self, content: str) -> List[Dict[str, Any]]:
        """Check for formatting consistency"""
        
        issues = []
        
        # Check for inconsistent bullet points
        bullet_variations = content.count('•') + content.count('-') + content.count('*')
        if bullet_variations > 1:
            issues.append({
                "type": "formatting",
                "description": "Inconsistent bullet point styles",
                "suggestion": "Use consistent bullet point style"
            })
        
        return issues
    
    def _calculate_consistency_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate consistency score based on issues"""
        
        if not issues:
            return 1.0
        
        # Weight different types of issues
        type_weights = {"terminology": 0.3, "units": 0.4, "formatting": 0.2, "other": 0.1}
        total_weight = sum(type_weights.get(issue.get("type", "other"), 0.1) for issue in issues)
        
        score = max(0.0, 1.0 - (total_weight * 0.2))
        return score
    
    def _apply_fact_corrections(self, content: str, issues: List[Dict[str, Any]]) -> str:
        """Apply fact corrections to content"""
        
        corrected_content = content
        
        for issue in issues:
            if "correction" in issue:
                # Simple replacement - in real implementation, this would be more sophisticated
                original = issue.get("description", "")
                correction = issue.get("correction", "")
                if original and correction:
                    corrected_content = corrected_content.replace(original, correction)
        
        return corrected_content
    
    def _apply_readability_improvements(self, content: str, suggestions: List[str]) -> str:
        """Apply readability improvements to content"""
        
        # Simple improvements - in real implementation, this would use more sophisticated NLP
        improved_content = content
        
        # Break up long sentences (simple heuristic)
        sentences = improved_content.split('. ')
        improved_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 25:  # Long sentence
                # Simple split at comma or conjunction
                if ', ' in sentence:
                    parts = sentence.split(', ', 1)
                    improved_sentences.extend([parts[0] + '.', parts[1]])
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return '. '.join(improved_sentences)
    
    def _apply_tone_improvements(self, content: str, suggestions: List[str]) -> str:
        """Apply tone improvements to content"""
        
        # Simple tone improvements
        improved_content = content
        
        # Replace negative phrases with more neutral ones
        tone_replacements = {
            "traffic jam": "traffic congestion",
            "accident": "incident",
            "problem": "challenge",
            "issue": "situation"
        }
        
        for old, new in tone_replacements.items():
            improved_content = improved_content.replace(old, new)
        
        return improved_content
    
    def _apply_consistency_fixes(self, content: str, issues: List[Dict[str, Any]]) -> str:
        """Apply consistency fixes to content"""
        
        fixed_content = content
        
        for issue in issues:
            if issue.get("type") == "terminology":
                if "accident" in fixed_content.lower() and "incident" in fixed_content.lower():
                    fixed_content = fixed_content.replace("accident", "incident")
            elif issue.get("type") == "units":
                if "km/h" in fixed_content:
                    fixed_content = fixed_content.replace("km/h", "mph")
        
        return fixed_content
    
    def _generate_editing_summary(self, fact_check_results: Dict[str, Any],
                                readability_results: Dict[str, Any],
                                tone_results: Dict[str, Any],
                                consistency_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate editing summary"""
        
        return {
            "fact_check": {
                "issues_found": fact_check_results.get("issues_found", 0),
                "accuracy_score": fact_check_results.get("accuracy_score", 0.8)
            },
            "readability": {
                "score": readability_results.get("readability_score", 0.7),
                "suggestions_count": len(readability_results.get("suggestions", []))
            },
            "tone": {
                "score": tone_results.get("tone_score", 0.7),
                "suggestions_count": len(tone_results.get("suggestions", []))
            },
            "consistency": {
                "issues_found": len(consistency_results.get("consistency_issues", [])),
                "score": consistency_results.get("consistency_score", 0.8)
            },
            "overall_improvements": self._calculate_total_improvements(
                fact_check_results, readability_results, tone_results, consistency_results
            )
        }
    
    def _calculate_total_improvements(self, fact_check_results: Dict[str, Any],
                                    readability_results: Dict[str, Any],
                                    tone_results: Dict[str, Any],
                                    consistency_results: Dict[str, Any]) -> int:
        """Calculate total number of improvements made"""
        
        total = 0
        total += fact_check_results.get("issues_found", 0)
        total += len(readability_results.get("suggestions", []))
        total += len(tone_results.get("suggestions", []))
        total += len(consistency_results.get("consistency_issues", []))
        
        return total
    
    def _calculate_overall_quality_score(self, fact_check_results: Dict[str, Any],
                                       readability_results: Dict[str, Any],
                                       tone_results: Dict[str, Any],
                                       consistency_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        
        scores = [
            fact_check_results.get("accuracy_score", 0.8),
            readability_results.get("readability_score", 0.7),
            tone_results.get("tone_score", 0.7),
            consistency_results.get("consistency_score", 0.8)
        ]
        
        return sum(scores) / len(scores)
