"""
Writer Agent (Storyteller) - Synthesizes data into narratives (concise summaries or detailed reports)
"""
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from models import ReportMode
from tech_config import tech_settings


class WriterAgent(BaseAgent):
    """Writer Agent responsible for creating compelling narratives from research data"""
    
    def __init__(self):
        super().__init__("writer")
        self.logger = logging.getLogger("traffix.writer")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=tech_settings.openai_api_key,
            temperature=0.7,
            max_tokens=2000
        )
        self._setup_prompts()
    
    async def generate_quick_analysis(self, location: str, ritis_events: List[Dict], 
                                    news_articles: List[Dict], time_period: str) -> Dict[str, Any]:
        """Generate quick mode analysis - executive summary format"""
        try:
            self.logger.info(f"Generating quick analysis for {location}")
            
            # Prepare data for analysis
            analysis_data = {
                "location": location,
                "time_period": time_period,
                "ritis_events": ritis_events,
                "news_articles": news_articles,
                "total_events": len(ritis_events),
                "total_articles": len(news_articles)
            }
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(analysis_data)
            
            # Create traffic events summary
            traffic_summary = await self._create_traffic_events_summary(ritis_events)
            
            # Create causes and factors analysis
            causes_factors = await self._analyze_causes_factors(ritis_events, news_articles)
            
            # Create impact assessment
            impact_assessment = await self._assess_impacts(ritis_events, news_articles)
            
            # Create areas of concern
            areas_concern = await self._identify_areas_concern(ritis_events, news_articles)
            
            # Create recommendations
            recommendations = await self._generate_recommendations(ritis_events, news_articles)
            
            # List actual incidents
            incident_list = await self._create_incident_list(ritis_events)
            
            # Data sources
            data_sources = await self._list_data_sources(news_articles, ritis_events)
            
            return {
                "executive_summary": executive_summary,
                "traffic_events_summary": traffic_summary,
                "causes_factors": causes_factors,
                "impact_assessment": impact_assessment,
                "areas_concern": areas_concern,
                "recommendations": recommendations,
                "incident_list": incident_list,
                "data_sources": data_sources,
                "mode": "quick"
            }
            
        except Exception as e:
            self.logger.error(f"Quick analysis generation failed: {e}")
            return {"error": str(e), "mode": "quick"}
    
    async def generate_deep_analysis(self, location: str, ritis_events: List[Dict], 
                                   news_articles: List[Dict], time_period: str) -> Dict[str, Any]:
        """Generate deep mode analysis - comprehensive professional report"""
        try:
            self.logger.info(f"Generating deep analysis for {location}")
            
            # Prepare comprehensive data
            analysis_data = {
                "location": location,
                "time_period": time_period,
                "ritis_events": ritis_events,
                "news_articles": news_articles,
                "total_events": len(ritis_events),
                "total_articles": len(news_articles)
            }
            
            # Create comprehensive report
            comprehensive_report = await self._create_comprehensive_report(analysis_data)
            
            # Detailed event analysis
            detailed_events = await self._analyze_detailed_events(ritis_events)
            
            # Pattern analysis
            patterns = await self._analyze_patterns(ritis_events, news_articles)
            
            # Professional recommendations
            professional_recommendations = await self._generate_professional_recommendations(
                ritis_events, news_articles, location
            )
            
            return {
                "comprehensive_report": comprehensive_report,
                "detailed_events": detailed_events,
                "patterns": patterns,
                "professional_recommendations": professional_recommendations,
                "mode": "deep"
            }
            
        except Exception as e:
            self.logger.error(f"Deep analysis generation failed: {e}")
            return {"error": str(e), "mode": "deep"}
    
    def _setup_prompts(self):
        """Setup LLM prompts for different narrative types"""
        
        # Executive Summary Prompt
        self.executive_summary_prompt = PromptTemplate(
            input_variables=["research_data", "location", "query_type", "audience"],
            template="""
            Create an executive summary for transportation leadership based on the following research data:
            
            Location: {location}
            Query Type: {query_type}
            Target Audience: {audience}
            
            Research Data:
            {research_data}
            
            Requirements:
            - 2-3 paragraphs maximum
            - Focus on key findings and actionable insights
            - Use clear, professional language
            - Include specific metrics and data points
            - Highlight any urgent issues requiring attention
            - Provide clear recommendations
            
            Format the summary for executive consumption with bullet points for key points.
            """
        )
        
        # Detailed Narrative Prompt
        self.detailed_narrative_prompt = PromptTemplate(
            input_variables=["research_data", "location", "query_type", "narrative_style"],
            template="""
            Create a detailed narrative report based on the research data:
            
            Location: {location}
            Query Type: {query_type}
            Narrative Style: {narrative_style}
            
            Research Data:
            {research_data}
            
            Structure the narrative with:
            1. Introduction - Set the context and scope
            2. Key Findings - Present the main discoveries
            3. Analysis - Explain what the data means
            4. Evidence - Support findings with specific data
            5. Implications - Discuss the significance
            6. Recommendations - Provide actionable next steps
            7. Conclusion - Summarize key takeaways
            
            Use engaging, professional language that tells a compelling story about the traffic situation.
            Include specific examples and data points to support your narrative.
            """
        )
        
        # Incident Story Prompt
        self.incident_story_prompt = PromptTemplate(
            input_variables=["incidents", "causes", "location", "timeframe"],
            template="""
            Create a narrative about traffic incidents and their impact:
            
            Location: {location}
            Timeframe: {timeframe}
            
            Incidents:
            {incidents}
            
            Causes:
            {causes}
            
            Write a compelling story that:
            - Explains what happened and when
            - Describes the impact on traffic flow
            - Identifies the root causes
            - Explains the response and resolution
            - Provides lessons learned
            - Suggests prevention strategies
            
            Use a journalistic style that is informative yet engaging.
            """
        )
        
        # Pattern Analysis Prompt
        self.pattern_analysis_prompt = PromptTemplate(
            input_variables=["patterns", "trends", "location", "timeframe"],
            template="""
            Create a narrative about traffic patterns and trends:
            
            Location: {location}
            Timeframe: {timeframe}
            
            Patterns:
            {patterns}
            
            Trends:
            {trends}
            
            Write an analytical narrative that:
            - Explains the patterns discovered
            - Discusses trends and their implications
            - Identifies recurring issues
            - Suggests long-term solutions
            - Provides data-driven insights
            
            Use an analytical but accessible tone.
            """
        )
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute writing tasks"""
        research_data = input_data.get("research_data", {})
        location = input_data.get("location", "Unknown")
        query_type = input_data.get("query_type", "daily_summary")
        narrative_style = input_data.get("narrative_style", "professional")
        audience = input_data.get("audience", "analysts")
        
        self.logger.info(f"Writer agent creating narrative for {location} - {query_type}")
        
        try:
            # Step 1: Create executive summary
            executive_summary = await self._create_executive_summary(
                research_data, location, query_type, audience
            )
            
            # Step 2: Create detailed narrative
            detailed_narrative = await self._create_detailed_narrative(
                research_data, location, query_type, narrative_style
            )
            
            # Step 3: Create specialized narratives based on query type
            specialized_narratives = await self._create_specialized_narratives(
                research_data, location, query_type
            )
            
            # Step 4: Generate story elements
            story_elements = await self._generate_story_elements(
                research_data, location, query_type
            )
            
            # Step 5: Create narrative metadata
            narrative_metadata = self._create_narrative_metadata(
                research_data, location, query_type
            )
            
            return {
                "executive_summary": executive_summary,
                "detailed_narrative": detailed_narrative,
                "specialized_narratives": specialized_narratives,
                "story_elements": story_elements,
                "narrative_metadata": narrative_metadata,
                "writer_metadata": {
                    "location": location,
                    "query_type": query_type,
                    "narrative_style": narrative_style,
                    "audience": audience,
                    "generated_at": datetime.now().isoformat(),
                    "narrative_quality": self._assess_narrative_quality(executive_summary, detailed_narrative)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Writing task failed: {e}")
            raise
    
    async def _create_executive_summary(self, research_data: Dict[str, Any], 
                                      location: str, query_type: str, 
                                      audience: str) -> str:
        """Create executive summary for leadership"""
        
        try:
            # Prepare research data for prompt
            formatted_data = self._format_research_data_for_prompt(research_data)
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.executive_summary_prompt)
            
            # Generate summary
            summary = await chain.arun(
                research_data=formatted_data,
                location=location,
                query_type=query_type,
                audience=audience
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Executive summary creation failed: {e}")
            return f"Executive summary for {location} traffic analysis - {query_type} mode analysis completed."
    
    async def _create_detailed_narrative(self, research_data: Dict[str, Any], 
                                       location: str, query_type: str, 
                                       narrative_style: str) -> str:
        """Create detailed narrative report with optimization for deep mode"""
        
        try:
            # For deep mode, use optimized narrative creation
            if query_type in ["deep_analysis", "comprehensive"]:
                return await self._create_optimized_deep_narrative(research_data, location, narrative_style)
            
            # Standard narrative creation for other modes
            formatted_data = self._format_research_data_for_prompt(research_data)
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.detailed_narrative_prompt)
            
            # Generate narrative
            narrative = await chain.arun(
                research_data=formatted_data,
                location=location,
                query_type=query_type,
                narrative_style=narrative_style
            )
            
            return narrative
            
        except Exception as e:
            self.logger.error(f"Detailed narrative creation failed: {e}")
            return f"Detailed narrative for {location} traffic analysis - comprehensive report generated."
    
    async def _create_optimized_deep_narrative(self, research_data: Dict[str, Any], 
                                             location: str, narrative_style: str) -> str:
        """Create optimized narrative for deep mode focusing on summaries and accidents"""
        
        try:
            # Get key incidents with optimization
            key_incidents = research_data.get("key_incidents", [])
            
            # Separate summary, accidents, and other incidents
            summary_incidents = [inc for inc in key_incidents if inc.get("priority") == "summary"]
            accident_incidents = [inc for inc in key_incidents if inc.get("priority") == "high"]
            other_incidents = [inc for inc in key_incidents if inc.get("priority") == "medium"]
            
            # Create optimized narrative structure
            narrative_parts = []
            
            # 1. Event Summary Section
            if summary_incidents:
                summary_text = summary_incidents[0].get("description", "")
                narrative_parts.append(f"## Event Summary\n{summary_text}\n")
            
            # 2. Detailed Accident Analysis
            if accident_incidents:
                narrative_parts.append("## Detailed Accident Analysis")
                for accident in accident_incidents[:3]:  # Limit to top 3 accidents
                    accident_details = self._format_accident_details(accident)
                    narrative_parts.append(accident_details)
            
            # 3. Other Significant Events (brief)
            if other_incidents:
                narrative_parts.append("## Other Significant Events")
                other_summary = self._create_other_events_summary(other_incidents)
                narrative_parts.append(other_summary)
            
            # 4. Analysis and Insights
            cause_analysis = research_data.get("cause_analysis", {})
            if cause_analysis:
                insights = self._format_analysis_insights(cause_analysis, location)
                narrative_parts.append(f"## Analysis and Insights\n{insights}")
            
            # Combine all parts
            optimized_narrative = "\n\n".join(narrative_parts)
            
            self.logger.info(f"Created optimized deep narrative for {location} with {len(accident_incidents)} accidents")
            return optimized_narrative
            
        except Exception as e:
            self.logger.error(f"Optimized deep narrative creation failed: {e}")
            return f"Optimized analysis for {location} - focusing on key incidents and accidents."
    
    def _format_accident_details(self, accident: Dict[str, Any]) -> str:
        """Format detailed accident information using RITIS data structure"""
        
        details = []
        
        # Event identification
        details.append(f"**Event ID:** {accident.get('incident_id', 'Unknown')}")
        
        # Location context from RITIS
        location_parts = []
        if accident.get("location"):
            location_parts.append(accident['location'])
        if accident.get("road"):
            location_parts.append(f"on {accident['road']}")
        if accident.get("direction"):
            location_parts.append(f"({accident['direction']})")
        if location_parts:
            details.append(f"**Location:** {' '.join(location_parts)}")
        
        # Regional context
        if accident.get("region"):
            details.append(f"**Region:** {accident['region']}")
        if accident.get("county"):
            details.append(f"**County:** {accident['county']}")
        
        # Timing information
        if accident.get("start_time"):
            details.append(f"**Start Time:** {accident['start_time']}")
        if accident.get("end_time"):
            details.append(f"**End Time:** {accident['end_time']}")
        if accident.get("duration"):
            details.append(f"**Duration:** {accident['duration']}")
        
        # Event classification
        if accident.get("event_type"):
            details.append(f"**Event Type:** {accident['event_type']}")
        if accident.get("agency_specific_type"):
            details.append(f"**Specific Type:** {accident['agency_specific_type']}")
        
        # Status and impact
        if accident.get("open_closed"):
            details.append(f"**Status:** {accident['open_closed']}")
        if accident.get("roadway_clearance_time"):
            details.append(f"**Clearance Time:** {accident['roadway_clearance_time']}")
        
        # Description and notes
        if accident.get("description"):
            details.append(f"**Description:** {accident['description']}")
        if accident.get("operator_notes"):
            details.append(f"**Notes:** {accident['operator_notes']}")
        
        return "\n".join(details) + "\n"
    
    def _create_other_events_summary(self, other_incidents: List[Dict[str, Any]]) -> str:
        """Create summary of other significant events"""
        
        if not other_incidents:
            return "No other significant events reported."
        
        summary_parts = []
        for incident in other_incidents[:5]:  # Limit to 5 other events
            summary_parts.append(f"- {incident.get('description', 'No description')} ({incident.get('severity', 'Unknown severity')})")
        
        return "\n".join(summary_parts)
    
    def _format_analysis_insights(self, cause_analysis: Dict[str, Any], location: str) -> str:
        """Format analysis insights"""
        
        insights = []
        
        primary_causes = cause_analysis.get("primary_causes", [])
        if primary_causes:
            insights.append(f"**Primary Causes:** {', '.join(primary_causes[:3])}")
        
        contributing_factors = cause_analysis.get("contributing_factors", [])
        if contributing_factors:
            insights.append(f"**Contributing Factors:** {', '.join(contributing_factors[:3])}")
        
        confidence = cause_analysis.get("cause_confidence", 0)
        insights.append(f"**Analysis Confidence:** {confidence:.1%}")
        
        return "\n".join(insights)
    
    async def _create_specialized_narratives(self, research_data: Dict[str, Any], 
                                           location: str, query_type: str) -> Dict[str, str]:
        """Create specialized narratives based on query type"""
        
        specialized_narratives = {}
        
        try:
            # Incident narrative
            if query_type in ["incident_analysis", "anomaly_investigation"]:
                incidents = research_data.get("key_incidents", [])
                causes = research_data.get("cause_analysis", {})
                
                if incidents:
                    chain = LLMChain(llm=self.llm, prompt=self.incident_story_prompt)
                    incident_narrative = await chain.arun(
                        incidents=self._format_incidents_for_prompt(incidents),
                        causes=self._format_causes_for_prompt(causes),
                        location=location,
                        timeframe="recent"
                    )
                    specialized_narratives["incident_narrative"] = incident_narrative
            
            # Pattern analysis narrative
            if query_type == "pattern_analysis":
                patterns = research_data.get("pattern_analysis", {})
                trends = research_data.get("research_insights", {})
                
                if patterns:
                    chain = LLMChain(llm=self.llm, prompt=self.pattern_analysis_prompt)
                    pattern_narrative = await chain.arun(
                        patterns=self._format_patterns_for_prompt(patterns),
                        trends=self._format_trends_for_prompt(trends),
                        location=location,
                        timeframe="analysis_period"
                    )
                    specialized_narratives["pattern_narrative"] = pattern_narrative
            
        except Exception as e:
            self.logger.error(f"Specialized narrative creation failed: {e}")
        
        return specialized_narratives
    
    async def _generate_story_elements(self, research_data: Dict[str, Any], 
                                     location: str, query_type: str) -> List[Dict[str, Any]]:
        """Generate individual story elements"""
        
        story_elements = []
        
        # Introduction element
        introduction = self._create_introduction_element(research_data, location, query_type)
        story_elements.append(introduction)
        
        # Key findings element
        key_findings = self._create_key_findings_element(research_data, location)
        story_elements.append(key_findings)
        
        # Analysis element
        analysis = self._create_analysis_element(research_data, location)
        story_elements.append(analysis)
        
        # Evidence element
        evidence = self._create_evidence_element(research_data, location)
        story_elements.append(evidence)
        
        # Recommendations element
        recommendations = self._create_recommendations_element(research_data, location)
        story_elements.append(recommendations)
        
        return story_elements
    
    def _create_introduction_element(self, research_data: Dict[str, Any], 
                                   location: str, query_type: str) -> Dict[str, Any]:
        """Create introduction story element"""
        
        data_quality = research_data.get("research_metadata", {}).get("data_quality_score", 0.5)
        incident_count = len(research_data.get("key_incidents", []))
        
        content = f"""
        Traffic analysis for {location} reveals significant patterns requiring attention. 
        Based on comprehensive data collection from multiple sources, this analysis 
        identifies {incident_count} key incidents and provides insights into traffic 
        patterns and their underlying causes. The analysis demonstrates 
        {'high' if data_quality > 0.7 else 'moderate'} data quality with reliable findings.
        """
        
        return {
            "element_type": "introduction",
            "content": content.strip(),
            "supporting_data": [
                {"type": "location", "value": location},
                {"type": "incident_count", "value": str(incident_count)},
                {"type": "data_quality", "value": f"{data_quality:.2f}"}
            ],
            "confidence": data_quality
        }
    
    def _create_key_findings_element(self, research_data: Dict[str, Any], 
                                   location: str) -> Dict[str, Any]:
        """Create key findings story element"""
        
        cause_analysis = research_data.get("cause_analysis", {})
        primary_causes = cause_analysis.get("primary_causes", [])
        contributing_factors = cause_analysis.get("contributing_factors", [])
        
        content = f"""
        Key findings for {location} traffic analysis:
        """
        
        if primary_causes:
            content += f"\n• Primary causes identified: {', '.join(primary_causes[:3])}"
        
        if contributing_factors:
            content += f"\n• Contributing factors: {', '.join(contributing_factors[:3])}"
        
        supporting_data = []
        for cause in primary_causes[:3]:
            supporting_data.append({"type": "primary_cause", "value": cause})
        
        return {
            "element_type": "key_findings",
            "content": content.strip(),
            "supporting_data": supporting_data,
            "confidence": cause_analysis.get("cause_confidence", 0.5)
        }
    
    def _create_analysis_element(self, research_data: Dict[str, Any], 
                               location: str) -> Dict[str, Any]:
        """Create analysis story element"""
        
        pattern_analysis = research_data.get("pattern_analysis", {})
        patterns_detected = pattern_analysis.get("patterns_detected", False)
        
        content = f"""
        Analysis of {location} traffic data reveals {'significant patterns' if patterns_detected else 'standard traffic patterns'}. 
        The data indicates various factors influencing traffic flow, with both immediate 
        and underlying causes contributing to observed patterns.
        """
        
        if patterns_detected:
            content += " Recurring patterns suggest systematic issues requiring attention."
        
        supporting_data = [
            {"type": "pattern_analysis", "value": "patterns_detected" if patterns_detected else "no_patterns"},
            {"type": "analysis_type", "value": "comprehensive"}
        ]
        
        return {
            "element_type": "analysis",
            "content": content.strip(),
            "supporting_data": supporting_data,
            "confidence": 0.8 if patterns_detected else 0.6
        }
    
    def _create_evidence_element(self, research_data: Dict[str, Any], 
                               location: str) -> Dict[str, Any]:
        """Create evidence story element"""
        
        key_incidents = research_data.get("key_incidents", [])
        evidence_support = research_data.get("cause_analysis", {}).get("evidence_support", [])
        
        content = f"""
        Evidence supporting this analysis includes {len(key_incidents)} documented incidents 
        and {len(evidence_support)} supporting data points. The evidence demonstrates 
        clear correlations between identified causes and observed traffic impacts.
        """
        
        supporting_data = [
            {"type": "incident_count", "value": str(len(key_incidents))},
            {"type": "evidence_count", "value": str(len(evidence_support))},
            {"type": "data_sources", "value": "multiple"}
        ]
        
        return {
            "element_type": "evidence",
            "content": content.strip(),
            "supporting_data": supporting_data,
            "confidence": 0.7
        }
    
    def _create_recommendations_element(self, research_data: Dict[str, Any], 
                                      location: str) -> Dict[str, Any]:
        """Create recommendations story element"""
        
        recommendations = research_data.get("research_insights", {}).get("recommendations", [])
        
        content = f"""
        Based on this analysis, the following recommendations are proposed for {location}:
        """
        
        for i, rec in enumerate(recommendations[:3], 1):
            content += f"\n{i}. {rec}"
        
        supporting_data = []
        for rec in recommendations[:3]:
            supporting_data.append({"type": "recommendation", "value": rec})
        
        return {
            "element_type": "recommendations",
            "content": content.strip(),
            "supporting_data": supporting_data,
            "confidence": 0.8
        }
    
    def _format_research_data_for_prompt(self, research_data: Dict[str, Any]) -> str:
        """Format research data for LLM prompts"""
        
        formatted = "Research Data Summary:\n\n"
        
        # Add key incidents
        key_incidents = research_data.get("key_incidents", [])
        if key_incidents:
            formatted += f"Key Incidents ({len(key_incidents)}):\n"
            for incident in key_incidents[:5]:  # Top 5 incidents
                formatted += f"- {incident.get('description', 'Unknown')} ({incident.get('severity', 'Unknown')})\n"
            formatted += "\n"
        
        # Add cause analysis
        cause_analysis = research_data.get("cause_analysis", {})
        if cause_analysis:
            formatted += "Cause Analysis:\n"
            formatted += f"Primary Causes: {', '.join(cause_analysis.get('primary_causes', []))}\n"
            formatted += f"Contributing Factors: {', '.join(cause_analysis.get('contributing_factors', []))}\n"
            formatted += f"Confidence: {cause_analysis.get('cause_confidence', 0):.2f}\n\n"
        
        # Add research insights
        insights = research_data.get("research_insights", {})
        if insights:
            formatted += "Research Insights:\n"
            formatted += f"Summary: {insights.get('summary', 'No summary available')}\n"
            formatted += f"Key Findings: {', '.join(insights.get('key_findings', []))}\n"
            formatted += f"Recommendations: {', '.join(insights.get('recommendations', []))}\n"
        
        return formatted
    
    def _format_incidents_for_prompt(self, incidents: List[Dict[str, Any]]) -> str:
        """Format incidents for prompt"""
        
        formatted = "Traffic Incidents:\n"
        for incident in incidents[:5]:  # Top 5 incidents
            formatted += f"- {incident.get('description', 'Unknown')} "
            formatted += f"({incident.get('severity', 'Unknown')}) "
            formatted += f"at {incident.get('location', 'Unknown location')}\n"
        
        return formatted
    
    def _format_causes_for_prompt(self, causes: Dict[str, Any]) -> str:
        """Format causes for prompt"""
        
        formatted = "Identified Causes:\n"
        formatted += f"Primary: {', '.join(causes.get('primary_causes', []))}\n"
        formatted += f"Contributing: {', '.join(causes.get('contributing_factors', []))}\n"
        
        return formatted
    
    def _format_patterns_for_prompt(self, patterns: Dict[str, Any]) -> str:
        """Format patterns for prompt"""
        
        formatted = "Traffic Patterns:\n"
        if patterns.get("temporal_patterns"):
            formatted += f"Temporal: {len(patterns['temporal_patterns'])} hourly patterns identified\n"
        if patterns.get("spatial_patterns"):
            formatted += f"Spatial: {len(patterns['spatial_patterns'])} location patterns identified\n"
        
        return formatted
    
    def _format_trends_for_prompt(self, trends: Dict[str, Any]) -> str:
        """Format trends for prompt"""
        
        formatted = "Trends and Insights:\n"
        formatted += f"Key Findings: {', '.join(trends.get('key_findings', []))}\n"
        formatted += f"Recommendations: {', '.join(trends.get('recommendations', []))}\n"
        
        return formatted
    
    def _create_narrative_metadata(self, research_data: Dict[str, Any], 
                                 location: str, query_type: str) -> Dict[str, Any]:
        """Create metadata for the narrative"""
        
        return {
            "narrative_type": query_type,
            "location": location,
            "data_sources": list(research_data.keys()),
            "incident_count": len(research_data.get("key_incidents", [])),
            "cause_confidence": research_data.get("cause_analysis", {}).get("cause_confidence", 0.5),
            "pattern_detected": research_data.get("pattern_analysis", {}).get("patterns_detected", False),
            "generated_at": datetime.now().isoformat()
        }
    
    def _assess_narrative_quality(self, executive_summary: str, 
                                detailed_narrative: str) -> float:
        """Assess the quality of generated narratives"""
        
        # Simple quality assessment based on length and content
        summary_quality = min(1.0, len(executive_summary) / 500)  # Expect ~500 chars
        narrative_quality = min(1.0, len(detailed_narrative) / 2000)  # Expect ~2000 chars
        
        return (summary_quality + narrative_quality) / 2
    
    # Helper methods for quick and deep analysis
    
    async def _create_executive_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Create executive summary for quick mode"""
        prompt = f"""
        Create an executive summary for traffic analysis in {analysis_data['location']} for {analysis_data['time_period']}.
        
        Data: {analysis_data['total_events']} RITIS events, {analysis_data['total_articles']} news articles
        
        Provide a 2-3 paragraph overview of the traffic situation, key incidents, and main concerns.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _create_traffic_events_summary(self, ritis_events: List[Dict]) -> str:
        """Create traffic events summary"""
        if not ritis_events:
            return "No RITIS events found for the specified time period."
        
        event_types = {}
        for event in ritis_events:
            event_type = event.get('event_type', 'Unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        summary = f"Traffic Events Summary:\n"
        summary += f"Total Events: {len(ritis_events)}\n"
        summary += f"Event Types: {', '.join([f'{k}: {v}' for k, v in event_types.items()])}\n"
        
        return summary
    
    async def _analyze_causes_factors(self, ritis_events: List[Dict], news_articles: List[Dict]) -> str:
        """Analyze causes and factors"""
        prompt = f"""
        Analyze the causes and contributing factors for traffic incidents based on:
        - {len(ritis_events)} RITIS events
        - {len(news_articles)} news articles
        
        Identify common patterns, weather conditions, construction impacts, and other factors.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _assess_impacts(self, ritis_events: List[Dict], news_articles: List[Dict]) -> str:
        """Assess traffic impacts"""
        prompt = f"""
        Assess the traffic impacts based on {len(ritis_events)} RITIS events and {len(news_articles)} news articles.
        Consider delays, congestion, safety concerns, and economic impacts.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _identify_areas_concern(self, ritis_events: List[Dict], news_articles: List[Dict]) -> str:
        """Identify areas of concern"""
        prompt = f"""
        Identify areas of concern based on traffic data:
        - {len(ritis_events)} RITIS events
        - {len(news_articles)} news articles
        
        Focus on recurring issues, high-risk locations, and systemic problems.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _generate_recommendations(self, ritis_events: List[Dict], news_articles: List[Dict]) -> str:
        """Generate recommendations"""
        prompt = f"""
        Generate actionable recommendations for traffic management based on:
        - {len(ritis_events)} RITIS events
        - {len(news_articles)} news articles
        
        Provide specific, implementable recommendations for road operators.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _create_incident_list(self, ritis_events: List[Dict]) -> str:
        """Create list of actual incidents"""
        if not ritis_events:
            return "No incidents found."
        
        incident_list = "RITIS Incidents:\n"
        for i, event in enumerate(ritis_events[:10], 1):  # Limit to 10 for readability
            incident_list += f"{i}. {event.get('event_type', 'Unknown')} - {event.get('location', 'Unknown location')}\n"
            incident_list += f"   Time: {event.get('timestamp', 'Unknown')}\n"
            incident_list += f"   Highway: {event.get('highway', 'N/A')}\n\n"
        
        return incident_list
    
    async def _list_data_sources(self, news_articles: List[Dict], ritis_events: List[Dict]) -> str:
        """List data sources"""
        sources = "Data Sources:\n"
        sources += f"- RITIS Database: {len(ritis_events)} traffic events\n"
        sources += f"- News Articles: {len(news_articles)} articles\n"
        
        if news_articles:
            unique_sources = set()
            for article in news_articles:
                source = article.get('source', 'Unknown')
                unique_sources.add(source)
            sources += f"- News Sources: {', '.join(unique_sources)}\n"
        
        return sources
    
    async def _create_comprehensive_report(self, analysis_data: Dict[str, Any]) -> str:
        """Create comprehensive report for deep mode in formal paragraph format"""
        
        # Get sample events to include in the prompt
        sample_events = analysis_data.get('ritis_events', [])[:10]
        event_details = ""
        for event in sample_events:
            event_details += f"- {event.get('event_type', 'Unknown')} at {event.get('location', 'Unknown location')} "
            event_details += f"on {event.get('road', 'Unknown road')}, "
            event_details += f"started {event.get('start_time', 'Unknown time')}\n"
        
        prompt = f"""
        You are a senior transportation analyst writing a formal research report.
        
        Location: {analysis_data['location']}
        Time Period: {analysis_data['time_period']}
        Data Available: {analysis_data['total_events']} RITIS traffic events, {analysis_data['total_articles']} news articles
        
        CRITICAL FORMATTING RULES:
        - Write ONLY in full paragraphs (4-6 sentences each)
        - ABSOLUTELY NO bullet points, NO lists, NO dashes
        - Use markdown headers (##, ###) to organize sections
        - Write flowing, connected prose with transition sentences
        - Cite specific incidents with locations, times, and details
        - Maintain formal, professional tone throughout
        
        REQUIRED STRUCTURE:
        
        ## Executive Summary
        [2-3 paragraphs synthesizing key findings]
        
        ## Current Traffic Conditions
        [3-4 paragraphs with detailed situation analysis]
        
        ## Major Incidents Analysis  
        [Detailed paragraphs on significant events]
        
        ## Traffic Patterns and Trends
        [Analytical paragraphs on observed patterns]
        
        ## Impact Assessment
        [Paragraphs evaluating broader impacts]
        
        ## Recommendations and Solutions
        [Strategic paragraphs with actionable recommendations]
        
        ## Conclusion
        [2-3 concluding paragraphs]
        
        Sample RITIS Events to Reference:
        {event_details}
        
        Write minimum 1000 words. Every section must be in paragraph format only.
        Think of this as a professional traffic study report.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _analyze_detailed_events(self, ritis_events: List[Dict]) -> str:
        """Analyze detailed events for deep mode"""
        if not ritis_events:
            return "No detailed events to analyze."
        
        prompt = f"""
        Provide detailed analysis of {len(ritis_events)} RITIS traffic events.
        Include specific examples with timestamps, locations, and impacts.
        Format like: "On [date], around [time], a [event_type] occurred at [location]..."
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _analyze_patterns(self, ritis_events: List[Dict], news_articles: List[Dict]) -> str:
        """Analyze patterns for deep mode"""
        prompt = f"""
        Analyze traffic patterns based on {len(ritis_events)} RITIS events and {len(news_articles)} news articles.
        Identify temporal patterns, geographic clusters, and recurring issues.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _generate_professional_recommendations(self, ritis_events: List[Dict], 
                                                   news_articles: List[Dict], location: str) -> str:
        """Generate professional recommendations for deep mode"""
        prompt = f"""
        Generate professional recommendations for {location} based on traffic analysis.
        Consider {len(ritis_events)} RITIS events and {len(news_articles)} news articles.
        
        Provide detailed, actionable recommendations for road operators and traffic management.
        Include implementation priorities and expected outcomes.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content
