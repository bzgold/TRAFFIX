"""
Research Agent (Analyst) - Queries RITIS + Tavily, extracts key incidents, identifies causes
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

from agents.base_agent import BaseAgent
from services.data_services import DataIntegrationService
from services.vector_service import VectorService
from models import ReportMode


class ResearchAgent(BaseAgent):
    """Research Agent responsible for data collection and analysis"""
    
    def __init__(self, vector_service=None):
        super().__init__("research")
        self.logger = logging.getLogger("traffix.research")
        self.data_service = DataIntegrationService()
        self.vector_service = vector_service or VectorService()
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research tasks"""
        location = input_data.get("location", "Unknown")
        query_type = input_data.get("query_type", "daily_summary")
        complexity = input_data.get("complexity", "medium")
        required_sources = input_data.get("required_sources", ["ritis", "news"])
        
        self.logger.info(f"Research agent analyzing {location} for {query_type}")
        
        try:
            # Step 1: Collect data from all required sources
            collected_data = await self._collect_comprehensive_data(
                location, required_sources, complexity
            )
            
            # Step 2: Extract key incidents and events
            key_incidents = await self._extract_key_incidents(collected_data, query_type)
            
            # Step 3: Identify causes and contributing factors
            cause_analysis = await self._identify_causes(collected_data, key_incidents, query_type)
            
            # Step 4: Perform pattern analysis if needed
            pattern_analysis = await self._analyze_patterns(collected_data, query_type, complexity)
            
            # Step 5: Generate research insights
            research_insights = await self._generate_research_insights(
                collected_data, key_incidents, cause_analysis, pattern_analysis
            )
            
            # Step 6: Store findings in vector database
            await self._store_research_findings(location, research_insights)
            
            return {
                "collected_data": collected_data,
                "key_incidents": key_incidents,
                "cause_analysis": cause_analysis,
                "pattern_analysis": pattern_analysis,
                "research_insights": research_insights,
                "research_metadata": {
                    "location": location,
                    "query_type": query_type,
                    "complexity": complexity,
                    "sources_used": required_sources,
                    "research_timestamp": datetime.now().isoformat(),
                    "data_quality_score": self._calculate_data_quality_score(collected_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Research analysis failed: {e}")
            raise
    
    async def _collect_comprehensive_data(self, location: str, 
                                        required_sources: List[str], 
                                        complexity: str) -> Dict[str, Any]:
        """Collect data from all required sources"""
        
        self.logger.info(f"Collecting data from sources: {required_sources}")
        
        # Determine time range based on complexity
        time_range_hours = self._get_time_range_for_complexity(complexity)
        
        # Collect data using the data integration service
        collected_data = await self.data_service.collect_all_data(
            location=location,
            time_range_hours=time_range_hours,
            mode="deep" if complexity in ["high", "critical"] else "quick"
        )
        
        # Filter data based on required sources
        filtered_data = self._filter_data_by_sources(collected_data, required_sources)
        
        return filtered_data
    
    async def _extract_key_incidents(self, collected_data: Dict[str, Any], 
                                   query_type: str) -> List[Dict[str, Any]]:
        """Extract key incidents and events from collected data"""
        
        incidents = collected_data.get("incidents", [])
        news_articles = collected_data.get("news_articles", [])
        
        key_incidents = []
        
        # Process traffic incidents
        for incident in incidents:
            incident_analysis = self._analyze_incident(incident, query_type)
            if incident_analysis["relevance_score"] > 0.5:
                key_incidents.append(incident_analysis)
        
        # Process news incidents
        for article in news_articles:
            if self._is_incident_related(article):
                news_incident = self._extract_incident_from_news(article)
                if news_incident["relevance_score"] > 0.5:
                    key_incidents.append(news_incident)
        
        # Sort by relevance and impact
        key_incidents.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Optimize for deep mode: summarize events and focus on accidents
        if query_type in ["deep_analysis", "comprehensive"]:
            key_incidents = await self._optimize_incidents_for_deep_mode(key_incidents, collected_data)
        
        return key_incidents[:10]  # Return top 10 most relevant incidents
    
    async def _optimize_incidents_for_deep_mode(self, key_incidents: List[Dict[str, Any]], 
                                             collected_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize incidents for deep mode using RITIS column information"""
        
        try:
            # First, create a summary of all events
            all_incidents = collected_data.get("incidents", [])
            event_summary = await self._create_event_summary(all_incidents)
            
            # Categorize incidents using RITIS data structure
            accident_incidents = []
            construction_incidents = []
            open_incidents = []
            high_duration_incidents = []
            other_high_priority = []
            
            for incident in key_incidents:
                # Use RITIS standardized type for categorization
                event_type = self._categorize_event_type(incident)
                priority = self._get_event_priority(incident)
                
                if event_type == "Accidents":
                    incident["priority"] = "high"
                    incident["detailed_analysis"] = True
                    incident["category"] = "accident"
                    accident_incidents.append(incident)
                elif event_type == "Construction":
                    incident["priority"] = "medium"
                    incident["detailed_analysis"] = True
                    incident["category"] = "construction"
                    construction_incidents.append(incident)
                elif incident.get("open_closed", "").upper() == "OPEN":
                    incident["priority"] = "high"
                    incident["detailed_analysis"] = True
                    incident["category"] = "open"
                    open_incidents.append(incident)
                elif priority == "high":
                    incident["priority"] = "medium"
                    incident["detailed_analysis"] = False
                    incident["category"] = "high_duration"
                    high_duration_incidents.append(incident)
                elif incident.get("relevance_score", 0) > 0.8:
                    incident["priority"] = "medium"
                    incident["detailed_analysis"] = False
                    incident["category"] = "other"
                    other_high_priority.append(incident)
            
            # Add summary as first item
            summary_incident = {
                "event_id": "summary",
                "description": f"Event Summary: {event_summary}",
                "relevance_score": 1.0,
                "priority": "summary",
                "detailed_analysis": False,
                "category": "summary",
                "timestamp": datetime.now().isoformat(),
                "source": "summary"
            }
            
            # Prioritize incidents: summary + accidents + open + construction + high duration + other
            optimized_incidents = [summary_incident] + \
                                accident_incidents[:5] + \
                                open_incidents[:3] + \
                                construction_incidents[:3] + \
                                high_duration_incidents[:2] + \
                                other_high_priority[:2]
            
            self.logger.info(f"Optimized incidents: {len(optimized_incidents)} total "
                           f"({len(accident_incidents)} accidents, {len(open_incidents)} open, "
                           f"{len(construction_incidents)} construction, {len(high_duration_incidents)} high-duration)")
            return optimized_incidents
            
        except Exception as e:
            self.logger.error(f"Incident optimization failed: {e}")
            return key_incidents
    
    async def _create_event_summary(self, incidents: List[Dict[str, Any]]) -> str:
        """Create a summary of all events by type and severity"""
        
        try:
            # Group events by type
            event_types = {}
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            
            for incident in incidents:
                event_type = self._categorize_event_type(incident.get("description", ""))
                severity = incident.get("severity", "low")
                
                if event_type not in event_types:
                    event_types[event_type] = 0
                event_types[event_type] += 1
                
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            # Create summary text
            summary_parts = []
            
            if event_types:
                summary_parts.append(f"Total events: {sum(event_types.values())}")
                for event_type, count in event_types.items():
                    summary_parts.append(f"{event_type}: {count}")
            
            if any(severity_counts.values()):
                summary_parts.append(f"Severity breakdown - High: {severity_counts['high']}, Medium: {severity_counts['medium']}, Low: {severity_counts['low']}")
            
            return "; ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Event summary creation failed: {e}")
            return "Event summary unavailable"
    
    def _categorize_event_type(self, incident: Dict[str, Any]) -> str:
        """Categorize event using RITIS standardized type and agency-specific information"""
        
        # Use RITIS standardized type first
        standardized_type = incident.get("event_type", "").upper()
        agency_specific_type = incident.get("agency_specific_type", "").upper()
        edc_incident_type = incident.get("edc_incident_type", "").upper()
        
        # Priority categorization based on RITIS structure
        if any(word in standardized_type for word in ["COLLISION", "ACCIDENT", "CRASH"]):
            return "Accidents"
        elif any(word in standardized_type for word in ["CONSTRUCTION", "PAVING", "ROADWORK", "BRIDGE MAINTENANCE"]):
            return "Construction"
        elif any(word in standardized_type for word in ["CONGESTION", "TRAFFIC CONGESTION", "DELAYS"]):
            return "Congestion"
        elif any(word in standardized_type for word in ["DISABLED VEHICLE", "BREAKDOWN"]):
            return "Disabled Vehicles"
        elif any(word in standardized_type for word in ["WEATHER", "FOG", "WINTER"]):
            return "Weather"
        elif any(word in standardized_type for word in ["OBSTRUCTION", "OBSTRUCTIONS"]):
            return "Obstructions"
        elif any(word in standardized_type for word in ["TRAFFIC SIGNAL", "SIGNAL"]):
            return "Traffic Signals"
        else:
            return "Other"
    
    def _get_event_priority(self, incident: Dict[str, Any]) -> str:
        """Determine event priority based on RITIS data"""
        
        # Check if event is still open
        if incident.get("open_closed", "").upper() == "OPEN":
            return "high"
        
        # Check duration for impact assessment
        duration = incident.get("duration", "")
        if "hour" in duration.lower() and any(word in duration for word in ["2", "3", "4", "5", "6", "7", "8", "9"]):
            return "high"
        elif "hour" in duration.lower():
            return "medium"
        else:
            return "low"
    
    def _extract_location_context(self, incident: Dict[str, Any]) -> Dict[str, str]:
        """Extract location context from RITIS data"""
        return {
            "specific_location": incident.get("location", ""),
            "region": incident.get("region", ""),
            "road": incident.get("road", ""),
            "direction": incident.get("direction", ""),
            "county": incident.get("county", ""),
            "state": incident.get("state", "")
        }
    
    async def _identify_causes(self, collected_data: Dict[str, Any], 
                             key_incidents: List[Dict[str, Any]], 
                             query_type: str) -> Dict[str, Any]:
        """Identify root causes and contributing factors"""
        
        cause_analysis = {
            "primary_causes": [],
            "contributing_factors": [],
            "cause_confidence": 0.0,
            "evidence_support": []
        }
        
        # Analyze traffic data for patterns
        traffic_data = collected_data.get("traffic_data", [])
        if traffic_data:
            traffic_causes = self._analyze_traffic_patterns(traffic_data)
            cause_analysis["primary_causes"].extend(traffic_causes["primary_causes"])
            cause_analysis["contributing_factors"].extend(traffic_causes["contributing_factors"])
        
        # Analyze incidents for causes
        for incident in key_incidents:
            incident_causes = self._extract_incident_causes(incident)
            cause_analysis["primary_causes"].extend(incident_causes["primary_causes"])
            cause_analysis["contributing_factors"].extend(incident_causes["contributing_factors"])
            cause_analysis["evidence_support"].extend(incident_causes["evidence"])
        
        # Analyze weather impact
        weather_data = collected_data.get("weather_data", [])
        if weather_data:
            weather_causes = self._analyze_weather_impact(weather_data)
            cause_analysis["contributing_factors"].extend(weather_causes)
        
        # Remove duplicates and calculate confidence
        cause_analysis["primary_causes"] = list(set(cause_analysis["primary_causes"]))
        cause_analysis["contributing_factors"] = list(set(cause_analysis["contributing_factors"]))
        cause_analysis["cause_confidence"] = self._calculate_cause_confidence(cause_analysis)
        
        return cause_analysis
    
    async def _analyze_patterns(self, collected_data: Dict[str, Any], 
                              query_type: str, complexity: str) -> Dict[str, Any]:
        """Analyze patterns in the data"""
        
        if complexity not in ["high", "critical"] and query_type != "pattern_analysis":
            return {"patterns_detected": False}
        
        pattern_analysis = {
            "temporal_patterns": {},
            "spatial_patterns": {},
            "causal_patterns": {},
            "patterns_detected": False
        }
        
        # Analyze temporal patterns
        traffic_data = collected_data.get("traffic_data", [])
        if traffic_data:
            temporal_patterns = self._analyze_temporal_patterns(traffic_data)
            pattern_analysis["temporal_patterns"] = temporal_patterns
        
        # Analyze incident patterns
        incidents = collected_data.get("incidents", [])
        if incidents:
            incident_patterns = self._analyze_incident_patterns(incidents)
            pattern_analysis["spatial_patterns"] = incident_patterns
        
        # Determine if patterns were detected
        pattern_analysis["patterns_detected"] = any([
            pattern_analysis["temporal_patterns"],
            pattern_analysis["spatial_patterns"],
            pattern_analysis["causal_patterns"]
        ])
        
        return pattern_analysis
    
    async def _generate_research_insights(self, collected_data: Dict[str, Any],
                                        key_incidents: List[Dict[str, Any]],
                                        cause_analysis: Dict[str, Any],
                                        pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research insights"""
        
        insights = {
            "summary": self._create_research_summary(collected_data, key_incidents),
            "key_findings": self._extract_key_findings(cause_analysis, pattern_analysis),
            "data_quality": self._assess_data_quality(collected_data),
            "confidence_level": self._calculate_research_confidence(cause_analysis, pattern_analysis),
            "recommendations": self._generate_research_recommendations(cause_analysis, pattern_analysis),
            "limitations": self._identify_research_limitations(collected_data)
        }
        
        return insights
    
    async def _store_research_findings(self, location: str, research_insights: Dict[str, Any]):
        """Store research findings in vector database for future reference"""
        
        try:
            # Create research document
            research_doc = {
                "location": location,
                "insights": research_insights,
                "timestamp": datetime.now().isoformat(),
                "type": "research_findings"
            }
            
            # Store in vector database
            await self.vector_service.add_traffic_data([research_doc], location, "research")
            
            self.logger.info("Research findings stored in vector database")
            
        except Exception as e:
            self.logger.error(f"Failed to store research findings: {e}")
    
    def _get_time_range_for_complexity(self, complexity: str) -> int:
        """Get time range in hours based on complexity"""
        time_ranges = {
            "low": 24,
            "medium": 48,
            "high": 168,  # 1 week
            "critical": 336  # 2 weeks
        }
        return time_ranges.get(complexity, 48)
    
    def _filter_data_by_sources(self, collected_data: Dict[str, Any], 
                               required_sources: List[str]) -> Dict[str, Any]:
        """Filter collected data based on required sources"""
        
        filtered_data = {}
        
        source_mapping = {
            "ritis": ["traffic_data", "incidents"],
            "news": ["news_articles"],
            "weather": ["weather_data"],
            "social": ["social_posts"],
            "incidents": ["incidents"]
        }
        
        for source in required_sources:
            if source in source_mapping:
                for data_key in source_mapping[source]:
                    if data_key in collected_data:
                        filtered_data[data_key] = collected_data[data_key]
        
        # Always include metadata
        filtered_data["collection_metadata"] = collected_data.get("collection_metadata", {})
        
        return filtered_data
    
    def _analyze_incident(self, incident: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Analyze a single incident for relevance and impact"""
        
        relevance_score = 0.5  # Base relevance
        
        # Increase relevance based on severity
        severity = incident.get("severity", "minor").lower()
        severity_scores = {"minor": 0.3, "moderate": 0.6, "major": 0.8, "critical": 1.0}
        relevance_score += severity_scores.get(severity, 0.3)
        
        # Increase relevance based on query type
        if query_type == "incident_analysis" and "incident" in incident.get("description", "").lower():
            relevance_score += 0.3
        
        # Calculate impact score
        impact_score = self._calculate_incident_impact(incident)
        
        return {
            "incident_id": incident.get("incident_id", ""),
            "description": incident.get("description", ""),
            "severity": severity,
            "location": incident.get("location", ""),
            "start_time": incident.get("start_time", ""),
            "end_time": incident.get("end_time", ""),
            "relevance_score": min(1.0, relevance_score),
            "impact_score": impact_score,
            "type": "traffic_incident"
        }
    
    def _is_incident_related(self, article: Dict[str, Any]) -> bool:
        """Check if a news article is incident-related"""
        
        title = article.get("title", "").lower()
        content = article.get("content", "").lower()
        
        incident_keywords = [
            "accident", "crash", "incident", "collision", "breakdown",
            "closure", "delay", "congestion", "jam", "backup"
        ]
        
        return any(keyword in title or keyword in content for keyword in incident_keywords)
    
    def _extract_incident_from_news(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract incident information from news article"""
        
        return {
            "incident_id": f"news_{article.get('url', '')[-8:]}",
            "description": article.get("title", ""),
            "severity": "moderate",  # Default for news incidents
            "location": "Unknown",
            "start_time": article.get("published_at", ""),
            "end_time": article.get("published_at", ""),
            "relevance_score": article.get("relevance_score", 0.5),
            "impact_score": 0.6,  # Default impact for news
            "type": "news_incident",
            "source": article.get("source", ""),
            "url": article.get("url", "")
        }
    
    def _analyze_traffic_patterns(self, traffic_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze traffic data for patterns and causes"""
        
        if not traffic_data:
            return {"primary_causes": [], "contributing_factors": []}
        
        # Calculate average metrics
        speeds = [item.get("speed", 0) for item in traffic_data]
        volumes = [item.get("volume", 0) for item in traffic_data]
        occupancies = [item.get("occupancy", 0) for item in traffic_data]
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        avg_occupancy = sum(occupancies) / len(occupancies) if occupancies else 0
        
        primary_causes = []
        contributing_factors = []
        
        # Analyze speed patterns
        if avg_speed < 30:  # Low speed indicates congestion
            primary_causes.append("Traffic congestion")
            if avg_occupancy > 0.7:
                contributing_factors.append("High vehicle occupancy")
        
        # Analyze volume patterns
        if avg_volume > 1500:  # High volume
            contributing_factors.append("High traffic volume")
        
        # Check for incidents
        incident_count = sum(1 for item in traffic_data if item.get("incident_detected", False))
        if incident_count > 0:
            primary_causes.append("Traffic incidents")
            contributing_factors.append(f"{incident_count} incidents detected")
        
        return {
            "primary_causes": primary_causes,
            "contributing_factors": contributing_factors
        }
    
    def _extract_incident_causes(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Extract causes from incident data"""
        
        description = incident.get("description", "").lower()
        
        primary_causes = []
        contributing_factors = []
        evidence = []
        
        # Analyze description for causes
        if "accident" in description or "crash" in description:
            primary_causes.append("Vehicle accident")
            evidence.append(f"Incident description: {incident.get('description', '')}")
        
        if "construction" in description:
            primary_causes.append("Road construction")
            evidence.append(f"Construction activity: {incident.get('description', '')}")
        
        if "breakdown" in description or "disabled" in description:
            primary_causes.append("Vehicle breakdown")
            evidence.append(f"Disabled vehicle: {incident.get('description', '')}")
        
        if "weather" in description:
            contributing_factors.append("Weather conditions")
            evidence.append(f"Weather impact: {incident.get('description', '')}")
        
        return {
            "primary_causes": primary_causes,
            "contributing_factors": contributing_factors,
            "evidence": evidence
        }
    
    def _analyze_weather_impact(self, weather_data: List[Dict[str, Any]]) -> List[str]:
        """Analyze weather impact on traffic"""
        
        weather_factors = []
        
        for weather in weather_data:
            condition = weather.get("condition", "").lower()
            impact = weather.get("traffic_impact", "low")
            
            if impact in ["high", "moderate"]:
                weather_factors.append(f"{condition.title()} weather causing {impact} impact")
        
        return weather_factors
    
    def _calculate_cause_confidence(self, cause_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in cause analysis"""
        
        evidence_count = len(cause_analysis.get("evidence_support", []))
        primary_causes_count = len(cause_analysis.get("primary_causes", []))
        
        # Base confidence on evidence and cause identification
        confidence = min(1.0, (evidence_count * 0.2) + (primary_causes_count * 0.1))
        
        return max(0.3, confidence)  # Minimum 30% confidence
    
    def _analyze_temporal_patterns(self, traffic_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in traffic data"""
        
        # Group by hour
        hourly_data = {}
        for item in traffic_data:
            try:
                timestamp = datetime.fromisoformat(item.get("timestamp", ""))
                hour = timestamp.hour
                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(item)
            except:
                continue
        
        # Calculate hourly averages
        hourly_averages = {}
        for hour, data in hourly_data.items():
            speeds = [item.get("speed", 0) for item in data]
            hourly_averages[hour] = {
                "avg_speed": sum(speeds) / len(speeds) if speeds else 0,
                "data_points": len(data)
            }
        
        return hourly_averages
    
    def _analyze_incident_patterns(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in incident data"""
        
        # Group by hour
        hourly_incidents = {}
        for incident in incidents:
            try:
                start_time = datetime.fromisoformat(incident.get("start_time", ""))
                hour = start_time.hour
                hourly_incidents[hour] = hourly_incidents.get(hour, 0) + 1
            except:
                continue
        
        return hourly_incidents
    
    def _create_research_summary(self, collected_data: Dict[str, Any], 
                               key_incidents: List[Dict[str, Any]]) -> str:
        """Create a summary of research findings"""
        
        data_sources = list(collected_data.keys())
        incident_count = len(key_incidents)
        
        summary = f"Research analysis completed with data from {len(data_sources)} sources. "
        summary += f"Identified {incident_count} key incidents requiring attention. "
        
        if incident_count > 0:
            high_impact_incidents = [i for i in key_incidents if i.get("impact_score", 0) > 0.7]
            summary += f"{len(high_impact_incidents)} incidents have high impact on traffic flow."
        
        return summary
    
    def _extract_key_findings(self, cause_analysis: Dict[str, Any], 
                            pattern_analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis"""
        
        findings = []
        
        # Add primary causes
        for cause in cause_analysis.get("primary_causes", []):
            findings.append(f"Primary cause identified: {cause}")
        
        # Add contributing factors
        for factor in cause_analysis.get("contributing_factors", []):
            findings.append(f"Contributing factor: {factor}")
        
        # Add pattern findings
        if pattern_analysis.get("patterns_detected"):
            findings.append("Recurring patterns detected in traffic data")
        
        return findings
    
    def _assess_data_quality(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of collected data"""
        
        quality_metrics = {
            "completeness": 0.0,
            "freshness": 0.0,
            "consistency": 0.0,
            "reliability": 0.0
        }
        
        # Calculate completeness
        expected_sources = ["traffic_data", "news_articles", "incidents"]
        available_sources = sum(1 for source in expected_sources if collected_data.get(source))
        quality_metrics["completeness"] = available_sources / len(expected_sources)
        
        # Calculate freshness (assume recent data)
        quality_metrics["freshness"] = 0.8
        
        # Calculate consistency
        traffic_data = collected_data.get("traffic_data", [])
        if traffic_data and len(traffic_data) > 1:
            quality_metrics["consistency"] = 0.7
        
        # Calculate reliability
        quality_metrics["reliability"] = (quality_metrics["completeness"] + 
                                        quality_metrics["freshness"] + 
                                        quality_metrics["consistency"]) / 3
        
        return quality_metrics
    
    def _calculate_research_confidence(self, cause_analysis: Dict[str, Any], 
                                     pattern_analysis: Dict[str, Any]) -> float:
        """Calculate overall research confidence"""
        
        cause_confidence = cause_analysis.get("cause_confidence", 0.5)
        pattern_confidence = 0.8 if pattern_analysis.get("patterns_detected") else 0.5
        
        return (cause_confidence + pattern_confidence) / 2
    
    def _generate_research_recommendations(self, cause_analysis: Dict[str, Any], 
                                         pattern_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on research findings"""
        
        recommendations = []
        
        # Recommendations based on causes
        primary_causes = cause_analysis.get("primary_causes", [])
        if "Traffic congestion" in primary_causes:
            recommendations.append("Implement traffic management strategies to reduce congestion")
        
        if "Vehicle accident" in primary_causes:
            recommendations.append("Enhance safety measures and incident response protocols")
        
        if "Road construction" in primary_causes:
            recommendations.append("Optimize construction scheduling and traffic management")
        
        # Recommendations based on patterns
        if pattern_analysis.get("patterns_detected"):
            recommendations.append("Investigate recurring patterns for long-term solutions")
        
        return recommendations
    
    def _identify_research_limitations(self, collected_data: Dict[str, Any]) -> List[str]:
        """Identify limitations in the research data"""
        
        limitations = []
        
        # Check data availability
        if not collected_data.get("traffic_data"):
            limitations.append("Limited traffic data availability")
        
        if not collected_data.get("news_articles"):
            limitations.append("Limited news coverage data")
        
        if not collected_data.get("incidents"):
            limitations.append("Limited incident data")
        
        # Check data quality
        traffic_data = collected_data.get("traffic_data", [])
        if len(traffic_data) < 10:
            limitations.append("Insufficient traffic data points for robust analysis")
        
        return limitations
    
    def _calculate_incident_impact(self, incident: Dict[str, Any]) -> float:
        """Calculate impact score for an incident"""
        
        impact_score = 0.5  # Base impact
        
        # Increase based on severity
        severity = incident.get("severity", "minor").lower()
        severity_impacts = {"minor": 0.2, "moderate": 0.4, "major": 0.6, "critical": 0.8}
        impact_score += severity_impacts.get(severity, 0.2)
        
        # Increase based on duration
        try:
            start_time = datetime.fromisoformat(incident.get("start_time", ""))
            end_time = datetime.fromisoformat(incident.get("end_time", ""))
            duration_hours = (end_time - start_time).total_seconds() / 3600
            impact_score += min(0.3, duration_hours * 0.1)
        except:
            pass
        
        return min(1.0, impact_score)
    
    def _calculate_data_quality_score(self, collected_data: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        
        quality_metrics = self._assess_data_quality(collected_data)
        return quality_metrics.get("reliability", 0.5)
