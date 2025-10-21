"""
Data integration services for various data sources
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from config import settings
from utils.error_handler import handle_errors, DataCollectionError
from data_processors.ritis_processor import RITISProcessor
from services.vector_service import VectorService


class RITISService:
    """Service for integrating with RITIS API"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffix.ritis")
        self.base_url = settings.ritis_base_url
        self.api_key = settings.ritis_api_key
        self.ritis_processor = RITISProcessor()
        self.vector_service = VectorService()
    
    @handle_errors(default_return=[], log_error=True)
    async def get_traffic_data(self, location: str, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get traffic data from RITIS vector database"""
        try:
            self.logger.info(f"Fetching RITIS traffic data from vector database for {location}")
            
            # Query vector database for RITIS traffic events
            ritis_events = await self.vector_service.search_similar_content(
                query=f"traffic flow speed volume congestion {location}",
                location=location,
                data_types=["ritis_event"],
                limit=30
            )
            
            # Convert to traffic data format
            traffic_data = []
            for event in ritis_events:
                if event.get("data_type") == "ritis_event":
                    # Extract traffic metrics from RITIS event
                    traffic_data.append({
                        "timestamp": event.get("timestamp", ""),
                        "location": event.get("location", location),
                        "speed": self._extract_speed_from_event(event),
                        "volume": self._extract_volume_from_event(event),
                        "occupancy": self._extract_occupancy_from_event(event),
                        "congestion_level": event.get("severity", "unknown"),
                        "incident_detected": "accident" in event.get("description", "").lower(),
                        "source": "ritis_vector_db",
                        "event_type": event.get("event_type", ""),
                        "highway": event.get("highway", ""),
                        "relevance_score": event.get("score", 0.0)
                    })
            
            self.logger.info(f"Retrieved {len(traffic_data)} RITIS traffic data points from vector database")
            return traffic_data
            
        except Exception as e:
            self.logger.error(f"RITIS traffic data fetch from vector database failed: {e}")
            # Fallback to mock data
            return await self._get_fallback_traffic_data(location, time_range_hours)
    
    def _extract_speed_from_event(self, payload: Dict[str, Any]) -> float:
        """Extract speed information from RITIS event payload"""
        # Try to extract speed from description or use default
        description = payload.get("description", "").lower()
        if "slow" in description or "congestion" in description:
            return 25.0
        elif "stopped" in description or "blocked" in description:
            return 0.0
        else:
            return 45.0  # Default speed
    
    def _extract_volume_from_event(self, payload: Dict[str, Any]) -> int:
        """Extract volume information from RITIS event payload"""
        # Estimate volume based on event type and severity
        severity = payload.get("severity", "low")
        if severity == "high":
            return 2000
        elif severity == "medium":
            return 1500
        else:
            return 1000
    
    def _extract_occupancy_from_event(self, payload: Dict[str, Any]) -> float:
        """Extract occupancy information from RITIS event payload"""
        # Estimate occupancy based on event type
        event_type = payload.get("event_type", "").lower()
        if "accident" in event_type or "breakdown" in event_type:
            return 0.8
        elif "construction" in event_type:
            return 0.6
        else:
            return 0.3
    
    async def _get_fallback_traffic_data(self, location: str, time_range_hours: int) -> List[Dict[str, Any]]:
        """Fallback mock traffic data if vector database fails"""
        self.logger.warning("Using fallback mock traffic data")
        traffic_data = []
        base_time = datetime.now() - timedelta(hours=time_range_hours)
        
        for i in range(0, time_range_hours, 2):  # Every 2 hours
            traffic_data.append({
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "location": location,
                "speed": 45.0 + (i % 20) - 10,  # Varying speeds
                "volume": 1200 + (i % 500) - 250,
                "occupancy": 0.3 + (i % 30) / 100,
                "congestion_level": self._determine_congestion_level(i),
                "incident_detected": i % 7 == 0,
                "source": "ritis_fallback"
            })
        
        return traffic_data
    
    def _determine_congestion_level(self, hour_offset: int) -> str:
        """Determine congestion level based on time patterns"""
        if 7 <= hour_offset % 24 <= 9 or 17 <= hour_offset % 24 <= 19:
            return "high"
        elif 6 <= hour_offset % 24 <= 10 or 16 <= hour_offset % 24 <= 20:
            return "moderate"
        else:
            return "light"
    
    async def get_incident_data(self, location: str, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get incident data from RITIS vector database with improved filtering"""
        try:
            self.logger.info(f"Fetching RITIS incident data from vector database for {location}")
            
            # Query vector database for RITIS events from the last 3 days
            # Use region-based filtering - search by region, not specific location
            ritis_events = await self.vector_service.search_similar_content(
                query=f"{location} region traffic incidents accidents",
                location=location,
                data_types=["ritis_event"],
                limit=100  # Get more events for comprehensive analysis
            )
            
            # Convert vector search results to incident format with proper RITIS structure
            # Filter by REGION field (not location) - location is just a detail
            incidents = []
            for event in ritis_events:
                # RITIS events are stored directly in the event structure, not in payload
                if event.get("data_type") == "ritis_event":
                    # Check if the region matches (case-insensitive)
                    event_region = event.get("region", "").lower()
                    search_region = location.lower()
                    
                    # Only include events from the matching region
                    if search_region in event_region or event_region in search_region:
                        # Map RITIS columns to incident format
                        incidents.append({
                            "incident_id": event.get("event_id", "unknown"),
                            "description": event.get("description", ""),
                            "location": event.get("location", location),
                            "region": event.get("region", ""),
                            "road": event.get("road", ""),
                            "direction": event.get("direction", ""),
                            "county": event.get("county", ""),
                            "state": event.get("state", ""),
                            "start_time": event.get("start_time", ""),
                            "end_time": event.get("closed_time", ""),
                            "duration": event.get("duration", ""),
                            "roadway_clearance_time": event.get("roadway_clearance_time", ""),
                            "event_type": event.get("standardized_type", ""),
                            "agency_specific_type": event.get("agency_specific_type", ""),
                            "agency_specific_sub_type": event.get("agency_specific_sub_type", ""),
                            "system": event.get("system", ""),
                            "agency": event.get("agency", ""),
                            "open_closed": event.get("open_closed", ""),
                            "edc_incident_type": event.get("edc_incident_type", ""),
                            "operator_notes": event.get("operator_notes", ""),
                            "latitude": event.get("latitude"),
                            "longitude": event.get("longitude"),
                            "source": "ritis_vector_db",
                            "relevance_score": event.get("score", 0.0)
                        })
            
            self.logger.info(f"Retrieved {len(incidents)} RITIS incidents from vector database")
            return incidents
            
        except Exception as e:
            self.logger.error(f"RITIS incident data fetch from vector database failed: {e}")
            # Fallback to mock data if vector database fails
            return await self._get_fallback_incident_data(location, time_range_hours)
    
    async def _get_fallback_incident_data(self, location: str, time_range_hours: int) -> List[Dict[str, Any]]:
        """Fallback mock incident data if vector database fails"""
        self.logger.warning("Using fallback mock incident data")
        incidents = []
        base_time = datetime.now() - timedelta(hours=time_range_hours)
        
        incident_types = [
            "Vehicle breakdown",
            "Accident",
            "Road construction", 
            "Weather related closure",
            "Emergency vehicle response"
        ]
        
        for i in range(2):  # 2 incidents
            incidents.append({
                "incident_id": f"RITIS_{i:04d}",
                "description": f"{incident_types[i % len(incident_types)]} on {location}",
                "location": f"{location} Highway, Mile {i+10}",
                "severity": "moderate" if i % 2 == 0 else "minor",
                "start_time": (base_time + timedelta(hours=i*6)).isoformat(),
                "end_time": (base_time + timedelta(hours=i*6+2)).isoformat(),
                "impact_radius": 2.0 + i,
                "affected_roads": [f"{location} Highway"],
                "source": "ritis_fallback"
            })
        
        return incidents


class NewsService:
    """Service for integrating with news APIs"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffix.news")
        self.api_key = settings.news_api_key
    
    async def get_traffic_news(self, location: str, time_range_hours: int = 24, 
                             max_articles: int = 10) -> List[Dict[str, Any]]:
        """Get traffic-related news articles"""
        try:
            self.logger.info(f"Fetching news data for {location}")
            
            # Mock news data - replace with actual news API integration
            articles = []
            base_time = datetime.now() - timedelta(hours=time_range_hours)
            
            news_sources = ["Local News", "Traffic Authority", "City Updates", "Transportation Dept"]
            news_titles = [
                f"Traffic congestion reported on {location} highway",
                f"Road construction causes delays in {location}",
                f"Accident on {location} freeway blocks traffic",
                f"Weather impacts traffic flow in {location}",
                f"New traffic management system for {location}",
                f"Rush hour delays on {location} corridor",
                f"Emergency road closure in {location}",
                f"Traffic signal malfunction affects {location}"
            ]
            
            for i in range(min(max_articles, len(news_titles))):
                articles.append({
                    "title": news_titles[i],
                    "content": self._generate_news_content(news_titles[i], location),
                    "url": f"https://example-news.com/article/{i}",
                    "published_at": (base_time + timedelta(hours=i*2)).isoformat(),
                    "source": news_sources[i % len(news_sources)],
                    "relevance_score": 0.8 + (i % 20) / 100,
                    "location_keywords": [location.lower(), "traffic", "highway", "road"],
                    "data_source": "news"
                })
            
            return articles
            
        except Exception as e:
            self.logger.error(f"News data fetch failed: {e}")
            return []
    
    def _generate_news_content(self, title: str, location: str) -> str:
        """Generate mock news content"""
        return f"""
        {title}
        
        This is a detailed news article about transportation issues affecting {location}. 
        The article discusses various factors contributing to traffic congestion and 
        provides insights into local transportation challenges.
        
        Key points covered:
        - Current traffic conditions
        - Impact on commuters
        - Planned improvements
        - Community response
        
        This content is generated for demonstration purposes and represents the type 
        of information that would be gathered from real news sources.
        """.strip()


class WeatherService:
    """Service for integrating with weather APIs"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffix.weather")
    
    async def get_weather_data(self, location: str, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get weather data that might affect traffic"""
        try:
            self.logger.info(f"Fetching weather data for {location}")
            
            # Mock weather data
            weather_data = []
            base_time = datetime.now() - timedelta(hours=time_range_hours)
            
            weather_conditions = ["clear", "rain", "fog", "snow", "cloudy"]
            
            for i in range(0, time_range_hours, 6):  # Every 6 hours
                condition = weather_conditions[i % len(weather_conditions)]
                weather_data.append({
                    "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                    "location": location,
                    "condition": condition,
                    "temperature": 20 + (i % 15) - 7,  # 13-35°C
                    "precipitation": 0.1 if condition in ["rain", "snow"] else 0.0,
                    "visibility": 10.0 if condition == "clear" else 5.0 if condition == "fog" else 8.0,
                    "wind_speed": 5 + (i % 10),
                    "traffic_impact": self._assess_traffic_impact(condition),
                    "source": "weather"
                })
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Weather data fetch failed: {e}")
            return []
    
    def _assess_traffic_impact(self, condition: str) -> str:
        """Assess traffic impact of weather condition"""
        if condition in ["rain", "snow", "fog"]:
            return "high"
        elif condition == "cloudy":
            return "moderate"
        else:
            return "low"


class SocialMediaService:
    """Service for integrating with social media APIs"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffix.social")
    
    async def get_traffic_posts(self, location: str, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get traffic-related social media posts"""
        try:
            self.logger.info(f"Fetching social media data for {location}")
            
            # Mock social media data
            posts = []
            base_time = datetime.now() - timedelta(hours=time_range_hours)
            
            platforms = ["Twitter", "Facebook", "Reddit", "Instagram"]
            post_types = ["complaint", "update", "warning", "question", "observation"]
            
            for i in range(5):  # 5 posts
                posts.append({
                    "post_id": f"social_{i:04d}",
                    "content": f"Traffic is terrible on {location} right now! #traffic #{location.lower()}",
                    "platform": platforms[i % len(platforms)],
                    "author": f"user_{i}",
                    "posted_at": (base_time + timedelta(hours=i*4)).isoformat(),
                    "post_type": post_types[i % len(post_types)],
                    "sentiment": "negative" if i % 2 == 0 else "neutral",
                    "location_keywords": [location.lower()],
                    "engagement": 10 + (i % 50),
                    "source": "social"
                })
            
            return posts
            
        except Exception as e:
            self.logger.error(f"Social media data fetch failed: {e}")
            return []


class DataIntegrationService:
    """Main service for coordinating data collection from all sources"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffix.data_integration")
        self.ritis = RITISService()
        self.news = NewsService()
        self.weather = WeatherService()
        self.social = SocialMediaService()
    
    async def collect_all_data(self, location: str, time_range_hours: int = 24, 
                             mode: str = "quick") -> Dict[str, Any]:
        """Collect data from all available sources with optimization"""
        self.logger.info(f"Starting data collection for {location} in {mode} mode")
        
        # For deep_optimized mode, use optimized collection
        if mode == "deep_optimized":
            return await self._collect_optimized_data(location, time_range_hours)
        
        # Determine max sources based on mode
        max_news_articles = 5 if mode == "quick" else 20
        max_social_posts = 3 if mode == "quick" else 10
        
        try:
            # Collect data from all sources in parallel
            tasks = [
                self.ritis.get_traffic_data(location, time_range_hours),
                self.ritis.get_incident_data(location, time_range_hours),
                self.news.get_traffic_news(location, time_range_hours, max_news_articles),
                self.weather.get_weather_data(location, time_range_hours),
                self.social.get_traffic_posts(location, time_range_hours)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            collected_data = {
                "traffic_data": results[0] if not isinstance(results[0], Exception) else [],
                "incidents": results[1] if not isinstance(results[1], Exception) else [],
                "news_articles": results[2] if not isinstance(results[2], Exception) else [],
                "weather_data": results[3] if not isinstance(results[3], Exception) else [],
                "social_posts": results[4] if not isinstance(results[4], Exception) else [],
                "collection_metadata": {
                    "location": location,
                    "time_range_hours": time_range_hours,
                    "mode": mode,
                    "collected_at": datetime.now().isoformat(),
                    "sources_used": self._get_successful_sources(results)
                }
            }
            
            self.logger.info(f"Data collection completed for {location}")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return {
                "traffic_data": [],
                "incidents": [],
                "news_articles": [],
                "weather_data": [],
                "social_posts": [],
                "collection_metadata": {
                    "location": location,
                    "time_range_hours": time_range_hours,
                    "mode": mode,
                    "collected_at": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
    
    async def _collect_optimized_data(self, location: str, time_range_hours: int) -> Dict[str, Any]:
        """Collect data with optimization for deep mode - focus on accidents and summaries"""
        
        try:
            self.logger.info(f"Collecting optimized data for {location}")
            
            # Collect only essential data sources
            tasks = [
                self.ritis.get_incident_data(location, time_range_hours),  # Focus on incidents
                self.news.get_traffic_news(location, time_range_hours, 10),  # Limited news
                self.weather.get_weather_data(location, time_range_hours)  # Weather for context
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results with optimization
            incidents = results[0] if not isinstance(results[0], Exception) else []
            news_articles = results[1] if not isinstance(results[1], Exception) else []
            weather_data = results[2] if not isinstance(results[2], Exception) else []
            
            # Optimize incidents - prioritize accidents
            optimized_incidents = self._optimize_incidents(incidents)
            
            # Create summary of all events
            event_summary = self._create_event_summary(incidents)
            
            return {
                "traffic_data": [],  # Skip detailed traffic data for speed
                "incidents": optimized_incidents,
                "news_articles": news_articles,
                "weather_data": weather_data,
                "social_posts": [],  # Skip social media for speed
                "event_summary": event_summary,
                "collection_metadata": {
                    "location": location,
                    "time_range_hours": time_range_hours,
                    "mode": "deep_optimized",
                    "collected_at": datetime.now().isoformat(),
                    "sources_used": ["ritis_incidents", "news", "weather"],
                    "optimization_applied": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Optimized data collection failed: {e}")
            return {
                "traffic_data": [],
                "incidents": [],
                "news_articles": [],
                "weather_data": [],
                "social_posts": [],
                "event_summary": "Data collection failed",
                "collection_metadata": {
                    "location": location,
                    "time_range_hours": time_range_hours,
                    "mode": "deep_optimized",
                    "collected_at": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
    
    def _optimize_incidents(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize incidents by prioritizing accidents and creating summaries"""
        
        if not incidents:
            return []
        
        # Separate accidents from other incidents
        accidents = []
        other_incidents = []
        
        for incident in incidents:
            description = incident.get("description", "").upper()
            if any(word in description for word in ["ACCIDENT", "CRASH", "COLLISION"]):
                incident["priority"] = "high"
                incident["detailed_analysis"] = True
                accidents.append(incident)
            else:
                incident["priority"] = "medium"
                incident["detailed_analysis"] = False
                other_incidents.append(incident)
        
        # Create summary incident
        summary_incident = {
            "event_id": "summary",
            "description": f"Event Summary: {self._create_incident_summary(incidents)}",
            "priority": "summary",
            "detailed_analysis": False,
            "timestamp": datetime.now().isoformat(),
            "source": "summary"
        }
        
        # Return optimized list: summary + accidents + limited other incidents
        return [summary_incident] + accidents[:5] + other_incidents[:3]
    
    def _create_incident_summary(self, incidents: List[Dict[str, Any]]) -> str:
        """Create a summary of all incidents"""
        
        if not incidents:
            return "No incidents reported"
        
        # Count by type
        accident_count = sum(1 for inc in incidents if "ACCIDENT" in inc.get("description", "").upper())
        construction_count = sum(1 for inc in incidents if "CONSTRUCTION" in inc.get("description", "").upper())
        other_count = len(incidents) - accident_count - construction_count
        
        return f"Total incidents: {len(incidents)} (Accidents: {accident_count}, Construction: {construction_count}, Other: {other_count})"
    
    def _create_event_summary(self, incidents: List[Dict[str, Any]]) -> str:
        """Create a comprehensive event summary"""
        
        if not incidents:
            return "No events to summarize"
        
        # Group by severity and type
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        type_counts = {}
        
        for incident in incidents:
            severity = incident.get("severity", "low")
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            event_type = self._categorize_incident_type(incident.get("description", ""))
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Create summary text
        summary_parts = [f"Total events: {len(incidents)}"]
        
        if any(severity_counts.values()):
            summary_parts.append(f"Severity: High({severity_counts['high']}) Medium({severity_counts['medium']}) Low({severity_counts['low']})")
        
        if type_counts:
            type_summary = ", ".join([f"{k}({v})" for k, v in type_counts.items()])
            summary_parts.append(f"Types: {type_summary}")
        
        return "; ".join(summary_parts)
    
    def _categorize_incident_type(self, description: str) -> str:
        """Categorize incident by description"""
        desc_upper = description.upper()
        
        if any(word in desc_upper for word in ["ACCIDENT", "CRASH", "COLLISION"]):
            return "Accidents"
        elif any(word in desc_upper for word in ["CONSTRUCTION", "ROADWORK"]):
            return "Construction"
        elif any(word in desc_upper for word in ["CONGESTION", "SLOW"]):
            return "Congestion"
        else:
            return "Other"
    
    def _get_successful_sources(self, results: List[Any]) -> List[str]:
        """Get list of sources that successfully returned data"""
        sources = []
        source_names = ["ritis_traffic", "ritis_incidents", "news", "weather", "social"]
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception) and result:
                sources.append(source_names[i])
        
        return sources
