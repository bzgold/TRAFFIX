"""
Data models for Traffix system
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ReportMode(str, Enum):
    """Report generation modes"""
    QUICK = "quick"
    DEEP = "deep"
    DEEP_DATA = "deep_data"
    ANOMALY_INVESTIGATION = "anomaly_investigation"
    LEADERSHIP_SUMMARY = "leadership_summary"


class DataSource(str, Enum):
    """Data source types"""
    RITIS = "ritis"
    NEWS = "news"
    INCIDENT = "incident"
    WEATHER = "weather"
    SOCIAL = "social"


class TrafficData(BaseModel):
    """Traffic data from RITIS"""
    timestamp: datetime
    location: str
    speed: float
    volume: int
    occupancy: float
    congestion_level: str
    incident_detected: bool = False


class NewsArticle(BaseModel):
    """News article data"""
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    relevance_score: float = 0.0
    location_keywords: List[str] = []


class IncidentReport(BaseModel):
    """Traffic incident report"""
    incident_id: str
    description: str
    location: str
    severity: str
    start_time: datetime
    end_time: Optional[datetime] = None
    impact_radius: float = 0.0
    affected_roads: List[str] = []


class AnalysisResult(BaseModel):
    """Analysis result from analyzer agent"""
    anomaly_detected: bool
    confidence_score: float
    primary_causes: List[str]
    supporting_evidence: List[str]
    impact_assessment: str
    recommendations: List[str]
    # Enhanced fields for anomaly investigation
    baseline_comparison: Optional[Dict[str, Any]] = None
    temporal_patterns: Optional[Dict[str, Any]] = None
    incident_correlation: Optional[Dict[str, Any]] = None
    weather_impact: Optional[Dict[str, Any]] = None
    event_correlation: Optional[Dict[str, Any]] = None


class StoryElement(BaseModel):
    """Individual story element"""
    element_type: str  # "introduction", "cause", "impact", "resolution", "conclusion"
    content: str
    supporting_data: List[Dict[str, Any]]
    confidence: float


class Report(BaseModel):
    """Final report structure"""
    report_id: str
    mode: ReportMode
    title: str
    executive_summary: str
    story_elements: List[StoryElement]
    data_sources: List[DataSource]
    generated_at: datetime
    analysis_duration: float
    confidence_score: float
    recommendations: List[str]


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class AgentTask(BaseModel):
    """Agent task definition"""
    task_id: str
    agent_type: str
    status: AgentStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class UserQuestion(BaseModel):
    """User question types for targeted analysis"""
    question_type: str  # "anomaly_investigation", "incident_summary", "weather_impact", "leadership_summary"
    location: str
    time_period: str
    specific_question: str
    context: Optional[Dict[str, Any]] = None


class AnomalyInvestigation(BaseModel):
    """Specific anomaly investigation result"""
    anomaly_type: str  # "congestion_spike", "speed_drop", "volume_increase", "reliability_decrease"
    severity: str  # "low", "moderate", "high", "critical"
    baseline_period: str
    anomaly_period: str
    deviation_percentage: float
    root_causes: List[str]
    contributing_factors: List[str]
    impact_metrics: Dict[str, Any]
    recommended_actions: List[str]


class LeadershipSummary(BaseModel):
    """Executive summary for leadership consumption"""
    period: str
    key_highlights: List[str]
    performance_metrics: Dict[str, Any]
    major_incidents: List[str]
    weather_impacts: List[str]
    recommendations: List[str]
    next_period_outlook: str
    confidence_level: str


class IncidentAnalysis(BaseModel):
    """Detailed incident analysis"""
    incident_id: str
    incident_type: str
    severity: str
    duration: float  # hours
    affected_corridors: List[str]
    traffic_impact: Dict[str, Any]
    contributing_factors: List[str]
    response_effectiveness: str
    lessons_learned: List[str]
    prevention_recommendations: List[str]
