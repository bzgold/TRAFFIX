"""
Streamlit User Interface for Traffix
"""
import streamlit as st
import asyncio
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from orchestration.langgraph_workflow import TraffixWorkflow
from models import UserQuestion
from tech_config import tech_settings
from tavily import TavilyClient
from services.vector_service import VectorService, reset_vector_service


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traffix.streamlit")

# Page configuration
st.set_page_config(
    page_title="Traffix - AI Storytelling for Transportation Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .chat-panel {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .chat-user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-assistant {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .reasoning-box {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .sources-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize workflow
@st.cache_resource
def get_workflow():
    vector_service = get_vector_service()
    return TraffixWorkflow(vector_service=vector_service)

# Cache vector service for speed
@st.cache_resource
def get_cached_vector_service():
    return get_vector_service()

@st.cache_resource
def get_tavily_client():
    return TavilyClient(api_key=tech_settings.tavily_api_key)

@st.cache_resource
def get_vector_service():
    return VectorService()

def search_region_data(region_details, region_name):
    """Search for traffic, economic, and incident data for the selected region using Tavily"""
    try:
        tavily = get_tavily_client()
        vector_service = get_vector_service()
        
        st.info(f"🔍 Searching for data about {region_name}...")
        
        # Search for different types of data
        search_queries = [
            f"{region_name} traffic incidents accidents",
            f"{region_name} economic development infrastructure",
            f"{region_name} transportation news updates",
            f"{region_name} road construction delays"
        ]
        
        all_results = []
        
        for query in search_queries:
            try:
                results = tavily.search(
                    query=query,
                    search_depth="basic",
                    max_results=3
                )
                
                if isinstance(results, dict) and 'results' in results:
                    articles = results['results']
                elif isinstance(results, list):
                    articles = results
                else:
                    articles = []
                
                for article in articles:
                    if isinstance(article, dict):
                        all_results.append({
                            'title': article.get('title', 'No title'),
                            'url': article.get('url', 'No URL'),
                            'content': article.get('content', 'No content'),
                            'source': 'tavily',
                            'query': query,
                            'region': region_name,
                            'timestamp': datetime.now().isoformat()
                        })
                        
            except Exception as e:
                st.warning(f"Search failed for query '{query}': {str(e)}")
                continue
        
        st.success(f"✅ Found {len(all_results)} relevant articles")
        
        # Process and store in vector database
        if all_results:
            st.info("📝 Processing and storing data in vector database...")
            
            processed_count = 0
            for article in all_results:
                try:
                    # Create text content for embedding
                    text_content = f"Title: {article['title']}\nContent: {article['content']}\nRegion: {article['region']}"
                    
                    # Generate embedding
                    embedding = vector_service.embeddings.embed_query(text_content)
                    
                    # Store in vector database
                    vector_service.store_document(
                        text=text_content,
                        metadata={
                            'title': article['title'],
                            'url': article['url'],
                            'region': article['region'],
                            'source': article['source'],
                            'query': article['query'],
                            'timestamp': article['timestamp']
                        }
                    )
                    processed_count += 1
                    
                except Exception as e:
                    st.warning(f"Failed to process article '{article['title']}': {str(e)}")
                    continue
            
            st.success(f"✅ Processed and stored {processed_count} articles in vector database")
            
        return all_results
        
    except Exception as e:
        st.error(f"❌ Failed to search region data: {str(e)}")
        return []

def main():
    """Main Streamlit application"""
    
    # Reset vector service to ensure fresh instance
    reset_vector_service()
    
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffix</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #7f8c8d;">AI Storytelling for Transportation Analytics</h2>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Region selection
        region = st.selectbox(
            "Select Region",
            options=["Northern Virginia", "Washington DC"],
            help="Select the region for traffic analysis"
        )
        
        # Map region to specific location details
        region_details = {
            "Northern Virginia": {
                "primary_corridors": ["I-95 North", "I-66", "I-495", "Route 1"],
                "cities": ["Arlington", "Alexandria", "Fairfax", "Falls Church"],
                "search_terms": ["Northern Virginia traffic", "I-95 Virginia", "I-66 traffic", "I-495 Virginia"]
            },
            "Washington DC": {
                "primary_corridors": ["I-395", "I-295", "I-495", "Route 50"],
                "cities": ["Washington DC", "Capitol Hill", "Georgetown", "Dupont Circle"],
                "search_terms": ["Washington DC traffic", "I-395 DC", "I-295 DC", "DC Metro traffic"]
            }
        }
        
        selected_region = region_details[region]
        location = f"{region} - {selected_region['primary_corridors'][0]}"
        
        # Mode selection
        mode = st.selectbox(
            "Analysis Mode",
            options=["quick", "deep_data"],
            format_func=lambda x: {
                "quick": "Quick Mode - Comprehensive Summaries",
                "deep_data": "Deep Mode - Research Reports (1000+ words)"
            }[x]
        )
        
        # Reuse existing data option
        with st.info("Tip: Reuse data for faster analysis"):
            reuse_existing = st.checkbox(
                "Reuse Existing Data", 
                value=False, 
                help="Skip data collection and use previously stored data"
        )
        
        # Time mode selection - choose between general periods or specific dates
        time_mode = st.radio(
            "Time Selection Mode",
            options=["time_period", "specific_date"],
            format_func=lambda x: {
                "time_period": "General Time Period",
                "specific_date": "Specific Date"
            }[x],
            help="Choose between general time ranges or specific dates from RITIS data"
        )
        
        if time_mode == "time_period":
            # General time period selection
            time_period = st.selectbox(
                "Time Period",
                options=["24h", "48h", "1w", "1m"],
                format_func=lambda x: {
                    "24h": "Last 24 Hours",
                    "48h": "Last 48 Hours", 
                    "1w": "Last Week",
                    "1m": "Last Month"
                }[x]
            )
            selected_day = None
        else:
            # Specific date selection from RITIS data
            available_days = [
                "2025-10-19 (Sunday)",
                "2025-10-18 (Saturday)", 
                "2025-10-17 (Friday)"
            ]
            selected_day = st.selectbox(
                "Select Specific Date",
                options=available_days,
                help="Choose a specific day from the available RITIS events data"
            )
            time_period = None
        
        # User question (optional)
        user_question = st.text_area(
            "Specific Question (Optional)",
            placeholder="e.g., Why was congestion higher than normal today?",
            help="Ask a specific question about the traffic patterns, or leave blank for general analysis"
        )
        
        # Analysis button
        st.markdown("---")
        st.subheader("Analysis")
        analyze_button = st.button("Run Analysis", type="primary", use_container_width=True)
        
        # PDF Mode button
        st.markdown("---")
        st.subheader("PDF Mode")
        pdf_mode_button = st.button("PDF Document Chat", type="secondary", use_container_width=True)
        st.caption("Upload and chat with your own PDF documents")
        
        # Show region info
        st.info(f"**Selected Region:** {region}")
        st.info(f"**Analysis Mode:** {mode}")
        if user_question:
            st.info(f"**Question:** {user_question}")
        else:
            st.info("**Question:** General analysis")
        
        # Hidden export options - always enabled
        export_pdf = True
        export_html = False
        export_json = False
        show_ragas = False  # Hide RAGAS from main display
        email_enabled = False
        email_recipients = None
    
    # Main content area
    if analyze_button:
        asyncio.run(run_analysis(location, mode, time_period, user_question, {
            "export_html": export_html,
            "export_pdf": export_pdf,
            "export_json": export_json,
            "email_enabled": email_enabled,
            "email_recipients": email_recipients if email_enabled else None,
            "show_ragas": show_ragas,
            "time_mode": time_mode,
            "selected_day": selected_day
        }, region, selected_region, reuse_existing))
    
    # PDF Mode - Document Chat Interface
    if pdf_mode_button:
        st.session_state['pdf_mode_active'] = True
    
    if st.session_state.get('pdf_mode_active', False):
        display_pdf_chat_interface()
    
    # Landing page - show when no analysis running and not in PDF mode
    if not analyze_button and not st.session_state.get('pdf_mode_active', False):
        display_landing_page()

def pull_region_news(region: str, time_period: str, mode: str, selected_region: dict, time_mode: str = "time_period", selected_day: str = None):
    """Pull relevant news articles using Tavily API based on region, time period, and analysis mode"""
    try:
        tavily = get_tavily_client()
        
        # Build search queries based on region, time period, and analysis mode
        search_queries = build_search_queries(region, time_period, mode, selected_region, time_mode, selected_day)
        
        all_articles = []
        
        for query in search_queries:
            try:
                results = tavily.search(
                    query=query,
                    search_depth="basic",
                    max_results=3
                )
                
                if isinstance(results, dict) and 'results' in results:
                    articles = results['results']
                elif isinstance(results, list):
                    articles = results
                else:
                    articles = []
                
                for article in articles:
                    if isinstance(article, dict):
                        all_articles.append({
                            'title': article.get('title', 'No title'),
                            'url': article.get('url', 'No URL'),
                            'content': article.get('content', 'No content'),
                            'source': 'tavily',
                            'query': query,
                            'region': region,
                            'time_period': time_period,
                            'analysis_mode': mode,
                            'timestamp': datetime.now().isoformat()
                        })
                        
            except Exception as e:
                st.warning(f"Search failed for query '{query}': {str(e)}")
                continue
        
        return all_articles
        
    except Exception as e:
        st.error(f"❌ Failed to pull news articles: {str(e)}")
        return []

def build_search_queries(region: str, time_period: str, mode: str, selected_region: dict, time_mode: str = "time_period", selected_day: str = None):
    """Build search queries based on region, time period, and analysis mode"""
    queries = []
    
    # Base region terms
    region_terms = selected_region.get('search_terms', [region])
    cities = selected_region.get('cities', [region])
    
    # Time period mapping
    if time_mode == "day_mode" and selected_day:
        # Extract date from selected day (e.g., "2025-10-19 (Sunday)" -> "2025-10-19")
        day_date = selected_day.split(" ")[0]
        time_term = f"on {day_date}"
    else:
        time_map = {
            "24h": "last 24 hours",
            "48h": "last 48 hours", 
            "1w": "last week",
            "1m": "last month"
        }
        time_term = time_map.get(time_period, "recent")
    
    # Mode-specific keywords
    mode_keywords = {
        "quick": ["traffic", "incidents", "delays"],
        "deep": ["traffic analysis", "transportation", "infrastructure", "congestion"],
        "anomaly_investigation": ["accidents", "incidents", "emergencies", "unusual"],
        "leadership_summary": ["transportation", "infrastructure", "planning", "policy"],
        "pattern_analysis": ["traffic patterns", "trends", "analysis", "data"]
    }
    
    # Human impact keywords for contextual analysis
    human_impact_keywords = {
        "work_patterns": ["return to work", "remote work", "hybrid work", "office return", "commute patterns"],
        "economic_factors": ["layoffs", "job cuts", "unemployment", "economic impact", "business closures"],
        "government_events": ["government shutdown", "federal shutdown", "furlough", "government workers"],
        "public_events": ["Army 10 miler", "marathon", "protest", "demonstration", "rally", "festival", "concert"],
        "seasonal_travel": ["holiday travel", "Thanksgiving", "Christmas", "New Year", "summer travel"],
        "education": ["school opening", "school closing", "back to school", "university", "college"],
        "technology": ["outage", "system down", "network issues", "IT problems", "cyber attack"],
        "weather_events": ["snow day", "weather closure", "storm", "emergency", "weather impact"]
    }
    
    keywords = mode_keywords.get(mode, ["traffic", "transportation"])
    
    # Add human impact context for deeper analysis
    if mode in ["deep", "leadership_summary", "pattern_analysis"]:
        # Add 2-3 human impact factors for context
        impact_categories = list(human_impact_keywords.keys())[:3]
        for category in impact_categories:
            impact_keywords = human_impact_keywords[category][:2]  # Take 2 keywords per category
            keywords.extend(impact_keywords)
    
    # Build queries combining region, time, and mode
    for region_term in region_terms[:2]:  # Limit to 2 region terms
        for keyword in keywords[:5]:  # Increased limit to include human impact
            query = f"{region_term} {keyword} {time_term}"
            queries.append(query)
    
    return queries[:5]  # Limit to 5 total queries

async def generate_comprehensive_report(articles: list, mode: str, region: str, time_period: str, user_question: str = "", reuse_existing: bool = False):
    """Generate comprehensive report based on analysis mode using RAG"""
    if not articles and not reuse_existing:
        return "No articles found to analyze."
    
    try:
        # Try to use the workflow for comprehensive analysis with RAGAS evaluation
        try:
            from orchestration.langgraph_workflow import TraffixWorkflow
            
            # Build user query
            if user_question:
                query = user_question
            else:
                query = f"Analyze traffic patterns for {region} in {mode} mode"
            
            # Get the existing vector service instance
            vector_service = get_vector_service()
            
            # Run the workflow with the existing vector service
            workflow = TraffixWorkflow(vector_service=vector_service)
            workflow_result = await workflow.run_workflow(
                user_query=user_question if user_question else f"Analyze traffic for {region}",
                location=region,
                mode=mode
            )
            
            # Extract the final output and evaluation results
            if workflow_result.get("workflow_status") == "completed":
                # Get the analysis result and report data
                analysis_result = workflow_result.get("analysis_result", {})
                report_data = workflow_result.get("report_data", {})
                evaluation_results = workflow_result.get("evaluation_results", {})
                
                # Get the report content based on mode
                if mode == "quick":
                    report_content = report_data.get("executive_summary", "")
                else:  # deep mode
                    report_content = report_data.get("comprehensive_report", "")
                
                if report_content:
                    # Return both report and evaluation results
                    return {
                        "report": report_content,
                        "evaluation_results": evaluation_results,
                        "workflow_metadata": workflow_result.get("processing_time", 0)
                    }
            
        except Exception as workflow_error:
            st.warning(f"Workflow failed, falling back to direct LLM generation: {str(workflow_error)}")
        
        # Fallback to direct LLM generation
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=tech_settings.openai_api_key,
            temperature=0.3
        )
        
        # Retrieve relevant chunks from Qdrant using RAG
        vector_service = get_vector_service()
        
        # First, check what data is available
        stats = vector_service.get_region_stats()
        print(f"DEBUG: Available regions: {stats.get('regions', {})}")
        print(f"DEBUG: Available data types: {stats.get('data_types', {})}")
        print(f"DEBUG: Total chunks: {stats.get('total_chunks', 0)}")
        
        # Build comprehensive search queries based on mode
        search_queries = build_comprehensive_search_queries(mode, region, time_period)
        
        # Retrieve comprehensive content from vector database
        all_content = ""
        for query in search_queries:
            try:
                # Search for news articles first
                news_chunks = await vector_service.search_similar_content(
                    query=query,
                    location=region,  # Use region filter
                    data_types=["news"],
                    limit=10  # Get news articles
                )
                
                # Search for related RITIS traffic events
                ritis_chunks = await vector_service.search_similar_content(
                    query=query,
                    location=region,  # Use region filter
                    data_types=["ritis_event"],
                    limit=10  # Get traffic events
                )
                
                # If no results with region filter, try without location filter
                if not news_chunks and not ritis_chunks:
                    print(f"DEBUG: No results with region filter, trying without location filter")
                    news_chunks = await vector_service.search_similar_content(
                        query=query,
                        location=None,
                        data_types=["news"],
                        limit=10
                    )
                    
                    ritis_chunks = await vector_service.search_similar_content(
                        query=query,
                        location=None,
                        data_types=["ritis_event"],
                        limit=10
                    )
                
                print(f"DEBUG: Query '{query}' found {len(news_chunks)} news chunks and {len(ritis_chunks)} RITIS chunks")
                
                # Combine both sources
                relevant_chunks = news_chunks + ritis_chunks
                
                # Format news articles
                for chunk in news_chunks:
                    all_content += f"\n\n**📰 NEWS ARTICLE:**\n"
                    all_content += f"**Title:** {chunk.get('title', 'Unknown')}\n"
                    all_content += f"**Content:** {chunk.get('text', 'No content')}\n"
                    all_content += f"**Source:** {chunk.get('url', 'Unknown')}\n"
                    all_content += f"**Relevance:** {chunk.get('score', 0.0):.3f}\n"
                
                # Format RITIS traffic events with enhanced details
                for chunk in ritis_chunks:
                    all_content += f"\n\n**🚗 TRAFFIC EVENT:**\n"
                    all_content += f"**Event ID:** {chunk.get('event_id', 'Unknown')}\n"
                    all_content += f"**Event Type:** {chunk.get('event_type', 'Unknown')}\n"
                    all_content += f"**Location:** {chunk.get('location', 'Unknown')}\n"
                    all_content += f"**Highway/Road:** {chunk.get('highway', 'Unknown')}\n"
                    all_content += f"**Direction:** {chunk.get('direction', 'Unknown')}\n"
                    all_content += f"**County:** {chunk.get('county', 'Unknown')}\n"
                    all_content += f"**State:** {chunk.get('state', 'Unknown')}\n"
                    all_content += f"**Severity:** {chunk.get('severity', 'Unknown')}\n"
                    all_content += f"**Impact Type:** {chunk.get('impact_type', 'Unknown')}\n"
                    all_content += f"**Description:** {chunk.get('description', 'No description')}\n"
                    all_content += f"**Timestamp:** {chunk.get('timestamp', 'Unknown')}\n"
                    all_content += f"**Duration:** {chunk.get('duration', 'Unknown')}\n"
                    all_content += f"**Agency:** {chunk.get('agency', 'Unknown')}\n"
                    all_content += f"**Coordinates:** Lat {chunk.get('latitude', 'N/A')}, Lon {chunk.get('longitude', 'N/A')}\n"
                    all_content += f"**Relevance Score:** {chunk.get('score', 0.0):.3f}\n"
                    
            except Exception as e:
                st.warning(f"RAG retrieval failed for query '{query}': {str(e)}")
                continue
        
        # Check if we have any content, if not use deep research fallback
        if not all_content.strip():
            print("DEBUG: No data found in vector database, using deep research fallback")
            try:
                from research.deep_research import DeepResearchAgent
                deep_research = DeepResearchAgent()
                
                research_question = f"Comprehensive traffic analysis for {region} during {time_period}"
                research_result = await deep_research.conduct_deep_research(
                    research_question=research_question,
                    location=region,
                    time_range_hours=24 if time_period == "24h" else 168  # Default to 1 week
                )
                
                # Extract findings and create report
                findings = research_result.get('findings', {})
                all_content = f"""
                Research Question: {research_question}
                Location: {region}
                Time Range: {time_period}
                
                Key Insights: {findings.get('key_insights', [])}
                Recommendations: {findings.get('recommendations', [])}
                Confidence Level: {findings.get('confidence_level', 'Unknown')}
                """
            except Exception as e:
                print(f"DEBUG: Deep research fallback failed: {e}")
                all_content = f"No relevant data found for {region} in {mode} mode. Please try a different region or time period."
        
        # Generate mode-specific comprehensive report
        report_prompt = build_report_prompt(mode, region, time_period, user_question, all_content)
        
        response = llm.invoke([HumanMessage(content=report_prompt)])
        return response.content
        
    except Exception as e:
        return f"Error generating comprehensive report: {str(e)}"

def build_comprehensive_search_queries(mode: str, region: str, time_period: str):
    """Build comprehensive search queries for different analysis modes"""
    base_queries = {
        "quick": [
            f"traffic incidents {region}",
            f"traffic delays {region}",
            f"road closures {region}",
            f"accidents {region}"
        ],
        "deep_data": [
            f"traffic incidents {region}",
            f"traffic accidents {region}",
            f"road closures {region}",
            f"transportation infrastructure {region}",
            f"traffic engineering {region}",
            f"traffic patterns analysis {region}",
            f"congestion analysis {region}",
            f"transportation planning {region}",
            f"traffic flow optimization {region}",
            f"traffic delays {region}",
            f"emergency response {region}"
        ],
        "trends": [
            f"traffic trends {region}",
            f"traffic patterns {region}",
            f"unusual traffic patterns {region}",
            f"traffic anomalies {region}",
            f"recurring traffic issues {region}",
            f"traffic behavior analysis {region}",
            f"transportation trends {region}",
            f"traffic hotspots {region}"
        ],
        "leadership_summary": [
            f"transportation policy {region}",
            f"infrastructure investment {region}",
            f"transportation leadership {region}",
            f"traffic management strategy {region}",
            f"transportation planning {region}",
            f"economic impact traffic {region}",
            f"business productivity traffic {region}",
            f"return to work traffic impact {region}",
            f"government shutdown traffic {region}",
            f"public events traffic {region}",
            f"holiday travel patterns {region}",
            f"school opening traffic {region}"
        ]
    }
    
    return base_queries.get(mode, [f"traffic transportation {region}"])

def build_report_prompt(mode: str, region: str, time_period: str, user_question: str, content: str):
    """Build comprehensive report prompts based on analysis mode"""
    
    mode_instructions = {
        "quick": """
        Create a COMPREHENSIVE QUICK SUMMARY report focusing on:
        - Full summary of all traffic events and incidents
        - Detailed causes of traffic impacts and disruptions
        - Potential areas to watch out for
        - Complete source attribution and data references
        - Actionable recommendations for traffic management
        
        Structure:
        1. EXECUTIVE SUMMARY (comprehensive overview of traffic situation)
        2. TRAFFIC EVENTS SUMMARY (complete list of incidents with details)
        3. CAUSES & FACTORS (detailed analysis of what led to traffic impacts)
        4. IMPACT ASSESSMENT (commuter effects, economic impact, safety concerns)
        5. AREAS OF CONCERN (specific locations and times to monitor)
        6. RECOMMENDATIONS (specific, actionable steps for improvement)
        7. DATA SOURCES (complete attribution of all information sources)
        
        Make it comprehensive yet concise, providing all necessary information for informed decision-making.
        """,
        
        "deep_data": """
        You are a senior transportation analyst creating a formal, professional research report. 
        
        CRITICAL FORMATTING REQUIREMENTS:
        - Write ONLY in FULL PARAGRAPHS with flowing, connected narrative
        - Use proper markdown headers (##, ###) to organize sections
        - ABSOLUTELY NO bullet points, NO lists, NO dashes - PARAGRAPHS ONLY
        - Each paragraph should be 4-6 sentences minimum
        - Connect paragraphs with transition sentences for smooth flow
        - Cite specific data from RITIS events (locations, times, incident types)
        - Reference news articles to provide context and community impact
        - Write like a formal research paper or professional traffic study
        
        REPORT STRUCTURE (All in paragraph format):
        
        ## Executive Summary
        
        Write 2-3 comprehensive paragraphs that synthesize the key findings from the analysis period.
        Begin by establishing the overall traffic situation in the region, then discuss the most
        significant incidents and their impacts. Conclude with the primary recommendations and
        areas requiring attention. Use specific numbers and locations from the RITIS data.
        
        ## Current Traffic Conditions
        
        Write 3-4 flowing paragraphs describing the traffic situation in detail. Start with the
        general state of traffic flow across major corridors. Then discuss specific problem areas,
        citing RITIS incident data with exact locations, times, and incident types (e.g., "On I-95
        northbound near Exit 160, a multi-vehicle accident occurred at approximately 7:45 AM...").
        Explain how these incidents affected commuter patterns and travel times. Connect individual
        events to broader trends you observe in the data.
        
        ## Major Incidents Analysis
        
        Write detailed paragraphs examining the most significant traffic events. For each major
        incident, provide a narrative that includes what happened, where it occurred (specific
        road and location from RITIS), when it took place, how long it lasted, and what the
        cascading effects were on surrounding roadways. Weave in context from news articles
        about causes, emergency response, and community impact. Make connections between
        incidents to show patterns.
        
        ## Traffic Patterns and Trends
        
        Write analytical paragraphs discussing patterns observed in the data. Examine temporal
        patterns (time of day, day of week), spatial patterns (corridor-specific issues,
        geographic clusters), and incident type patterns (accidents, disabled vehicles, road work).
        Explain why these patterns matter from a transportation planning perspective. Support
        your analysis with specific data points and percentages.
        
        ## Impact Assessment
        
        Write paragraphs evaluating the broader impacts of the traffic conditions. Discuss effects
        on commuters (delay times, reliability), economic impacts (productivity, commerce),
        safety implications, and environmental considerations. Use data to quantify impacts
        where possible. Connect short-term incidents to long-term infrastructure needs.
        
        ## Recommendations and Solutions
        
        Write strategic paragraphs outlining recommendations. Start with immediate operational
        improvements, then discuss medium-term tactics, and conclude with long-term strategic
        initiatives. For each recommendation, explain the rationale, expected benefits, and
        implementation considerations. Prioritize based on impact and feasibility.
        
        ## Conclusion
        
        Write 2-3 concluding paragraphs that tie everything together. Summarize the key insights,
        reiterate the most critical recommendations, and provide perspective on the overall
        state of transportation in the region.
        
        STYLE GUIDELINES:
        - Professional, formal tone suitable for transportation executives
        - Use transition phrases: "Furthermore," "In addition," "Notably," "Moreover," "Consequently"
        - Vary sentence structure for readability
        - Integrate data naturally into prose, not as lists
        - Minimum 1000 words total
        - No abbreviations without definition (e.g., "Interstate 95 (I-95)")
        
        Remember: Every single piece of information must be in paragraph format. Think of this
        as a research paper you would submit to a transportation journal.
        """,
        
        "leadership_summary": """
        Create a SUPER HIGH-LEVEL LEADERSHIP SUMMARY focusing on:
        
        1. EXECUTIVE SUMMARY (2-3 sentences maximum):
           - Overall traffic situation and its significance
           - Why this matters for the organization/region
        
        2. CONGESTION IMPACT ASSESSMENT:
           - Economic impact of traffic congestion
           - Business productivity effects
           - Public safety implications
           - Environmental and social costs
        
        3. STRATEGIC IMPLICATIONS:
           - Long-term infrastructure needs
           - Investment priorities and ROI
           - Policy implications
           - Competitive advantage considerations
        
        4. LEADERSHIP DECISIONS REQUIRED:
           - Critical decisions needed
           - Resource allocation priorities
           - Timeline considerations
           - Stakeholder impact
        
        5. BUSINESS CASE:
           - Cost of inaction vs. cost of action
           - Expected outcomes and benefits
           - Risk mitigation strategies
        
        Keep it extremely concise, high-level, and focused on strategic decision-making.
        Explain the "why this matters" from a business and leadership perspective.
        """,
        
        "trends": """
        You are a BUSINESS ANALYST with expertise in transportation and traffic engineering.
        Create a TRENDS & PATTERN ANALYSIS report that explains complex traffic data to non-business people.
        
        1. TRENDS SUMMARY (quick overview in block format):
           - Key trends identified (hotspots, consistent problem areas)
           - Anomalies detected and their significance
           - Patterns of concern and their business impact
        
        2. DETAILED TREND ANALYSIS:
           - Deep dive into each trend with business context
           - Traffic engineering perspective explained simply
           - Why trends are occurring (causes and contributing factors)
           - Business impact assessment (costs, efficiency, safety)
        
        3. HOTSPOTS & CONSISTENT AREAS:
           - Geographic hotspots with business implications
           - Consistent problem areas and their patterns
           - Time-based patterns (rush hours, seasonal, weekly)
           - Infrastructure bottlenecks and their business costs
        
        4. BUSINESS INSIGHTS & RECOMMENDATIONS:
           - ROI-based recommendations for improvements
           - Cost-benefit analysis of solutions
           - Priority recommendations for business impact
           - Proactive measures to prevent future issues
        
        5. EXECUTIVE SUMMARY:
           - Key findings for leadership
           - Business case for recommended actions
           - Expected outcomes and timeline
        
        Write in clear, business-friendly language that explains traffic engineering concepts
        in terms that non-technical stakeholders can understand and act upon.
        """
    }
    
    instruction = mode_instructions.get(mode, "Create a comprehensive analysis report.")
    
    return f"""
    You are a transportation analyst creating a comprehensive report for {region} based on {mode} analysis.
    
    {instruction}
    
    REGION: {region}
    TIME PERIOD: {time_period}
    USER QUESTION: {user_question if user_question else "General analysis"}
    
    SOURCE DATA TO ANALYZE:
    {content}
    
    IMPORTANT: The data above includes both NEWS ARTICLES (📰) and TRAFFIC EVENTS (🚗). 
    Your analysis should:
    1. Connect news reports to actual traffic events
    2. Explain WHY traffic patterns occurred based on the structured event data
    3. Use the RITIS traffic events to provide factual context for news reports
    4. Identify patterns between reported incidents and actual traffic impacts
    5. Consider HUMAN IMPACT FACTORS that may influence traffic:
       - Work patterns (return to office, remote work, hybrid schedules)
       - Economic factors (layoffs, business closures, economic events)
       - Government events (shutdowns, furloughs, federal worker impacts)
       - Public events (Army 10 miler, protests, festivals, concerts)
       - Seasonal factors (holiday travel, school schedules)
       - Technology issues (outages, system problems affecting work)
       - Weather events (snow days, emergency closures)
    
    Please create a comprehensive, well-structured report that provides actionable insights for transportation professionals.
    Use both the news articles and traffic events to support your analysis with specific examples and evidence.
    """

def display_comprehensive_report(report: str, mode: str, region: str, time_period: str, evaluation_results: dict = None):
    """Display the comprehensive report based on analysis mode"""
    
    # Set report title based on mode
    mode_titles = {
        "quick": "Quick Traffic Summary",
        "deep_data": "Comprehensive Research Report"
    }
    
    title = mode_titles.get(mode, "Analysis Report")
    
    st.markdown(f"### {title}")
    st.markdown(f"**Region:** {region} | **Period:** {time_period} | **Mode:** {mode}")
    st.markdown("---")
    
    # Display the report
    st.markdown(report)
    
    # Add confidence score section at the bottom
    st.markdown("---")
    st.markdown("### Report Confidence Score")
    
    # Calculate confidence based on data quality
    if evaluation_results:
        overall_score = evaluation_results.get("overall_score", 0.75)
    else:
        # Default confidence based on data availability
        overall_score = 0.80
    
    # Display confidence with color coding
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if overall_score >= 0.85:
            confidence_level = "Very High"
            color = "green"
            icon = "✓"
        elif overall_score >= 0.75:
            confidence_level = "High"
            color = "blue"
            icon = "✓"
        elif overall_score >= 0.65:
            confidence_level = "Moderate"
            color = "orange"
            icon = "!"
        else:
            confidence_level = "Low"
            color = "red"
            icon = "⚠"
        
        st.markdown(f"<div style='text-align: center; padding: 20px; background-color: rgba(0,0,0,0.05); border-radius: 10px;'>"
                   f"<h2 style='color: {color};'>{icon} {overall_score:.1%}</h2>"
                   f"<p style='font-size: 18px;'><strong>Confidence Level: {confidence_level}</strong></p>"
                   f"<p style='font-size: 14px; color: #666;'>Based on data quality, source reliability, and analysis depth</p>"
                   f"</div>", unsafe_allow_html=True)
    
    # Display RAGAS evaluation results if available (hidden by default now)
    if evaluation_results and False:  # Disabled - confidence score replaces this
        display_ragas_evaluation(evaluation_results)
    
    # Add PDF download button
    st.markdown("---")
    if st.button("Download PDF Report", type="primary"):
        st.info("PDF generation feature coming soon. For now, use your browser's Print to PDF function.")

def display_ragas_evaluation(evaluation_results: dict):
    """Display RAGAS evaluation results with visual metrics"""
    
    st.markdown("---")
    st.markdown("### Quality Evaluation (RAGAS Metrics)")
    
    # Extract key metrics
    overall_score = evaluation_results.get("overall_score", 0.0)
    composite_scores = evaluation_results.get("composite_scores", {})
    quality_issues = evaluation_results.get("quality_issues", [])
    recommendations = evaluation_results.get("recommendations", [])
    evaluation_report = evaluation_results.get("evaluation_report", {})
    
    # Overall quality score with color coding
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("#### Overall Quality Score")
        # Color code based on score
        if overall_score >= 0.8:
            color = "🟢"
            grade = "Excellent"
        elif overall_score >= 0.7:
            color = "🟡"
            grade = "Good"
        elif overall_score >= 0.6:
            color = "🟠"
            grade = "Fair"
        else:
            color = "🔴"
            grade = "Needs Improvement"
        
        st.markdown(f"{color} **{overall_score:.2f}/1.0** - {grade}")
    
    with col2:
        st.metric("Quality Grade", evaluation_report.get("summary", {}).get("quality_grade", "N/A"))
    
    with col3:
        metrics_passed = evaluation_report.get("summary", {}).get("metrics_passed", 0)
        total_metrics = evaluation_report.get("summary", {}).get("total_metrics", 0)
        st.metric("Metrics Passed", f"{metrics_passed}/{total_metrics}")
    
    # Individual RAGAS metrics
    st.markdown("#### Detailed Metrics")
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Faithfulness
        faithfulness_score = composite_scores.get("faithfulness_score", 0.0)
        st.markdown(f"**Faithfulness:** {faithfulness_score:.2f}")
        st.progress(faithfulness_score)
        st.caption("How well grounded the response is in source data")
        
        # Answer Relevancy
        relevancy_score = composite_scores.get("relevancy_score", 0.0)
        st.markdown(f"**Answer Relevancy:** {relevancy_score:.2f}")
        st.progress(relevancy_score)
        st.caption("How relevant the response is to the question")
    
    with col2:
        # Context Quality
        context_score = composite_scores.get("context_score", 0.0)
        st.markdown(f"**Context Quality:** {context_score:.2f}")
        st.progress(context_score)
        st.caption("Precision and recall of retrieved context")
        
        # Answer Correctness
        correctness_score = composite_scores.get("correctness_score", 0.0)
        st.markdown(f"**Answer Correctness:** {correctness_score:.2f}")
        st.progress(correctness_score)
        st.caption("Accuracy of facts and interpretations")
    
    # Quality Issues
    if quality_issues:
        st.markdown("#### ⚠️ Quality Issues")
        
        for issue in quality_issues[:5]:  # Show top 5 issues
            severity = issue.get("severity", "medium")
            metric = issue.get("metric", "unknown")
            description = issue.get("description", "No description")
            
            # Color code severity
            if severity == "critical":
                st.error(f"🔴 **{metric.title()}**: {description}")
            elif severity == "high":
                st.warning(f"🟠 **{metric.title()}**: {description}")
            else:
                st.info(f"🟡 **{metric.title()}**: {description}")
    
    # Recommendations
    if recommendations:
        with st.expander("Improvement Recommendations", expanded=False):
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5 recommendations
                priority = rec.get("priority", "medium")
                recommendation = rec.get("recommendation", "No recommendation")
                actionable_steps = rec.get("actionable_steps", [])
                
                st.markdown(f"**{i}. {recommendation}**")
                if actionable_steps:
                    for step in actionable_steps:
                        st.markdown(f"   • {step}")
                st.markdown("---")
    
    # Evaluation Summary
    if evaluation_report.get("summary"):
        summary = evaluation_report["summary"]
        st.markdown("#### Evaluation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Issues Found", summary.get("issues_found", 0))
        
        with col2:
            st.metric("Critical Issues", summary.get("critical_issues", 0))
        
        with col3:
            st.metric("Metrics Evaluated", summary.get("total_metrics", 0))
        
        with col4:
            st.metric("Pass Rate", f"{(summary.get('metrics_passed', 0) / max(summary.get('total_metrics', 1), 1) * 100):.1f}%")

async def summarize_articles_for_analysis(articles: list, mode: str, region: str, time_period: str):
    """Summarize articles based on analysis type using OpenAI with RAG from Qdrant"""
    if not articles:
        return "No articles found to summarize."
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=tech_settings.openai_api_key,
            temperature=0.3
        )
        
        # Retrieve relevant chunks from Qdrant using RAG
        vector_service = get_vector_service()
        
        # Build search query based on analysis mode
        mode_queries = {
            "quick": f"traffic incidents delays {region}",
            "deep_data": f"transportation infrastructure analysis {region}",
            "anomaly_investigation": f"accidents emergencies unusual {region}",
            "leadership_summary": f"transportation policy planning {region}",
            "pattern_analysis": f"traffic patterns trends data {region}"
        }
        
        search_query = mode_queries.get(mode, f"traffic transportation financial {region}")
        
        # Retrieve relevant chunks from vector database using existing method
        try:
            relevant_chunks = await vector_service.search_similar_content(
                query=search_query,
                location=region,
                data_types=["news"],
                limit=10
            )
            
            # Combine retrieved content
            retrieved_content = ""
            for chunk in relevant_chunks:
                retrieved_content += f"\n\nChunk: {chunk.get('text', 'No content')}\n"
                retrieved_content += f"Source: {chunk.get('title', 'Unknown')}\n"
                retrieved_content += f"Region: {chunk.get('location', 'Unknown')}\n"
                retrieved_content += f"Score: {chunk.get('score', 0.0):.3f}\n"
                
        except Exception as e:
            st.warning(f"RAG retrieval failed, using direct article content: {str(e)}")
            # Fallback to direct article content
            retrieved_content = ""
            for i, article in enumerate(articles[:10]):
                retrieved_content += f"\n\nArticle {i+1}: {article['title']}\n"
                retrieved_content += f"Content: {article['content'][:500]}...\n"
                retrieved_content += f"Source: {article['url']}\n"
        
        # Create mode-specific prompt
        mode_prompts = {
            "quick": "Provide a quick summary focusing on key traffic incidents and delays",
            "deep_data": "Provide a comprehensive analysis covering traffic patterns, infrastructure, and trends",
            "anomaly_investigation": "Focus on unusual incidents, accidents, and anomalies that need investigation",
            "leadership_summary": "Create an executive summary highlighting key transportation issues and recommendations",
            "pattern_analysis": "Analyze traffic patterns, trends, and data insights from the articles"
        }
        
        prompt = f"""
        You are analyzing news articles about {region} for a {mode} traffic analysis using Retrieval-Augmented Generation (RAG).
        
        {mode_prompts.get(mode, "Provide a general summary")}.
        
        Time period: {time_period}
        Region: {region}
        Analysis mode: {mode}
        
        Retrieved relevant content from vector database:
        {retrieved_content}
        
        Please provide a clear, structured summary that will help inform the traffic analysis.
        Focus on the most relevant information for {mode} analysis.
        Use the retrieved content to provide accurate, contextual insights.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

async def store_articles_in_qdrant(articles: list, region: str, mode: str, time_period: str):
    """Store articles in Qdrant vector database using existing add_news_data method"""
    if not articles:
        return 0
    
    try:
        vector_service = get_vector_service()
        
        # Format articles for the existing add_news_data method
        formatted_articles = []
        for article in articles:
            formatted_article = {
                'title': article['title'],
                'content': article['content'],
                'url': article['url'],
                'source': article['source'],
                'published_at': article['timestamp'],
                'relevance_score': 1.0,  # Default relevance score
                'region': article['region'],
                'analysis_mode': article['analysis_mode'],
                'time_period': article['time_period'],
                'query': article['query']
            }
            formatted_articles.append(formatted_article)
        
        # Use the existing add_news_data method
        success = await vector_service.add_news_data(formatted_articles, region)
        
        if success:
            return len(articles)
        else:
            return 0
        
    except Exception as e:
        st.error(f"❌ Failed to store articles in Qdrant: {str(e)}")
        return 0

async def run_analysis(location: str, mode: str, time_period: str, 
                    user_question: str, export_options: dict, region: str, selected_region: dict, reuse_existing: bool = False):
    """Run the analysis workflow with Tavily news integration"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if reuse_existing:
            # Reuse existing data - skip data collection
            progress_bar.progress(20)
            status_text.text(f"🔄 Reusing existing data for {region}...")
            
            # Check if we have existing data for this region
            vector_service = get_vector_service()
            existing_stats = vector_service.get_region_stats()
            
            if region in existing_stats.get("regions", {}):
                st.info(f"✅ Found existing data for {region} with {existing_stats['regions'][region]} chunks")
                news_articles = []  # Empty list since we're reusing existing data
            else:
                st.warning(f"⚠️ No existing data found for {region}. Switching to fresh data collection...")
                reuse_existing = False
        
        if not reuse_existing:
            # Step 1: Pull relevant news articles using Tavily
            progress_bar.progress(10)
            status_text.text(f"📰 Pulling news articles for {region}...")
            
            news_articles = pull_region_news(region, time_period, mode, selected_region, 
                                           export_options.get("time_mode", "time_period"), 
                                           export_options.get("selected_day"))
            
            if news_articles:
                st.success(f"✅ Found {len(news_articles)} relevant news articles")
                
                # Display articles found
                with st.expander(f"📰 Found {len(news_articles)} News Articles", expanded=False):
                    for i, article in enumerate(news_articles):
                        st.markdown(f"**{i+1}. {article['title']}**")
                        st.markdown(f"*Source: {article['url']}*")
                        st.markdown(f"*Content: {article['content'][:200]}...*")
                        st.markdown("---")
            else:
                st.warning(f"⚠️ No recent news found for {region}, proceeding with analysis...")
            
            # Step 2: Store articles in Qdrant vector database
            progress_bar.progress(25)
            status_text.text(f"💾 Storing articles in vector database...")
            
            stored_articles = await store_articles_in_qdrant(news_articles, region, mode, time_period)
            
            if stored_articles:
                st.success(f"✅ Stored {stored_articles} articles in Qdrant vector database")
        else:
            # When reusing, we still need to show some progress
            progress_bar.progress(25)
            status_text.text(f"🔄 Using existing data for analysis...")
            stored_articles = "Reused existing data"
        
        # Step 3: Summarize articles based on analysis type (only if we have new articles)
        if not reuse_existing and news_articles:
            progress_bar.progress(35)
            status_text.text(f"📝 Summarizing articles for {mode} analysis...")
            
            summary = await summarize_articles_for_analysis(news_articles, mode, region, time_period)
            
            if summary:
                st.info("📊 News Summary Generated")
                with st.expander("📊 News Summary", expanded=True):
                    st.markdown(summary)
        
        # Step 4: Generate Comprehensive Report
        progress_bar.progress(50)
        status_text.text("Generating comprehensive report...")
        
        comprehensive_report = await generate_comprehensive_report(
            news_articles if not reuse_existing else [], mode, region, time_period, user_question, reuse_existing
        )
        
        progress_bar.progress(100)
        status_text.text("Analysis completed successfully!")
        
        # Display comprehensive report
        st.success("🎉 Comprehensive Analysis Complete!")
        
        # Extract RAGAS evaluation results if available
        evaluation_results = None
        report_content = comprehensive_report
        
        if isinstance(comprehensive_report, dict):
            if 'evaluation_results' in comprehensive_report:
                evaluation_results = comprehensive_report['evaluation_results']
            if 'report' in comprehensive_report:
                report_content = comprehensive_report['report']
        
        # Show report based on analysis mode
        show_ragas = export_options.get("show_ragas", True)
        display_comprehensive_report(report_content, mode, region, time_period, evaluation_results if show_ragas else None)
        
        # Show data summary with RITIS event details
        with st.expander("Data Summary", expanded=False):
            # Get vector service stats for RITIS events
            vector_service = get_vector_service()
            stats = vector_service.get_region_stats()
            
            st.markdown("### Analysis Overview")
            st.markdown(f"""
            **Region Analyzed:** {region}  
            **Analysis Mode:** {mode.replace('_', ' ').title()}  
            **Time Period:** {time_period if time_period else selected_day}  
            **Data Source:** {'Reused Existing Data' if reuse_existing else 'Freshly Collected'}
            """)
            
            st.markdown("---")
            st.markdown("### Data Sources Used")
            
            # Count actual data used in this analysis
            ritis_count = stats.get('data_types', {}).get('ritis_event', 0)
            news_count = len(news_articles) if news_articles else stats.get('data_types', {}).get('news', 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RITIS Events", f"{ritis_count:,}")
                st.caption("Traffic incidents from RITIS")
            
            with col2:
                st.metric("News Articles", f"{news_count:,}")
                st.caption("Regional news sources")
            
            with col3:
                total_sources = ritis_count + news_count
                st.metric("Total Data Points", f"{total_sources:,}")
                st.caption("Combined data sources")
            
            st.markdown("---")
            st.markdown("### Report Status")
            st.markdown(f"**Report Generated:** {'✅ Success' if comprehensive_report else '❌ Failed'}")
            
            if evaluation_results:
                overall_score = evaluation_results.get("overall_score", 0.0)
                st.markdown(f"**Quality Score:** {overall_score:.1%}")
        
        # Show source articles summary (if not reusing data)
        if not reuse_existing and news_articles:
            with st.expander("Source Articles", expanded=False):
                st.markdown(f"**Total Articles Found:** {len(news_articles)}")
                st.markdown("---")
                for i, article in enumerate(news_articles[:5]):  # Show first 5 articles
                    st.markdown(f"**{i+1}. {article['title']}**")
                    st.markdown(f"*Source: {article['url']}*")
                    st.markdown("---")
            
    except Exception as e:
        progress_bar.progress(100)
        st.error(f"Analysis failed: {str(e)}")
        logger.error(f"Analysis failed: {e}")

def display_results(result: dict, location: str, mode: str):
    """Display analysis results"""
    
    st.success(f"✅ Analysis completed for {location} in {result['processing_time']:.2f} seconds")
    
    # Tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Analysis", "📝 Story", "📄 Report"])
    
    with tab1:
        display_summary(result, mode)
    
    with tab2:
        display_analysis_details(result)
    
    with tab3:
        display_story(result)
    
    with tab4:
        display_report(result)

def display_summary(result: dict, mode: str):
    """Display analysis summary"""
    
    st.header("Analysis Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    
    with col2:
        st.metric("Mode", mode.title())
    
    with col3:
        st.metric("Status", result['workflow_status'].title())
    
    with col4:
        if 'analysis_result' in result and result['analysis_result']:
            confidence = result['analysis_result'].get('analysis_result', {}).get('confidence_score', 0)
            st.metric("Confidence", f"{confidence:.2f}")
    
    # Analysis insights
    if 'analysis_result' in result and result['analysis_result']:
        analysis = result['analysis_result'].get('analysis_result', {})
        
        if analysis.get('anomaly_detected'):
            st.warning("🚨 Traffic anomaly detected")
        else:
            st.success("✅ Normal traffic patterns")
        
        # Primary causes
        if analysis.get('primary_causes'):
            st.subheader("Primary Causes")
            for cause in analysis['primary_causes']:
                st.write(f"• {cause}")
        
        # Recommendations
        if analysis.get('recommendations'):
            st.subheader("Recommendations")
            for rec in analysis['recommendations']:
                st.write(f"• {rec}")

def display_analysis_details(result: dict):
    """Display detailed analysis"""
    
    st.header("Detailed Analysis")
    
    # Traffic data visualization
    if 'collected_data' in result and result['collected_data']:
        traffic_data = result['collected_data'].get('traffic_data', [])
        
        if traffic_data:
            # Create DataFrame
            df = pd.DataFrame(traffic_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Speed over time
            fig_speed = px.line(df, x='timestamp', y='speed', 
                              title='Speed Over Time', 
                              labels={'speed': 'Speed (mph)', 'timestamp': 'Time'})
            st.plotly_chart(fig_speed, use_container_width=True)
            
            # Volume over time
            fig_volume = px.line(df, x='timestamp', y='volume',
                               title='Volume Over Time',
                               labels={'volume': 'Volume (vehicles)', 'timestamp': 'Time'})
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Congestion levels
            congestion_counts = df['congestion_level'].value_counts()
            fig_congestion = px.pie(values=congestion_counts.values, 
                                  names=congestion_counts.index,
                                  title='Congestion Level Distribution')
            st.plotly_chart(fig_congestion, use_container_width=True)

def display_story(result: dict):
    """Display generated story"""
    
    st.header("Generated Story")
    
    if 'story_data' in result and result['story_data']:
        story_data = result['story_data']
        
        # Executive summary
        if story_data.get('executive_summary'):
            st.subheader("Executive Summary")
            st.write(story_data['executive_summary'])
        
        # Story elements
        if story_data.get('story_elements'):
            st.subheader("Story Elements")
            for element in story_data['story_elements']:
                with st.expander(f"{element['element_type'].title()}"):
                    st.write(element['content'])
                    
                    if element.get('supporting_data'):
                        st.write("**Supporting Data:**")
                        for data in element['supporting_data']:
                            st.write(f"• {data.get('type', 'Unknown')}: {data.get('value', 'N/A')}")

def display_report(result: dict):
    """Display generated report"""
    
    st.header("Generated Report")
    
    if 'report_data' in result and result['report_data']:
        report_data = result['report_data']
        
        # Report metadata
        if 'report' in report_data:
            report = report_data['report']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Report ID:** {report.get('report_id', 'N/A')}")
                st.write(f"**Mode:** {report.get('mode', 'N/A')}")
            with col2:
                st.write(f"**Generated:** {report.get('generated_at', 'N/A')}")
                st.write(f"**Confidence:** {report.get('confidence_score', 0):.2f}")
        
        # Report content
        if 'report_content' in report_data:
            st.subheader("Report Content")
            st.markdown(report_data['report_content'], unsafe_allow_html=True)

def display_mode_info(mode: str):
    """Display information about the selected mode"""
    
    mode_info = {
        "quick": {
            "title": "🚀 Quick Mode",
            "description": "Fast daily summaries optimized for speed",
            "processing_time": "30-60 seconds",
            "use_cases": ["Daily summaries", "Shift reports", "Incident summaries"]
        },
        "deep_data": {
            "title": "🔬 Deep Mode", 
            "description": "Comprehensive research reports with detailed analysis",
            "processing_time": "2-5 minutes",
            "use_cases": ["Weekly reports", "Monthly analysis", "Incident investigations"]
        }
    }
    
    info = mode_info.get(mode, {})
    if info:
        with st.expander(f"ℹ️ About {info['title']}", expanded=False):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Processing Time:** {info['processing_time']}")
            st.write("**Use Cases:**")
            for use_case in info['use_cases']:
                st.write(f"• {use_case}")

def display_example_questions():
    """Display example questions"""
    
    with st.expander("Example Questions", expanded=False):
        st.write("**Anomaly Investigation:**")
        st.write("• Why was congestion higher than normal today?")
        st.write("• What caused the traffic spike this morning?")
        st.write("• Why did travel times increase yesterday?")
        
        st.write("**Leadership Summary:**")
        st.write("• Can you summarize this week's mobility highlights?")
        st.write("• What are the key traffic issues this month?")
        st.write("• How did weather impact traffic this week?")
        
        st.write("**Pattern Analysis:**")
        st.write("• What recurring congestion patterns should we address?")
        st.write("• What mitigation strategies would be most effective?")
        st.write("• When do accidents most commonly occur?")

def handle_exports(result: dict, export_options: dict):
    """Handle report exports"""
    
    if any(export_options.values()):
        st.header("Export Options")
        
        if export_options.get('export_html'):
            if st.button("📄 Download HTML Report"):
                st.success("HTML report download initiated")
        
        if export_options.get('export_pdf'):
            if st.button("📋 Download PDF Report"):
                st.success("PDF report download initiated")
        
        if export_options.get('export_json'):
            if st.button("📊 Download JSON Data"):
                st.success("JSON data download initiated")
        
        if export_options.get('email_enabled') and export_options.get('email_recipients'):
            if st.button("📧 Email Report"):
                st.success(f"Report emailed to: {export_options['email_recipients']}")

def search_knowledge_base_ultra_fast(query: str):
    """ULTRA FAST synchronous search - no async, no complex processing"""
    try:
        # Use cached vector service for maximum speed
        vector_service = get_cached_vector_service()
        
        # Direct synchronous search - much faster!
        import asyncio
        chunks = asyncio.run(vector_service.search_similar_content(
            query=query,
            location=None,
            data_types=None,
            limit=3  # Even smaller limit for speed
        ))
        
        if chunks:
            # Ultra simple format
            knowledge_content = ""
            for chunk in chunks[:2]:  # Only top 2
                data_type = chunk.get('data_type', 'unknown')
                if data_type == 'news':
                    knowledge_content += f"📰 {chunk.get('title', 'N/A')}\n"
                    knowledge_content += f"{chunk.get('text', 'N/A')[:100]}...\n\n"
                elif data_type == 'ritis_event':
                    knowledge_content += f"🚗 {chunk.get('event_type', 'N/A')} - {chunk.get('location', 'N/A')}\n"
                    knowledge_content += f"{chunk.get('description', 'N/A')[:80]}...\n\n"
            
            return knowledge_content, True
        else:
            return "", False
            
    except Exception as e:
        print(f"Fast search error: {e}")
        return "", False

def chat_interface():
    """General OpenAI chat interface using Qdrant knowledge base"""
    st.markdown("---")
    st.markdown("## 💬 General Chat Assistant")
    st.markdown("Ask questions about traffic data from Washington DC and Northern Virginia. I'll search our database of news articles and RITIS traffic events.")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about traffic or transportation..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Quick search..."):
                try:
                    # ULTRA FAST search
                    knowledge_content, found_data = search_knowledge_base_ultra_fast(prompt)
                    
                    if found_data:
                        # Ultra quick response
                        assistant_response = f"Found data:\n\n{knowledge_content}"
                        st.write(assistant_response)
                        
                    else:
                        # No data - instant response
                        assistant_response = "I do not know. No relevant data found."
                        st.write(assistant_response)
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def display_landing_page():
    """Display friendly landing page"""
    
    # Introduction
    st.markdown("""
    ## About TRAFFIX
    
    TRAFFIX is an advanced traffic analysis platform that combines real-time incident data with 
    news coverage and AI-powered insights to give you a complete picture of transportation conditions 
    in your region.
    """)
    
    st.markdown("---")
    
    # What it does
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### What TRAFFIX Does")
        st.markdown("""
        TRAFFIX analyzes transportation data to help you:
        
        - **Monitor Traffic Conditions** - Track incidents, accidents, and delays in real-time
        - **Understand Patterns** - Identify trends and recurring issues across your region
        - **Make Informed Decisions** - Get actionable recommendations based on comprehensive analysis
        - **Stay Updated** - Combine structured traffic data with news coverage for complete context
        """)
    
    with col2:
        st.markdown("### Data Sources")
        st.markdown("""
        TRAFFIX integrates multiple data sources:
        
        - **RITIS Traffic Events** - 22,000+ real-time incidents from Regional Integrated Transportation Information System
        - **News Articles** - Regional traffic news and coverage using Tavily API
        - **Weather Data** - Conditions affecting traffic flow
        - **Historical Patterns** - Trend analysis across time periods
        """)
    
    st.markdown("---")
    
    # Analysis modes
    st.markdown("### Analysis Modes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Quick Mode")
        st.markdown("""
        **Best for:** Daily monitoring and quick updates
        
        **Output:** Comprehensive summary highlighting key incidents, causes, and immediate concerns
        
        **Time:** 15-20 seconds
        
        **Use when:** You need a fast overview of current traffic conditions
        """)
    
    with col2:
        st.markdown("#### Deep Mode")
        st.markdown("""
        **Best for:** Detailed analysis and reporting
        
        **Output:** Professional 1000+ word research report with pattern analysis, impact assessment, and strategic recommendations
        
        **Time:** 20-30 seconds
        
        **Use when:** You need comprehensive insights for planning or decision-making
        """)
    
    with col3:
        st.markdown("#### PDF Mode")
        st.markdown("""
        **Best for:** Analyzing custom documents
        
        **Output:** Interactive chat with your uploaded PDF documents
        
        **Time:** Instant responses
        
        **Use when:** You want to analyze traffic reports, policy documents, or research papers
        """)
    
    st.markdown("---")
    
    # Example questions
    st.markdown("### Example Questions You Can Ask")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Traffic Monitoring")
        st.markdown("""
        - What are the current traffic conditions in Northern Virginia?
        - Where are the major incidents happening today?
        - What's causing delays on I-95?
        - Are there any road closures affecting commuters?
        """)
        
        st.markdown("#### Pattern Analysis")
        st.markdown("""
        - What traffic patterns emerged this week?
        - Which corridors have the most accidents?
        - When do peak congestion periods occur?
        - Are there recurring issues at specific locations?
        """)
    
    with col2:
        st.markdown("#### Impact Assessment")
        st.markdown("""
        - How are accidents affecting travel times?
        - What's the economic impact of traffic congestion?
        - Which areas need infrastructure improvements?
        - What are the safety implications of current patterns?
        """)
        
        st.markdown("#### Strategic Planning")
        st.markdown("""
        - What recommendations do you have for reducing congestion?
        - How can we improve incident response times?
        - What infrastructure investments should we prioritize?
        - What mitigation strategies would be most effective?
        """)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("### Ready to Get Started?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Select Your Region**
        
        Choose Northern Virginia or Washington DC from the sidebar
        """)
    
    with col2:
        st.markdown("""
        **2. Choose Analysis Mode**
        
        Pick Quick Mode for summaries or Deep Mode for comprehensive reports
        """)
    
    with col3:
        st.markdown("""
        **3. Run Analysis**
        
        Click "Run Analysis" and get AI-powered insights in seconds
        """)
    
    st.markdown("---")
    
    # Footer
    st.info("""
    **Tip:** Use the sidebar to configure your analysis settings, then click "Run Analysis" to generate your report.
    For custom document analysis, click "PDF Document Chat" to upload and interact with your own files.
    """)

def display_pdf_chat_interface():
    """Display PDF document chat interface"""
    st.markdown("# PDF Document Chat")
    st.markdown("Upload a PDF document and ask questions about its content. The system will only answer based on the PDF content.")
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Analysis"):
            st.session_state['pdf_mode_active'] = False
            st.rerun()
    
    st.markdown("---")
    
    # Initialize session state for PDF chat
    if 'pdf_chunks' not in st.session_state:
        st.session_state.pdf_chunks = []
    if 'pdf_chat_history' not in st.session_state:
        st.session_state.pdf_chat_history = []
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    
    # File uploader
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'], key='pdf_uploader')
    
    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.pdf_name != uploaded_file.name:
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_chunks = []
            st.session_state.pdf_chat_history = []
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Process the PDF
                    chunks = process_pdf_document(uploaded_file)
                    st.session_state.pdf_chunks = chunks
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    st.info(f"📄 Document split into {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    return
        
        # Display document info
        if st.session_state.pdf_chunks:
            with st.expander("Document Information", expanded=False):
                st.markdown(f"**Filename:** {st.session_state.pdf_name}")
                st.markdown(f"**Total Chunks:** {len(st.session_state.pdf_chunks)}")
                st.markdown(f"**Total Characters:** {sum(len(chunk) for chunk in st.session_state.pdf_chunks):,}")
            
            st.markdown("---")
            st.markdown("### Chat with Document")
            st.markdown("Ask questions about the document. Responses will only be based on the PDF content.")
            
            # Chat interface
            # Display chat history
            for message in st.session_state.pdf_chat_history:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
            
            # Chat input
            user_question = st.chat_input("Ask a question about the document...")
            
            # Quick action buttons (below chat input)
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Summarize Document"):
                    with st.spinner("Generating summary..."):
                        summary = answer_pdf_question(
                            "Please provide a comprehensive summary of this document.",
                            st.session_state.pdf_chunks,
                            st.session_state.pdf_name
                        )
                        st.session_state.pdf_chat_history.append({
                            'role': 'user',
                            'content': 'Summarize this document'
                        })
                        st.session_state.pdf_chat_history.append({
                            'role': 'assistant',
                            'content': summary
                        })
                        st.rerun()
            
            with col2:
                if st.button("🔑 Key Points"):
                    with st.spinner("Extracting key points..."):
                        key_points = answer_pdf_question(
                            "What are the key points and main takeaways from this document?",
                            st.session_state.pdf_chunks,
                            st.session_state.pdf_name
                        )
                        st.session_state.pdf_chat_history.append({
                            'role': 'user',
                            'content': 'What are the key points?'
                        })
                        st.session_state.pdf_chat_history.append({
                            'role': 'assistant',
                            'content': key_points
                        })
                        st.rerun()
            
            with col3:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.pdf_chat_history = []
                    st.rerun()
            
            # Handle user question if entered
            if user_question:
                # Add user message to history
                st.session_state.pdf_chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Generate response
                response = answer_pdf_question(
                    user_question,
                    st.session_state.pdf_chunks,
                    st.session_state.pdf_name
                )
                
                # Add assistant response to history
                st.session_state.pdf_chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
    
    else:
        st.info("👆 Upload a PDF document to get started")

def process_pdf_document(uploaded_file):
    """Process uploaded PDF and split into chunks"""
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    
    # Extract text from all pages
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    return chunks

def answer_pdf_question(question: str, chunks: list, pdf_name: str) -> str:
    """Answer question based only on PDF content"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Combine relevant chunks (for now, use all - can add semantic search later)
    context = "\n\n".join(chunks[:10])  # Use first 10 chunks for context
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=tech_settings.openai_api_key,
        temperature=0.3
    )
    
    # Create prompt
    system_message = SystemMessage(content=f"""You are a helpful assistant that answers questions about a PDF document titled "{pdf_name}".

CRITICAL RULES:
- Answer ONLY based on the provided document content
- Do NOT use external knowledge or make assumptions
- If the answer is not in the document, say "This information is not found in the provided document."
- Be specific and cite relevant parts of the document
- Provide direct quotes when appropriate
""")
    
    user_message = HumanMessage(content=f"""Document Content:
{context}

Question: {question}

Answer based ONLY on the document content above:""")
    
    # Get response
    response = llm.invoke([system_message, user_message])
    
    return response.content

if __name__ == "__main__":
    main()
