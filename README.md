# Traffix - AI Storytelling for Transportation Analytics

Traffix is an AI-powered storytelling assistant that integrates structured traffic data (RITIS) and unstructured sources (news articles, incident summaries) to reduce analysts' manual investigation and provide quick or deep narrative reports. It consolidates multiple data sources to automatically retrieve, correlate, and synthesize information, highlighting notable patterns or events and generating evidence-based narratives that can be exported or emailed.

# CODE TO APP: https://github.com/bzgold/TRAFFIX

# LOOM VIDEO: https://www.loom.com/share/f0df205ee7614b4c9fd32046a5904707?sid=a825d7e4-4a44-4fcb-a7a2-1107f30bde0a

# WRITTEN PAPER: https://github.com/bzgold/TRAFFIX/blob/main/TRAFFIX%20REPORT.pdf

## Features

- **Multi-Agent Architecture**: Specialized agents for data collection, analysis, storytelling, and reporting
- **Multiple Analysis Modes**:
  - **Quick Mode**: Concise daily summaries (30-60 seconds)
  - **Deep Mode**: Detailed research reports with causes, contributing factors, and historical context (2-5 minutes)
  - **Anomaly Investigation**: Investigates why congestion patterns changed (1-3 minutes)
  - **Leadership Summary**: Executive summaries for transportation leadership (2-4 minutes)
  - **Pattern Analysis**: Identifies recurring congestion patterns and mitigation strategies (3-5 minutes)
- **Multi-Source Data Integration**: RITIS, news, incidents, weather, social media
- **AI-Powered Storytelling**: Converts data into compelling narratives
- **Export & Email Capabilities**: Reports can be exported in multiple formats or emailed
- **Pattern Recognition**: Identifies recurring congestion patterns and causes to inform mitigation strategies
- **Multiple Output Formats**: HTML, JSON, Markdown, PDF reports
- **RESTful API**: Easy integration with existing systems

## System Architecture

Traffix uses a sophisticated multi-agent system with specialized roles and responsibilities:

### Agent Architecture

#### Core Agents
- **Supervisor Agent (Orchestrator)**: Routes queries between research and writing teams, creates work plans, and monitors execution
- **Research Agent (Analyst)**: Queries RITIS + Tavily, extracts key incidents, identifies causes, and performs comprehensive data analysis
- **Writer Agent (Storyteller)**: Synthesizes data into narratives (concise summaries or detailed reports) with compelling storytelling
- **Editor Agent (Copy & Context Checker)**: Ensures factual accuracy, readability, empathetic tone, and consistency
- **Evaluator Agent (QA)**: Uses RAGAS-style heuristics to improve pipeline quality and assess output quality

#### Legacy Agents (for backward compatibility)
- **Data Collector Agent**: Collects data from various sources (RITIS, news, incidents, weather, social media)
- **Analyzer Agent**: Performs statistical and AI-powered analysis to identify patterns and anomalies
- **Storyteller Agent**: Creates compelling narratives from analysis results
- **Reporter Agent**: Generates final reports in multiple formats
- **Pattern Analyzer Agent**: Identifies recurring patterns and suggests mitigation strategies

### Workflow
- **LangGraph Workflow**: Orchestrates multi-agent collaboration with intelligent routing and quality assurance
- **RAGAS Evaluation**: Comprehensive quality assessment using faithfulness, relevancy, correctness, and other metrics

### Modes

- **Quick Mode**: Optimized for speed, ideal for daily summaries and shift reports
- **Deep Mode**: Comprehensive analysis with historical patterns, trend analysis, and impact assessment
- **Anomaly Investigation**: Focused on uncovering the underlying causes of congestion anomalies
- **Leadership Summary**: Executive-ready reports for transportation leadership
- **Pattern Analysis**: Identifies recurring congestion patterns and generates mitigation strategies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TRAFFIX
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start required services:
```bash
# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant

# Or use Docker Compose
docker-compose up -d
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Initialize the system:
```bash
python startup.py
```

6. Run the system:
```bash
# Streamlit UI (Recommended)
streamlit run streamlit_app.py

# FastAPI backend
python main.py

# Demo
python demo.py
```

## Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/traffix

# RITIS API Configuration
RITIS_API_KEY=your_ritis_api_key_here
RITIS_BASE_URL=https://api.ritis.org

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# System Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=5
REPORT_OUTPUT_DIR=./reports
```

## API Usage

### Quick Analysis

```bash
curl -X POST "http://localhost:8000/analyze/quick" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "I-95 North",
    "time_range_hours": 24,
    "output_format": "html"
  }'
```

### Deep Analysis

```bash
curl -X POST "http://localhost:8000/analyze/deep" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "I-95 North",
    "time_range_hours": 168,
    "analysis_depth": "comprehensive",
    "output_format": "html"
  }'
```

### Get Insights

```bash
curl "http://localhost:8000/insights/I-95%20North?mode=quick"
```

## API Endpoints

- `GET /` - System information and status
- `GET /health` - Health check
- `GET /status` - System status
- `POST /analyze` - Run traffic analysis
- `POST /analyze/quick` - Quick analysis
- `POST /analyze/deep` - Deep analysis
- `GET /insights/{location}` - Get location insights
- `GET /modes` - Available analysis modes
- `GET /reports/{report_id}` - Get specific report
- `GET /docs` - API documentation

## Data Sources

### Structured Data
- **RITIS**: Real-time traffic data, incidents, speed, volume, occupancy
- **Weather**: Weather conditions affecting traffic
- **Incidents**: Traffic incidents and closures

### Unstructured Data
- **News**: Traffic-related news articles
- **Social Media**: Public sentiment and reports

## Report Formats

### HTML Reports
- Professional web-based reports
- Interactive elements and visualizations
- Executive summary and detailed analysis

### JSON Reports
- Machine-readable format
- Easy integration with other systems
- Complete data structure

### Markdown Reports
- Human-readable format
- Easy to share and version control
- Compatible with documentation systems

## Logging

The system provides comprehensive logging:

- **Main logs**: `logs/traffix.log`
- **Error logs**: `logs/traffix_errors.log`
- **Agent logs**: Individual log files for each agent
- **Performance logs**: `logs/performance.log`
- **API logs**: `logs/api.log`

## Development

### Project Structure

```
TRAFFIX/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── data_collector.py  # Data collection agent
│   ├── analyzer.py        # Analysis agent
│   ├── storyteller.py     # Storytelling agent
│   └── reporter.py        # Reporting agent
├── modes/                 # Analysis modes
│   ├── quick_mode.py      # Quick mode processor
│   └── deep_mode.py       # Deep mode processor
├── services/              # Data services
│   └── data_services.py   # Data integration services
├── models.py              # Data models
├── config.py              # Configuration
├── logging_config.py      # Logging setup
├── main.py                # Main application
└── requirements.txt       # Dependencies
```

### Adding New Data Sources

1. Create a new service class in `services/data_services.py`
2. Implement the data collection methods
3. Add the service to `DataIntegrationService`
4. Update the data models if needed

### Adding New Analysis Types

1. Extend the `AnalyzerAgent` class
2. Add new analysis methods
3. Update the mode processors to use new analyses
4. Add corresponding data models

## Technology Stack

**Core AI & LLM:**
- **OpenAI GPT-4o**: Narrative reasoning and report writing
- **text-embedding-3-large**: Mixed data embeddings (incidents + news)

**Orchestration & Coordination:**
- **LangGraph**: Multi-agent coordination and workflow management
- **OpenAI Agents SDK**: Advanced agent orchestration

**Vector Database & RAG:**
- **Qdrant**: High-performance vector database for local/hybrid RAG pipelines
- **LangChain**: RAG pipeline implementation

**Monitoring & Evaluation:**
- **LangSmith**: Track agent performance and reasoning
- **RAGAS**: Assess faithfulness, relevancy, and precision

**User Interface:**
- **Streamlit**: Interactive UI with Quick and Deep modes
- **Plotly**: Interactive visualizations

**Backend & API:**
- **FastAPI**: Web framework and API
- **Python 3.8+**: Core language
- **Pydantic**: Data validation
- **AsyncIO**: Asynchronous processing

## Performance

- **Quick Mode**: 30-60 seconds processing time
- **Deep Mode**: 2-5 minutes processing time
- **Concurrent Processing**: Up to 5 concurrent analyses
- **Memory Usage**: Optimized for large datasets
- **Scalability**: Horizontal scaling support
- **Vector Search**: Sub-second similarity search with Qdrant
- **RAG Performance**: High-quality context retrieval for analysis

## Monitoring

The system provides built-in monitoring:

- Real-time status endpoint
- Performance metrics logging
- Error tracking and reporting
- System health checks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for troubleshooting

## Roadmap

- [ ] Real RITIS API integration
- [ ] Real news API integration
- [ ] Database persistence
- [ ] Advanced visualizations
- [ ] Machine learning model integration
- [ ] Real-time streaming analysis
- [ ] Mobile app interface
- [ ] Advanced reporting templates
