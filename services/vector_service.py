"""
Vector Database Service using Qdrant for RAG pipelines
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, Range
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tech_config import tech_settings


class VectorService:
    """Service for vector database operations using Qdrant"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if VectorService._initialized:
            return
            
        self.logger = logging.getLogger("traffix.vector_service")
        # Force local storage for now
        local_path = "./data/qdrant_db"
        self.logger.info(f"Using local Qdrant storage at: {local_path}")
        
        try:
            # Try with force_disable_check_same_thread for concurrent access
            self.client = QdrantClient(path=local_path, force_disable_check_same_thread=True)
        except Exception as e:
            self.logger.warning(f"Failed to connect to Qdrant at {local_path}: {e}")
            # Try without the flag
            try:
                self.client = QdrantClient(path=local_path)
            except Exception as e2:
                self.logger.warning(f"Failed again, using in-memory: {e2}")
                # Fallback to in-memory mode
                try:
                    self.client = QdrantClient(":memory:")
                    self.logger.info("Using in-memory Qdrant storage as fallback")
                except Exception as e3:
                    self.logger.error(f"Failed to create Qdrant client: {e3}")
                    raise
        
        # Use consistent embedding model for all data types
        self.embeddings = OpenAIEmbeddings(
            model=tech_settings.embedding_model,
            openai_api_key=tech_settings.openai_api_key
        )
        
        # Get the actual embedding dimension from the model
        self.embedding_dimension = tech_settings.embedding_dimension
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=tech_settings.chunk_size,
            chunk_overlap=tech_settings.chunk_overlap
        )
        self.collection_name = tech_settings.qdrant_collection_name
        self._ensure_collection_exists()
        VectorService._initialized = True
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance - useful for testing or cleanup"""
        if cls._instance and hasattr(cls._instance, 'client'):
            try:
                cls._instance.client.close()
            except:
                pass
        cls._instance = None
        cls._initialized = False
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=tech_settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def add_traffic_data(self, traffic_data: List[Dict[str, Any]], 
                             location: str, data_type: str = "traffic") -> bool:
        """Add traffic data to vector database"""
        try:
            points = []
            
            for item in traffic_data:
                # Create text representation of traffic data
                text = self._create_traffic_text(item, location)
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = await self._get_embedding(chunk)
                    
                    # Create point
                    point_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "location": location,
                            "data_type": data_type,
                            "timestamp": item.get("timestamp", datetime.now().isoformat()),
                            "original_data": item,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    points.append(point)
            
            # Insert points into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            self.logger.info(f"Added {len(points)} traffic data points to vector database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add traffic data: {e}")
            return False
    
    async def add_news_data(self, news_data: List[Dict[str, Any]], 
                          location: str) -> bool:
        """Add news data to vector database with duplicate detection"""
        try:
            points = []
            new_articles = []
            
            for article in news_data:
                article_url = article.get("url", "")
                if not article_url:
                    continue
                
                # Check if article already exists
                if await self._article_exists(article_url):
                    self.logger.info(f"Article already exists, skipping: {article.get('title', '')[:50]}...")
                    continue
                
                new_articles.append(article)
                
                # Create text representation
                text = f"Title: {article.get('title', '')}\nContent: {article.get('content', '')}"
                
                # Split into chunks
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    embedding = await self._get_embedding(chunk)
                    
                    # Create deterministic UUID based on URL and chunk index
                    import uuid
                    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # Standard namespace
                    name = f"news_{article_url}_{i}"
                    point_id = str(uuid.uuid5(namespace, name))
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "location": location,
                            "data_type": "news",
                            "title": article.get("title", ""),
                            "url": article_url,
                            "published_at": article.get("published_at", ""),
                            "source": article.get("source", ""),
                            "relevance_score": article.get("relevance_score", 0.0),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "article_hash": hash(article_url),
                            # Enhanced metadata for better organization
                            "analysis_mode": article.get("analysis_mode", ""),
                            "time_period": article.get("time_period", ""),
                            "run_timestamp": article.get("run_timestamp", datetime.now().isoformat()),
                            "region_code": location.lower().replace(" ", "_")
                        }
                    )
                    points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                self.logger.info(f"Added {len(points)} new chunks from {len(new_articles)} articles to vector database")
            else:
                self.logger.info("No new articles to add - all articles already exist")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add news data: {e}")
            return False
    
    async def _article_exists(self, article_url: str) -> bool:
        """Check if an article with the given URL already exists"""
        try:
            # Search for existing articles with this URL
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * tech_settings.embedding_dimension,  # Dummy vector
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="url",
                            match=MatchValue(value=article_url)
                        )
                    ]
                ),
                limit=1
            )
            return len(search_results) > 0
        except Exception as e:
            self.logger.warning(f"Error checking if article exists: {e}")
            return False
    
    async def add_incident_data(self, incident_data: List[Dict[str, Any]], 
                              location: str) -> bool:
        """Add incident data to vector database"""
        try:
            points = []
            
            for incident in incident_data:
                text = f"Incident: {incident.get('description', '')}\nLocation: {incident.get('location', '')}\nSeverity: {incident.get('severity', '')}"
                
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    embedding = await self._get_embedding(chunk)
                    
                    point_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "location": location,
                            "data_type": "incident",
                            "incident_id": incident.get("incident_id", ""),
                            "severity": incident.get("severity", ""),
                            "start_time": incident.get("start_time", ""),
                            "end_time": incident.get("end_time", ""),
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            self.logger.info(f"Added {len(points)} incident data points to vector database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add incident data: {e}")
            return False
    
    async def add_ritis_data(self, ritis_data: List[Dict[str, Any]], 
                           location: str) -> bool:
        """Add RITIS traffic event data to vector database"""
        try:
            points = []
            
            for event in ritis_data:
                # Create text representation
                text = event.get("text_content", "")
                if not text:
                    # Fallback text creation
                    text_parts = []
                    if event.get("description"):
                        text_parts.append(f"Description: {event['description']}")
                    if event.get("location"):
                        text_parts.append(f"Location: {event['location']}")
                    if event.get("event_type"):
                        text_parts.append(f"Event Type: {event['event_type']}")
                    text = " | ".join(text_parts)
                
                if not text.strip():
                    continue
                
                # Split into chunks
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    embedding = await self._get_embedding(chunk)
                    
                    # Create deterministic UUID based on event ID and chunk index
                    import uuid
                    event_id = event.get("event_id", "unknown")
                    # Create a deterministic UUID from the hash
                    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # Standard namespace
                    name = f"ritis_{event_id}_{i}"
                    point_id = str(uuid.uuid5(namespace, name))
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "location": location,
                            "data_type": "ritis_event",
                            "event_id": event_id,
                            "timestamp": event.get("timestamp", ""),
                            "latitude": event.get("latitude"),
                            "longitude": event.get("longitude"),
                            "event_type": event.get("event_type", ""),
                            "severity": event.get("severity"),
                            "description": event.get("description", ""),
                            "highway": event.get("highway", ""),
                            "direction": event.get("direction", ""),
                            "impact_type": event.get("impact_type", ""),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "data_source": "ritis_excel"
                        }
                    )
                    points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                self.logger.info(f"Added {len(points)} RITIS event chunks to vector database")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add RITIS data: {e}")
            return False
    
    async def add_ritis_data_with_progress(self, ritis_data: List[Dict[str, Any]], 
                                         location: str, max_events: int = 10000) -> bool:
        """Add RITIS traffic event data to vector database with optimized parallel processing"""
        try:
            import asyncio
            import uuid
            
            # Limit events for faster processing
            if len(ritis_data) > max_events:
                print(f"⚡ Limiting to {max_events} most recent events for speed")
                ritis_data = ritis_data[-max_events:]  # Take most recent events
            
            total_events = len(ritis_data)
            processed_events = 0
            points = []
            batch_size = 200  # Larger batch size for better performance
            
            print(f"📊 Processing {total_events} events with optimized parallel embeddings...")
            
            async def process_event_batch(event_batch):
                """Process a batch of events in parallel with timeout"""
                tasks = []
                for event in event_batch:
                    task = self._process_single_event(event, location)
                    tasks.append(task)
                
                # Wait for all events in batch to complete with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=60.0  # 60 second timeout per batch
                    )
                except asyncio.TimeoutError:
                    print(f"⚠️ Batch timeout, continuing with next batch")
                    return []
                
                # Collect successful results
                batch_points = []
                for result in batch_results:
                    if isinstance(result, list):
                        batch_points.extend(result)
                    elif isinstance(result, Exception):
                        print(f"⚠️ Error processing event: {result}")
                
                return batch_points
            
            # Process events in larger batches
            for i in range(0, total_events, batch_size):
                event_batch = ritis_data[i:i + batch_size]
                
                # Process batch in parallel
                batch_points = await process_event_batch(event_batch)
                points.extend(batch_points)
                
                processed_events += len(event_batch)
                
                # Update progress
                progress = (processed_events / total_events) * 100
                bar_length = 50
                filled_length = int(bar_length * processed_events // total_events)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f'\r🔄 Progress: |{bar}| {progress:.1f}% ({processed_events}/{total_events}) events processed', end='', flush=True)
                
                # Store in larger batches for better performance
                if len(points) >= 2000:  # Larger batch size
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"\n💾 Stored batch of {len(points)} events")
                    points = []  # Reset for next batch
            
            # Store remaining points
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"\n💾 Stored final batch of {len(points)} events")
            
            print(f"\n✅ Successfully processed {processed_events} events with optimized parallel processing!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.logger.error(f"Failed to add RITIS data with progress: {e}")
            return False
    
    async def add_ritis_data_fast(self, ritis_data: List[Dict[str, Any]], 
                                location: str, max_events: int = 5000) -> bool:
        """Ultra-fast RITIS data loading with smart sampling and clustering"""
        try:
            import asyncio
            import uuid
            from collections import defaultdict
            
            # Smart sampling: take most recent events and cluster by type
            if len(ritis_data) > max_events:
                print(f"⚡ Smart sampling: selecting {max_events} most relevant events")
                
                # Sort by timestamp (most recent first)
                sorted_events = sorted(ritis_data, 
                                     key=lambda x: x.get('timestamp', ''), 
                                     reverse=True)
                
                # Take most recent events
                sampled_events = sorted_events[:max_events]
                
                # Show sampling statistics
                event_types = defaultdict(int)
                for event in sampled_events:
                    event_type = event.get('event_type', 'Unknown')
                    event_types[event_type] += 1
                
                print("📊 Event type distribution:")
                for event_type, count in event_types.items():
                    print(f"  {event_type}: {count} events")
            else:
                sampled_events = ritis_data
            
            total_events = len(sampled_events)
            processed_events = 0
            points = []
            batch_size = 500  # Very large batch size for maximum speed
            
            print(f"🚀 Processing {total_events} events with ultra-fast parallel embeddings...")
            
            async def process_event_batch(event_batch):
                """Process a large batch of events in parallel"""
                tasks = []
                for event in event_batch:
                    task = self._process_single_event(event, location)
                    tasks.append(task)
                
                # Process with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=120.0  # 2 minute timeout for large batches
                    )
                except asyncio.TimeoutError:
                    print(f"⚠️ Large batch timeout, continuing with next batch")
                    return []
                
                # Collect successful results
                batch_points = []
                for result in batch_results:
                    if isinstance(result, list):
                        batch_points.extend(result)
                    elif isinstance(result, Exception):
                        print(f"⚠️ Error processing event: {result}")
                
                return batch_points
            
            # Process events in very large batches
            for i in range(0, total_events, batch_size):
                event_batch = sampled_events[i:i + batch_size]
                
                # Process batch in parallel
                batch_points = await process_event_batch(event_batch)
                points.extend(batch_points)
                
                processed_events += len(event_batch)
                
                # Update progress
                progress = (processed_events / total_events) * 100
                bar_length = 50
                filled_length = int(bar_length * processed_events // total_events)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f'\r🔄 Progress: |{bar}| {progress:.1f}% ({processed_events}/{total_events}) events processed', end='', flush=True)
                
                # Store in very large batches for maximum efficiency
                if len(points) >= 5000:  # Very large batch size
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"\n💾 Stored mega-batch of {len(points)} events")
                    points = []  # Reset for next batch
            
            # Store remaining points
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"\n💾 Stored final batch of {len(points)} events")
            
            print(f"\n✅ Ultra-fast processing complete! Processed {processed_events} events!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.logger.error(f"Failed to add RITIS data fast: {e}")
            return False
    
    async def _process_single_event(self, event: Dict[str, Any], location: str) -> List[PointStruct]:
        """Process a single event and return its points"""
        try:
            import uuid
            import asyncio
            
            # Create text representation
            text = event.get("text_content", "")
            if not text:
                # Fallback text creation
                text_parts = []
                if event.get("description"):
                    text_parts.append(f"Description: {event['description']}")
                if event.get("location"):
                    text_parts.append(f"Location: {event['location']}")
                if event.get("event_type"):
                    text_parts.append(f"Event Type: {event['event_type']}")
                text = " | ".join(text_parts)
            
            if not text.strip():
                return []
            
            # For RITIS events, use the text as-is (no chunking needed)
            # since each event is already a short, focused piece of information
            chunks = [text]  # Use the whole text as one chunk
            points = []
            
            # Generate embedding using consistent model
            embedding = await self._get_embedding(text)
            embeddings = [embedding]
            
            # Create points for each chunk
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create deterministic UUID based on event ID and chunk index
                event_id = event.get("event_id", "unknown")
                namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                name = f"ritis_{event_id}_{i}"
                point_id = str(uuid.uuid5(namespace, name))
                
                # Ensure embedding dimension matches collection
                if len(embedding) != self.embedding_dimension:
                    self.logger.warning(f"Embedding dimension mismatch: {len(embedding)} vs {self.embedding_dimension}")
                    # Truncate or pad as needed
                    if len(embedding) > self.embedding_dimension:
                        embedding = embedding[:self.embedding_dimension]
                    else:
                        embedding = embedding + [0.0] * (self.embedding_dimension - len(embedding))
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "location": event.get("location", ""),  # Specific sub-location (street, highway, etc.)
                        "region": location,  # Broad geographic region (Northern Virginia, Washington DC)
                        "data_type": "ritis_event",
                        "event_id": event_id,
                        "timestamp": event.get("timestamp", ""),
                        "latitude": event.get("latitude"),
                        "longitude": event.get("longitude"),
                        "event_type": event.get("event_type", ""),
                        "severity": event.get("severity"),
                        "description": event.get("description", ""),
                        "highway": event.get("highway", ""),
                        "direction": event.get("direction", ""),
                        "impact_type": event.get("impact_type", ""),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "data_source": "ritis_excel",
                        "embedding_model": "text-embedding-3-small"  # Mark as fast embedding
                    }
                )
                points.append(point)
            
            return points
            
        except Exception as e:
            self.logger.error(f"Error processing single event: {e}")
            return []
    
    async def search_similar_content(self, query: str, location: str, 
                                   data_types: List[str] = None, 
                                   limit: int = None,
                                   analysis_mode: str = None,
                                   time_period: str = None,
                                   region_code: str = None) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity with enhanced filtering"""
        try:
            # Generate query embedding using consistent model
            query_embedding = await self._get_embedding(query)
            
            # Build comprehensive filter
            filter_conditions = []
            
            # Location filter (flexible - can use location or region_code)
            if region_code:
                filter_conditions.append(
                    FieldCondition(key="region_code", match=MatchValue(value=region_code))
                )
            elif location:
                # For RITIS events, filter by region; for news, filter by location
                if data_types and 'ritis_event' in data_types:
                    filter_conditions.append(
                        FieldCondition(key="region", match=MatchValue(value=location))
                    )
                else:
                    filter_conditions.append(
                        FieldCondition(key="location", match=MatchValue(value=location))
                    )
            
            # Data type filter
            if data_types:
                if len(data_types) == 1:
                    filter_conditions.append(
                        FieldCondition(key="data_type", match=MatchValue(value=data_types[0]))
                    )
                else:
                    # Multiple data types - use MatchAny
                    filter_conditions.append(
                        FieldCondition(key="data_type", match=MatchAny(any=data_types))
                    )
            
            # Analysis mode filter
            if analysis_mode:
                filter_conditions.append(
                    FieldCondition(key="analysis_mode", match=MatchValue(value=analysis_mode))
                )
            
            # Time period filter - handle both metadata time_period and actual timestamps
            if time_period:
                # First try to filter by time_period metadata (for news articles)
                filter_conditions.append(
                    FieldCondition(key="time_period", match=MatchValue(value=time_period))
                )
                
                # For RITIS events, we need to also filter by actual timestamp
                # Calculate the cutoff date based on time_period
                from datetime import datetime, timedelta
                now = datetime.now()
                time_mappings = {
                    '24h': now - timedelta(hours=24),
                    '48h': now - timedelta(hours=48),
                    '1w': now - timedelta(weeks=1),
                    '2w': now - timedelta(weeks=2),
                    '1m': now - timedelta(days=30),
                    '3m': now - timedelta(days=90)
                }
                
                if time_period in time_mappings:
                    cutoff_date = time_mappings[time_period]
                    # Add timestamp filter for RITIS events (convert to timestamp)
                    filter_conditions.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(
                                gte=cutoff_date.timestamp()
                            )
                        )
                    )
            
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Debug logging
            self.logger.debug(f"Search filter: {search_filter}")
            self.logger.debug(f"Query vector length: {len(query_embedding)}")
            self.logger.debug(f"Limit: {limit or tech_settings.top_k_results}")
            
            # Search with very low threshold for maximum speed and results
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit or tech_settings.top_k_results,
                score_threshold=0.1  # Very low threshold for maximum speed
            )
            
            self.logger.debug(f"Raw search results: {len(search_results)} items")
            
            # Format results
            results = []
            for result in search_results:
                # Extract all relevant fields from payload
                result_dict = {
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "data_type": result.payload.get("data_type", ""),
                    "location": result.payload.get("location", ""),
                }
                
                # Add all other payload fields directly
                for key, value in result.payload.items():
                    if key not in ["text", "data_type", "location"]:
                        result_dict[key] = value
                
                results.append(result_dict)
            
            self.logger.info(f"Found {len(results)} similar content items")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "total_points": collection_info.points_count,
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def get_region_stats(self) -> Dict[str, Any]:
        """Get statistics by region"""
        try:
            # Get all points and group by region
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust as needed
            )[0]
            
            region_counts = {}
            mode_counts = {}
            
            for point in all_points:
                payload = point.payload
                region = payload.get("location", "Unknown")
                mode = payload.get("analysis_mode", "Unknown")
                
                region_counts[region] = region_counts.get(region, 0) + 1
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            return {
                "regions": region_counts,
                "analysis_modes": mode_counts,
                "total_chunks": len(all_points)
            }
        except Exception as e:
            self.logger.error(f"Failed to get region stats: {e}")
            return {}


    async def get_context_for_analysis(self, query: str, location: str, 
                                     analysis_type: str = "general") -> Dict[str, List[Dict[str, Any]]]:
        """Get relevant context for analysis based on query and location"""
        try:
            # Determine data types based on analysis type
            data_type_mapping = {
                "anomaly_investigation": ["traffic", "incident", "news"],
                "leadership_summary": ["traffic", "news", "incident"],
                "pattern_analysis": ["traffic", "incident"],
                "general": ["traffic", "news", "incident"]
            }
            
            data_types = data_type_mapping.get(analysis_type, ["traffic", "news", "incident"])
            
            # Search for each data type
            context = {}
            for data_type in data_types:
                results = await self.search_similar_content(
                    query=query,
                    location=location,
                    data_types=[data_type],
                    limit=3
                )
                context[data_type] = results
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get context: {e}")
            return {}
    
    def _create_traffic_text(self, item: Dict[str, Any], location: str) -> str:
        """Create text representation of traffic data"""
        return f"""
        Location: {location}
        Timestamp: {item.get('timestamp', '')}
        Speed: {item.get('speed', 0)} mph
        Volume: {item.get('volume', 0)} vehicles
        Occupancy: {item.get('occupancy', 0):.2f}
        Congestion Level: {item.get('congestion_level', 'unknown')}
        Incident Detected: {item.get('incident_detected', False)}
        """.strip()
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI (large model for complex content)"""
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to get embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    
    async def clear_location_data(self, location: str) -> bool:
        """Clear all data for a specific location"""
        try:
            # Delete points with location filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="location", match=MatchValue(value=location))]
                )
            )
            
            self.logger.info(f"Cleared all data for location: {location}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear location data: {e}")
            return False
    
    async def search_similar_content(self, query: str, location: str = None, 
                                   data_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content in the vector database"""
        try:
            # Generate query embedding using consistent model
            query_embedding = await self._get_embedding(query)
            
            # Build filter conditions
            filter_conditions = []
            
            # Location filter (flexible - can use location or region_code)
            if location:
                # For RITIS events, filter by region; for news, filter by location
                if data_types and 'ritis_event' in data_types:
                    filter_conditions.append(
                        FieldCondition(key="region", match=MatchValue(value=location))
                    )
                else:
                    filter_conditions.append(
                        FieldCondition(key="location", match=MatchValue(value=location))
                    )
            
            # Data type filter
            if data_types:
                if len(data_types) == 1:
                    filter_conditions.append(
                        FieldCondition(key="data_type", match=MatchValue(value=data_types[0]))
                    )
                else:
                    filter_conditions.append(
                        FieldCondition(key="data_type", match=MatchAny(any=data_types))
                    )
            
            # Create filter
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload.get("text", ""),
                    "location": result.payload.get("location", ""),
                    "region": result.payload.get("region", ""),
                    "data_type": result.payload.get("data_type", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                    "score": result.score,
                    "id": result.id
                })
            
            self.logger.info(f"Found {len(results)} similar content items")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}


def get_vector_service() -> VectorService:
    """Get VectorService singleton instance"""
    if VectorService._instance is None:
        VectorService._instance = VectorService()
    return VectorService._instance

def reset_vector_service():
    """Reset the singleton instance (for testing/debugging)"""
    VectorService._instance = None
