"""
RITIS Data Processor for Excel Files
Handles loading and processing RITIS traffic events from Excel files
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger("traffix.ritis_processor")


class RITISProcessor:
    """Process RITIS traffic events from Excel files"""
    
    def __init__(self, data_directory: str = "./data"):
        self.data_directory = Path(data_directory)
        self.ritis_directory = self.data_directory / "ritis"
        self.ritis_directory.mkdir(exist_ok=True)
        
        # Expected RITIS columns (adjust based on your actual Excel structure)
        self.expected_columns = [
            'event_id', 'timestamp', 'latitude', 'longitude', 
            'event_type', 'severity', 'description', 'location',
            'region', 'highway', 'direction', 'impact_type'
        ]
        
    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load RITIS data from Excel or CSV file"""
        try:
            logger.info(f"Loading RITIS data from: {file_path}")
            
            # Check file extension and load accordingly
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                # Load CSV file
                df = pd.read_csv(file_path)
                logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            elif file_ext in ['.xlsx', '.xls']:
                # Load Excel file
                try:
                    df = pd.read_excel(file_path, sheet_name=0)
                except Exception:
                    # Try reading all sheets and combine
                    excel_file = pd.ExcelFile(file_path)
                    sheets = excel_file.sheet_names
                    logger.info(f"Found sheets: {sheets}")
                    
                    # Read the first sheet or largest sheet
                    df = pd.read_excel(file_path, sheet_name=sheets[0])
                logger.info(f"Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
            else:
                # Try to auto-detect format
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"Auto-detected CSV format with {len(df)} rows and {len(df.columns)} columns")
                except Exception:
                    df = pd.read_excel(file_path, sheet_name=0)
                    logger.info(f"Auto-detected Excel format with {len(df)} rows and {len(df.columns)} columns")
            
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate RITIS data"""
        try:
            logger.info("Cleaning and validating RITIS data...")
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Convert timestamp columns if they exist
            timestamp_columns = ['timestamp', 'time', 'datetime', 'event_time', 'start_time']
            for col in timestamp_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Converted {col} to datetime")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to datetime: {e}")
            
            # Clean text columns
            text_columns = ['description', 'event_type', 'location', 'highway', 'direction']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace('nan', '')
            
            # Handle numeric columns
            numeric_columns = ['latitude', 'longitude', 'severity']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing critical data
            critical_columns = []
            if 'latitude' in df.columns:
                critical_columns.append('latitude')
            if 'longitude' in df.columns:
                critical_columns.append('longitude')
            
            if critical_columns:
                initial_count = len(df)
                df = df.dropna(subset=critical_columns)
                logger.info(f"Removed {initial_count - len(df)} rows with missing coordinates")
            
            logger.info(f"Cleaned data: {len(df)} rows remaining")
            return df
            
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            raise
    
    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Excel columns to standardized RITIS format"""
        try:
            logger.info("Mapping columns to standardized format...")
            
            # Create a mapping dictionary based on common column names
            column_mapping = {}
            
            # Map common variations - updated for RITIS CSV format
            mappings = {
                'event_id': ['event id', 'id', 'event_id', 'incident_id', 'event_number'],
                'timestamp': ['start time', 'timestamp', 'time', 'datetime', 'event_time', 'start_time', 'date'],
                'latitude': ['latitude', 'lat', 'y_coord', 'y'],
                'longitude': ['longitude', 'lon', 'lng', 'x_coord', 'x'],
                'event_type': ['standardized type', 'agency-specific type', 'type', 'event_type', 'incident_type', 'category'],
                'severity': ['severity', 'level', 'priority', 'impact_level'],
                'description': ['operator notes', 'description', 'details', 'summary', 'notes'],
                'location': ['location', 'address', 'place', 'street'],
                'region': ['region', 'area', 'district', 'zone'],
                'highway': ['road', 'highway', 'route', 'corridor'],
                'direction': ['direction', 'dir', 'travel_direction', 'heading'],
                'impact_type': ['edc incident type', 'impact_type', 'impact', 'effect', 'disruption'],
                'county': ['county'],
                'state': ['state'],
                'duration': ['duration (incident clearance time)', 'duration', 'clearance_time'],
                'agency': ['agency'],
                'system': ['system']
            }
            
            # Find matching columns (case-insensitive)
            for standard_name, variations in mappings.items():
                for variation in variations:
                    # Check exact match first
                    if variation in df.columns:
                        column_mapping[variation] = standard_name
                        break
                    # Check case-insensitive match
                    for col in df.columns:
                        if col.lower().replace(' ', '') == variation.lower().replace(' ', ''):
                            column_mapping[col] = standard_name
                            break
                    else:
                        continue
                    break
            
            # Apply mapping
            df = df.rename(columns=column_mapping)
            logger.info(f"Mapped columns: {column_mapping}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to map columns: {e}")
            raise
    
    def filter_by_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Filter data by region"""
        try:
            if 'region' not in df.columns:
                logger.warning("No region column found, returning all data")
                return df
            
            # Define region mappings
            region_mappings = {
                'Northern Virginia': ['northern virginia', 'nova', 'northern va', 'va'],
                'Washington DC': ['washington dc', 'dc', 'district of columbia', 'washington', 'wmata', 'metro', 'metropolitan']
            }
            
            region_terms = region_mappings.get(region, [region.lower()])
            
            # Filter data
            filtered_df = df[df['region'].str.lower().isin([term.lower() for term in region_terms])]
            
            logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} for region: {region}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Failed to filter by region: {e}")
            return df
    
    def filter_by_time_period(self, df: pd.DataFrame, time_period: str) -> pd.DataFrame:
        """Filter data by time period"""
        try:
            if 'timestamp' not in df.columns:
                logger.warning("No timestamp column found, returning all data")
                return df
            
            # Define time period mappings
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
                filtered_df = df[df['timestamp'] >= cutoff_date]
                logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} for time period: {time_period}")
                return filtered_df
            else:
                logger.warning(f"Unknown time period: {time_period}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to filter by time period: {e}")
            return df
    
    def convert_to_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of event dictionaries"""
        try:
            events = []
            
            for _, row in df.iterrows():
                event = {
                    'event_id': str(row.get('event_id', '')),
                    'timestamp': row.get('timestamp'),
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'event_type': str(row.get('event_type', '')),
                    'severity': row.get('severity'),
                    'description': str(row.get('description', '')),
                    'location': str(row.get('location', '')),
                    'region': str(row.get('region', '')),
                    'highway': str(row.get('highway', '')),
                    'direction': str(row.get('direction', '')),
                    'impact_type': str(row.get('impact_type', '')),
                    'county': str(row.get('county', '')),
                    'state': str(row.get('state', '')),
                    'duration': str(row.get('duration', '')),
                    'agency': str(row.get('agency', '')),
                    'system': str(row.get('system', '')),
                    'data_source': 'ritis_excel'
                }
                
                # Create text representation for vector search
                text_parts = []
                if event['event_type']:
                    text_parts.append(f"Event Type: {event['event_type']}")
                if event['location']:
                    text_parts.append(f"Location: {event['location']}")
                if event['highway']:
                    text_parts.append(f"Road: {event['highway']}")
                if event['direction']:
                    text_parts.append(f"Direction: {event['direction']}")
                if event['county']:
                    text_parts.append(f"County: {event['county']}")
                if event['description']:
                    text_parts.append(f"Notes: {event['description']}")
                if event['impact_type']:
                    text_parts.append(f"Impact: {event['impact_type']}")
                
                event['text_content'] = " | ".join(text_parts)
                
                events.append(event)
            
            logger.info(f"Converted {len(df)} rows to {len(events)} events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to convert to events: {e}")
            raise
    
    def process_excel_file(self, file_path: str, region: Optional[str] = None, 
                          time_period: Optional[str] = None) -> List[Dict[str, Any]]:
        """Complete processing pipeline for RITIS Excel file"""
        try:
            logger.info(f"Processing RITIS Excel file: {file_path}")
            
            # Load data
            df = self.load_excel_file(file_path)
            
            # Clean and validate
            df = self.clean_and_validate_data(df)
            
            # Map columns
            df = self.map_columns(df)
            
            # Filter by region if specified
            if region:
                df = self.filter_by_region(df, region)
            
            # Filter by time period if specified
            if time_period:
                df = self.filter_by_time_period(df, time_period)
            
            # Convert to events
            events = self.convert_to_events(df)
            
            logger.info(f"Successfully processed {len(events)} RITIS events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to process RITIS Excel file: {e}")
            raise
    
    def save_processed_data(self, events: List[Dict[str, Any]], filename: str = None) -> str:
        """Save processed events to JSON file"""
        try:
            if filename is None:
                filename = f"ritis_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            file_path = self.ritis_directory / filename
            
            import json
            with open(file_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
            
            logger.info(f"Saved {len(events)} events to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise


def get_ritis_processor() -> RITISProcessor:
    """Get RITIS processor instance"""
    return RITISProcessor()
