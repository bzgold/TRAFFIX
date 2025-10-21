"""
Base agent class for all Traffix agents
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from models import AgentStatus, AgentTask


class BaseAgent(ABC):
    """Base class for all Traffix agents"""
    
    def __init__(self, agent_type: str, max_retries: int = 3):
        self.agent_type = agent_type
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"traffix.{agent_type}")
        self.current_task: Optional[AgentTask] = None
        
    async def execute_task(self, input_data: Dict[str, Any]) -> AgentTask:
        """Execute a task with error handling and retries"""
        task_id = str(uuid4())
        self.current_task = AgentTask(
            task_id=task_id,
            agent_type=self.agent_type,
            status=AgentStatus.RUNNING,
            input_data=input_data,
            created_at=datetime.now()
        )
        
        self.logger.info(f"Starting task {task_id}")
        
        for attempt in range(self.max_retries):
            try:
                result = await self._execute(input_data)
                self.current_task.status = AgentStatus.COMPLETED
                self.current_task.output_data = result
                self.current_task.completed_at = datetime.now()
                self.logger.info(f"Task {task_id} completed successfully")
                return self.current_task
                
            except Exception as e:
                self.logger.error(f"Task {task_id} attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.current_task.status = AgentStatus.ERROR
                    self.current_task.error_message = str(e)
                    self.current_task.completed_at = datetime.now()
                    return self.current_task
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return self.current_task
    
    @abstractmethod
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's specific logic"""
        pass
    
    def get_status(self) -> Optional[AgentStatus]:
        """Get current agent status"""
        return self.current_task.status if self.current_task else AgentStatus.IDLE
    
    def get_current_task(self) -> Optional[AgentTask]:
        """Get current task details"""
        return self.current_task
