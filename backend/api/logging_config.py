"""
Structured logging configuration
Production-ready JSON logs with context
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
import contextvars

# Context variable for request tracking
request_context = contextvars.ContextVar('request_context', default={})

class JSONFormatter(logging.Formatter):
    """
    Format logs as JSON for easy parsing and searching
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request context if available
        context = request_context.get()
        if context:
            log_data.update(context)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)

def setup_logging(log_level: str = "INFO", json_logs: bool = True):
    """
    Setup structured logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Use JSON format (True for production)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Set formatter
    if json_logs:
        formatter = JSONFormatter()
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    return root_logger

def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    Log with additional context fields
    
    Usage:
        log_with_context(logger, "info", "User logged in", user_id="123", ip="1.2.3.4")
    """
    extra_fields = kwargs
    getattr(logger, level.lower())(message, extra={'extra_fields': extra_fields})

def set_request_context(
    request_id: str = None,
    conversation_id: str = None,
    user_id: str = None,
    ip_address: str = None,
    **kwargs
):
    """
    Set context for all logs in this request
    
    Call this at the start of each request
    """
    context = {}
    
    if request_id:
        context['request_id'] = request_id
    if conversation_id:
        context['conversation_id'] = conversation_id
    if user_id:
        context['user_id'] = user_id
    if ip_address:
        context['ip_address'] = ip_address
    
    context.update(kwargs)
    request_context.set(context)

def clear_request_context():
    """Clear request context"""
    request_context.set({})
