import logging
from unittest.mock import patch, MagicMock
from src.utils.logging_config import setup_logging

def test_setup_logging_creates_handlers():
    """Verify that setup_logging adds stream and file handlers to the root logger."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    
    try:
       with patch("src.utils.logging_config.Path.mkdir") as mock_mkdir, \
                 patch("src.utils.logging_config.RotatingFileHandler") as mock_file_handler:
            
            # Reset root logger handlers
            root_logger.handlers = []
            
            setup_logging()
            
            mock_mkdir.assert_called_once()
            mock_file_handler.assert_called_once()
            
            assert len(root_logger.handlers) >= 2 # File and Stream
    finally:
        # Restore previous handlers so we don't break other tests
        root_logger.handlers = original_handlers
