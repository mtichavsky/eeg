"""
Structured logging configuration using structlog.

This module provides JSON logging with automatic request tracing.
Middleware adds request_id and user_id to all logs for distributed tracing.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

from api.config import config


def add_log_level(_logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add log level to event dict.

    :param WrappedLogger _logger: The logger instance (unused, required by structlog protocol).
    :param str method_name: The method name.
    :param EventDict event_dict: The event dictionary.
    :return: Updated event dictionary.
    :rtype: EventDict
    """
    event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging with JSON or human-readable output.

    Default: JSON output for log aggregation systems.
    JSON_PRETTY_PRINT=true: Colored human-readable logs.
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level.upper(), logging.INFO),
    )

    # Structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if config.json_pretty_print:
        # Human-readable colored output (JSON_PRETTY_PRINT=true)
        processors = shared_processors + [
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for log aggregation (default)
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    :param str name: Logger name (usually __name__).
    :return: Configured structlog logger.
    :rtype: structlog.BoundLogger
    """
    return structlog.get_logger(name)
