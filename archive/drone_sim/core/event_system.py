"""
Pub-sub event system for parameter changes, emergency stops, and simulation resets
"""
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging

class EventType(Enum):
    """Types of events in the system"""
    PARAMETER_CHANGE = "parameter_change"
    EMERGENCY_STOP = "emergency_stop"
    SIMULATION_RESET = "simulation_reset"
    STATE_CHANGE = "state_change"
    CUSTOM = "custom"

class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    data: Dict[str, Any]
    source: str = "unknown"
    timestamp: float = None
    priority: EventPriority = EventPriority.NORMAL
    immediate: bool = False  # Whether to process immediately vs. phased
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class EventHandler:
    """Base event handler interface"""
    
    def handle_event(self, event: Event) -> bool:
        """
        Handle an event
        
        Args:
            event: The event to handle
            
        Returns:
            bool: True if event was handled successfully
        """
        raise NotImplementedError
        
    def can_handle(self, event_type: EventType) -> bool:
        """Check if this handler can handle the given event type"""
        raise NotImplementedError

class EventSystem:
    """
    Central event system using pub-sub pattern
    """
    
    def __init__(self):
        # Event handlers organized by event type
        self._handlers: Dict[EventType, List[EventHandler]] = {
            event_type: [] for event_type in EventType
        }
        
        # Direct callback subscriptions
        self._callbacks: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # Event queue for asynchronous processing
        self._event_queue: List[Event] = []
        self._queue_lock = threading.Lock()
        
        # Processing control
        self._processing = False
        self._process_thread = None
        
        # Emergency stop state
        self._emergency_stop_active = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Subscribe a callback to an event type
        
        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event occurs
        """
        self._callbacks[event_type].append(callback)
        self.logger.debug(f"Subscribed callback to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Unsubscribe a callback from an event type
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to remove
        """
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed callback from {event_type.value}")
            
    def register_handler(self, handler: EventHandler):
        """
        Register an event handler
        
        Args:
            handler: The event handler to register
        """
        for event_type in EventType:
            if handler.can_handle(event_type):
                self._handlers[event_type].append(handler)
                self.logger.debug(f"Registered handler for {event_type.value}")
                
    def unregister_handler(self, handler: EventHandler):
        """
        Unregister an event handler
        
        Args:
            handler: The event handler to remove
        """
        for event_type in EventType:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                self.logger.debug(f"Unregistered handler for {event_type.value}")
                
    def publish(self, event: Event):
        """
        Publish an event
        
        Args:
            event: The event to publish
        """
        self.logger.debug(f"Publishing event: {event.event_type.value} from {event.source}")
        
        # Handle emergency stops immediately
        if event.event_type == EventType.EMERGENCY_STOP:
            self._handle_emergency_stop(event)
            return
            
        # Process immediately if requested or if critical priority
        if event.immediate or event.priority == EventPriority.CRITICAL:
            self._process_event(event)
        else:
            # Add to queue for asynchronous processing
            with self._queue_lock:
                self._event_queue.append(event)
                # Sort by priority (higher priority first)
                self._event_queue.sort(key=lambda e: e.priority.value, reverse=True)
                
    def publish_parameter_change(self, parameter_name: str, old_value: Any, new_value: Any, 
                                source: str = "unknown", immediate: bool = False):
        """
        Convenience method to publish parameter change events
        
        Args:
            parameter_name: Name of the parameter that changed
            old_value: Previous value
            new_value: New value
            source: Source of the change
            immediate: Whether to process immediately
        """
        event = Event(
            event_type=EventType.PARAMETER_CHANGE,
            data={
                'parameter_name': parameter_name,
                'old_value': old_value,
                'new_value': new_value
            },
            source=source,
            immediate=immediate
        )
        self.publish(event)
        
    def publish_emergency_stop(self, reason: str, source: str = "unknown"):
        """
        Convenience method to publish emergency stop events
        
        Args:
            reason: Reason for the emergency stop
            source: Source of the emergency stop
        """
        event = Event(
            event_type=EventType.EMERGENCY_STOP,
            data={'reason': reason},
            source=source,
            priority=EventPriority.CRITICAL,
            immediate=True
        )
        self.publish(event)
        
    def publish_simulation_reset(self, source: str = "unknown"):
        """
        Convenience method to publish simulation reset events
        
        Args:
            source: Source of the reset
        """
        event = Event(
            event_type=EventType.SIMULATION_RESET,
            data={},
            source=source,
            priority=EventPriority.HIGH,
            immediate=True
        )
        self.publish(event)
        
    def start_processing(self):
        """Start asynchronous event processing"""
        if self._processing:
            return
            
        self._processing = True
        self._process_thread = threading.Thread(target=self._process_events, daemon=True)
        self._process_thread.start()
        self.logger.info("Started event processing")
        
    def stop_processing(self):
        """Stop asynchronous event processing"""
        self._processing = False
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=1.0)
        self.logger.info("Stopped event processing")
        
    def process_pending_events(self):
        """Process all pending events in the queue"""
        with self._queue_lock:
            events_to_process = self._event_queue.copy()
            self._event_queue.clear()
            
        for event in events_to_process:
            self._process_event(event)
            
    def clear_queue(self):
        """Clear the event queue"""
        with self._queue_lock:
            self._event_queue.clear()
            
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active"""
        return self._emergency_stop_active
        
    def clear_emergency_stop(self):
        """Clear the emergency stop state"""
        self._emergency_stop_active = False
        self.logger.info("Emergency stop cleared")
        
    def _process_events(self):
        """Background thread for processing events"""
        while self._processing:
            try:
                with self._queue_lock:
                    if self._event_queue:
                        event = self._event_queue.pop(0)
                    else:
                        event = None
                        
                if event:
                    self._process_event(event)
                else:
                    time.sleep(0.001)  # Small delay when no events
                    
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                
    def _process_event(self, event: Event):
        """
        Process a single event
        
        Args:
            event: The event to process
        """
        try:
            # Skip processing if emergency stop is active (except for emergency stop events)
            if (self._emergency_stop_active and 
                event.event_type not in [EventType.EMERGENCY_STOP, EventType.SIMULATION_RESET]):
                return
                
            # Call registered handlers
            for handler in self._handlers[event.event_type]:
                try:
                    handler.handle_event(event)
                except Exception as e:
                    self.logger.error(f"Handler error for {event.event_type.value}: {e}")
                    
            # Call subscribed callbacks
            for callback in self._callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Callback error for {event.event_type.value}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing event {event.event_type.value}: {e}")
            
    def _handle_emergency_stop(self, event: Event):
        """
        Handle emergency stop events
        
        Args:
            event: The emergency stop event
        """
        self._emergency_stop_active = True
        reason = event.data.get('reason', 'Unknown')
        self.logger.critical(f"EMERGENCY STOP: {reason}")
        
        # Process the emergency stop event immediately
        self._process_event(event)
        
        # Clear the event queue (except for emergency stop and reset events)
        with self._queue_lock:
            critical_events = [e for e in self._event_queue 
                             if e.event_type in [EventType.EMERGENCY_STOP, EventType.SIMULATION_RESET]]
            self._event_queue = critical_events
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get event system statistics"""
        with self._queue_lock:
            queue_size = len(self._event_queue)
            
        return {
            'queue_size': queue_size,
            'processing': self._processing,
            'emergency_stop_active': self._emergency_stop_active,
            'handler_count': sum(len(handlers) for handlers in self._handlers.values()),
            'callback_count': sum(len(callbacks) for callbacks in self._callbacks.values())
        }

# Global event system instance
_global_event_system = None

def get_event_system() -> EventSystem:
    """Get the global event system instance"""
    global _global_event_system
    if _global_event_system is None:
        _global_event_system = EventSystem()
    return _global_event_system 