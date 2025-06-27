
def calculate_realistic_rtf(actual_duration, expected_duration):
    """Calculate realistic real-time factor with bounds checking"""
    if expected_duration <= 0:
        return 1.0  # Default to real-time
    
    raw_rtf = expected_duration / actual_duration
    
    # Apply physical bounds
    min_rtf = 0.01  # 100x slower than real-time
    max_rtf = 100.0  # 100x faster than real-time
    
    bounded_rtf = max(min_rtf, min(raw_rtf, max_rtf))
    
    # Calculate efficiency as fraction of target achieved
    efficiency = min(1.0, bounded_rtf / max(0.1, raw_rtf))
    
    return bounded_rtf, efficiency
