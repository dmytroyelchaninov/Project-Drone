
def calculate_realistic_efficiency(power_output, power_input):
    """Calculate realistic efficiency with physical bounds"""
    if power_input <= 0:
        return 0.0
    
    raw_efficiency = power_output / power_input
    
    # Physical efficiency bounds
    max_efficiency = 0.95  # 95% maximum for electric systems
    min_efficiency = 0.1   # 10% minimum for meaningful operation
    
    bounded_efficiency = max(min_efficiency, min(raw_efficiency, max_efficiency))
    
    if raw_efficiency > max_efficiency:
        print(f"Warning: Calculated efficiency {raw_efficiency:.3f} exceeds physical limits")
    
    return bounded_efficiency
