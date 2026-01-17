import psutil
import logging
import platform

class ThermalGovernor:
    def __init__(self, max_temp=80.0):
        self.max_temp = max_temp
        self.logger = logging.getLogger("Thermal")
        self.os_type = platform.system()

    def is_safe(self):
        """
        Checks if the system temperature is within safe limits.
        Returns True if safe, False if overheating.
        """
        try:
            # Linux specific temperature check
            if self.os_type == "Linux":
                temps = psutil.sensors_temperatures()
                if not temps:
                    # Some VMs or containers don't expose sensors
                    return True
                
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > self.max_temp:
                            self.logger.critical(f"OVERHEAT DETECTED: {name} at {entry.current}°C")
                            return False
            
            # macOS / Window implementation or fallback
            # Note: psutil often supports Linux best for sensors. 
            # On generic cloud instances, this might return empty, defaulting to safe.
            
            return True
        except Exception as e:
            self.logger.error(f"Thermal check failed: {e}")
            return True # Fail open to avoid blocking if sensors aren't available
