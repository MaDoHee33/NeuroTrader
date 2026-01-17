import time

class TimeExpirableCache:
    def __init__(self, default_ttl_sec=60):
        self.store = {}
        self.default_ttl = default_ttl_sec

    def set(self, key, value, ttl=None):
        expire_at = time.time() + (ttl or self.default_ttl)
        self.store[key] = (value, expire_at)

    def get(self, key):
        if key not in self.store:
            return None
            
        value, expire_at = self.store[key]
        if time.time() > expire_at:
            del self.store[key]
            return None
            
        return value

    def clear(self):
        self.store.clear()
