"""
Advanced performance caching system for Edge TPU v6 benchmarking
Multi-level caching, intelligent invalidation, and performance optimization
"""

import time
import hashlib
import threading
import pickle
import sqlite3
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import weakref
import collections
import json

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache level priorities"""
    L1_MEMORY = "l1_memory"      # Fast in-memory cache
    L2_DISK = "l2_disk"          # Persistent disk cache
    L3_DATABASE = "l3_database"  # SQLite database cache

class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"    # Adaptive policy based on usage patterns

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    hit_rate: float = 0.0
    average_response_time_ms: float = 0.0

class PerformanceCache:
    """
    Multi-level intelligent caching system
    
    Features:
    - Multi-level caching (Memory -> Disk -> Database)
    - Intelligent cache warming and prefetching
    - Adaptive eviction policies
    - Performance monitoring and optimization
    - Thread-safe operations
    """
    
    def __init__(self,
                 max_memory_size: int = 100_000_000,  # 100MB
                 max_disk_size: int = 1_000_000_000,  # 1GB
                 cache_dir: Optional[Path] = None,
                 policy: CachePolicy = CachePolicy.ADAPTIVE):
        """
        Initialize performance cache
        
        Args:
            max_memory_size: Maximum L1 memory cache size in bytes
            max_disk_size: Maximum L2 disk cache size in bytes  
            cache_dir: Directory for persistent cache files
            policy: Cache eviction policy
        """
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.policy = policy
        
        # Initialize cache directories
        self.cache_dir = cache_dir or Path.home() / '.edge_tpu_v6_bench' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # L1 Memory Cache
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_size_bytes = 0
        self.l1_lock = threading.RLock()
        
        # L2 Disk Cache
        self.l2_cache: Dict[str, Path] = {}  # Maps keys to file paths
        self.l2_size_bytes = 0
        self.l2_lock = threading.RLock()
        
        # L3 Database Cache
        self.db_path = self.cache_dir / 'cache.db'
        self.db_lock = threading.RLock()
        self._init_database()
        
        # Cache statistics
        self.stats = {
            CacheLevel.L1_MEMORY: CacheStats(),
            CacheLevel.L2_DISK: CacheStats(),
            CacheLevel.L3_DATABASE: CacheStats()
        }
        
        # LRU order tracking
        self.l1_access_order = collections.OrderedDict()
        self.l2_access_order = collections.OrderedDict()
        
        # Performance monitoring
        self.performance_history: List[Dict[str, float]] = []
        self.warmup_queue: List[str] = []
        
        logger.info(f"PerformanceCache initialized: {policy.value} policy, "
                   f"L1={max_memory_size//1024//1024}MB, L2={max_disk_size//1024//1024}MB")
    
    def _init_database(self):
        """Initialize SQLite database for L3 cache"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    access_count INTEGER,
                    last_access REAL,
                    ttl REAL,
                    size_bytes INTEGER,
                    tags TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tags ON cache_entries(tags)
            ''')
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with intelligent level selection
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        # Try L1 cache first (fastest)
        value = self._get_l1(key)
        if value is not None:
            self._update_stats(CacheLevel.L1_MEMORY, hit=True, response_time=time.time() - start_time)
            self._track_access(key, CacheLevel.L1_MEMORY)
            return value
        
        # Try L2 disk cache
        value = self._get_l2(key)
        if value is not None:
            self._update_stats(CacheLevel.L2_DISK, hit=True, response_time=time.time() - start_time)
            # Promote to L1 if frequently accessed
            if self._should_promote_to_l1(key):
                self._set_l1(key, value)
            self._track_access(key, CacheLevel.L2_DISK)
            return value
        
        # Try L3 database cache
        value = self._get_l3(key)
        if value is not None:
            self._update_stats(CacheLevel.L3_DATABASE, hit=True, response_time=time.time() - start_time)
            # Promote to higher levels based on access patterns
            if self._should_promote_to_l2(key):
                self._set_l2(key, value)
            if self._should_promote_to_l1(key):
                self._set_l1(key, value)
            self._track_access(key, CacheLevel.L3_DATABASE)
            return value
        
        # Cache miss
        for level in CacheLevel:
            self._update_stats(level, hit=False, response_time=time.time() - start_time)
        
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None):
        """
        Set value in cache with intelligent level placement
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for cache invalidation
        """
        if tags is None:
            tags = []
        
        value_size = self._estimate_size(value)
        
        # Always try to store in L1 for fast access
        if value_size <= self.max_memory_size // 10:  # Max 10% of L1 for single entry
            self._set_l1(key, value, ttl=ttl, tags=tags)
        
        # Store in L2 for persistence
        if value_size <= self.max_disk_size // 10:  # Max 10% of L2 for single entry
            self._set_l2(key, value, ttl=ttl, tags=tags)
        
        # Always store in L3 database
        self._set_l3(key, value, ttl=ttl, tags=tags)
        
        logger.debug(f"Cached key '{key}' ({value_size} bytes) with TTL={ttl}")
    
    def _get_l1(self, key: str) -> Any:
        """Get from L1 memory cache"""
        with self.l1_lock:
            entry = self.l1_cache.get(key)
            if entry is None:
                return None
            
            # Check TTL
            if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                self._evict_l1(key)
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Update LRU order
            if key in self.l1_access_order:
                del self.l1_access_order[key]
            self.l1_access_order[key] = entry
            
            return entry.value
    
    def _set_l1(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None):
        """Set in L1 memory cache"""
        with self.l1_lock:
            value_size = self._estimate_size(value)
            
            # Ensure we have space
            while self.l1_size_bytes + value_size > self.max_memory_size and self.l1_cache:
                self._evict_l1_lru()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=value_size,
                tags=tags or []
            )
            
            # Remove old entry if exists
            if key in self.l1_cache:
                self.l1_size_bytes -= self.l1_cache[key].size_bytes
            
            # Add new entry
            self.l1_cache[key] = entry
            self.l1_size_bytes += value_size
            self.l1_access_order[key] = entry
    
    def _get_l2(self, key: str) -> Any:
        """Get from L2 disk cache"""
        with self.l2_lock:
            cache_file = self.l2_cache.get(key)
            if cache_file is None or not cache_file.exists():
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check TTL
                if cached_data.get('ttl') and time.time() - cached_data['timestamp'] > cached_data['ttl']:
                    self._evict_l2(key)
                    return None
                
                # Update access order
                if key in self.l2_access_order:
                    del self.l2_access_order[key]
                self.l2_access_order[key] = time.time()
                
                return cached_data['value']
                
            except Exception as e:
                logger.warning(f"Failed to load from L2 cache: {e}")
                self._evict_l2(key)
                return None
    
    def _set_l2(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None):
        """Set in L2 disk cache"""
        with self.l2_lock:
            try:
                # Create cache file
                cache_file = self.cache_dir / f"l2_{hashlib.sha256(key.encode()).hexdigest()[:16]}.cache"
                
                cached_data = {
                    'key': key,
                    'value': value,
                    'timestamp': time.time(),
                    'ttl': ttl,
                    'tags': tags or []
                }
                
                # Estimate size and ensure space
                data_size = self._estimate_size(cached_data)
                while self.l2_size_bytes + data_size > self.max_disk_size and self.l2_cache:
                    self._evict_l2_lru()
                
                # Write to disk
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Remove old entry if exists
                if key in self.l2_cache:
                    old_file = self.l2_cache[key]
                    if old_file.exists():
                        self.l2_size_bytes -= old_file.stat().st_size
                        old_file.unlink()
                
                # Add new entry
                self.l2_cache[key] = cache_file
                self.l2_size_bytes += cache_file.stat().st_size
                self.l2_access_order[key] = time.time()
                
            except Exception as e:
                logger.error(f"Failed to write to L2 cache: {e}")
    
    def _get_l3(self, key: str) -> Any:
        """Get from L3 database cache"""
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT value, timestamp, ttl, access_count FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    
                    value_blob, timestamp, ttl, access_count = row
                    
                    # Check TTL
                    if ttl and time.time() - timestamp > ttl:
                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                        return None
                    
                    # Update access statistics
                    conn.execute(
                        'UPDATE cache_entries SET access_count = ?, last_access = ? WHERE key = ?',
                        (access_count + 1, time.time(), key)
                    )
                    
                    return pickle.loads(value_blob)
                    
            except Exception as e:
                logger.warning(f"Failed to load from L3 cache: {e}")
                return None
    
    def _set_l3(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None):
        """Set in L3 database cache"""
        with self.db_lock:
            try:
                value_blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                tags_json = json.dumps(tags or [])
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, timestamp, access_count, last_access, ttl, size_bytes, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (key, value_blob, time.time(), 0, time.time(), ttl, len(value_blob), tags_json))
                    
            except Exception as e:
                logger.error(f"Failed to write to L3 cache: {e}")
    
    def _evict_l1_lru(self):
        """Evict least recently used item from L1"""
        if not self.l1_access_order:
            return
        
        key = next(iter(self.l1_access_order))  # Get oldest item
        self._evict_l1(key)
        self.stats[CacheLevel.L1_MEMORY].evictions += 1
    
    def _evict_l2_lru(self):
        """Evict least recently used item from L2"""
        if not self.l2_access_order:
            return
        
        key = min(self.l2_access_order.keys(), key=lambda k: self.l2_access_order[k])
        self._evict_l2(key)
        self.stats[CacheLevel.L2_DISK].evictions += 1
    
    def _evict_l1(self, key: str):
        """Evict specific key from L1"""
        with self.l1_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                self.l1_size_bytes -= entry.size_bytes
                del self.l1_cache[key]
            
            if key in self.l1_access_order:
                del self.l1_access_order[key]
    
    def _evict_l2(self, key: str):
        """Evict specific key from L2"""
        with self.l2_lock:
            if key in self.l2_cache:
                cache_file = self.l2_cache[key]
                if cache_file.exists():
                    self.l2_size_bytes -= cache_file.stat().st_size
                    cache_file.unlink()
                del self.l2_cache[key]
            
            if key in self.l2_access_order:
                del self.l2_access_order[key]
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1"""
        # Simple heuristic: promote if accessed multiple times recently
        return True  # For now, always promote hot items
    
    def _should_promote_to_l2(self, key: str) -> bool:
        """Determine if key should be promoted to L2"""
        return True  # For now, always promote for persistence
    
    def _track_access(self, key: str, level: CacheLevel):
        """Track access patterns for optimization"""
        # Add to warmup queue if frequently accessed
        if key not in self.warmup_queue:
            self.warmup_queue.append(key)
            if len(self.warmup_queue) > 100:  # Keep queue bounded
                self.warmup_queue.pop(0)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(value.items())[:10])  # Sample first 10
            else:
                return 1024  # Default estimate
    
    def _update_stats(self, level: CacheLevel, hit: bool, response_time: float):
        """Update cache statistics"""
        stats = self.stats[level]
        
        if hit:
            stats.hits += 1
        else:
            stats.misses += 1
        
        stats.total_requests += 1
        stats.hit_rate = stats.hits / stats.total_requests if stats.total_requests > 0 else 0.0
        
        # Update average response time
        if stats.total_requests == 1:
            stats.average_response_time_ms = response_time * 1000
        else:
            # Exponential moving average
            alpha = 0.1
            stats.average_response_time_ms = (
                alpha * response_time * 1000 + 
                (1 - alpha) * stats.average_response_time_ms
            )
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with specific tag"""
        # Invalidate L1
        with self.l1_lock:
            keys_to_remove = []
            for key, entry in self.l1_cache.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._evict_l1(key)
        
        # Invalidate L2
        with self.l2_lock:
            keys_to_remove = []
            for key in list(self.l2_cache.keys()):
                cache_file = self.l2_cache[key]
                try:
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        if tag in cached_data.get('tags', []):
                            keys_to_remove.append(key)
                except Exception:
                    keys_to_remove.append(key)  # Remove corrupted entries
            
            for key in keys_to_remove:
                self._evict_l2(key)
        
        # Invalidate L3
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'DELETE FROM cache_entries WHERE tags LIKE ?',
                        (f'%"{tag}"%',)
                    )
            except Exception as e:
                logger.error(f"Failed to invalidate L3 cache by tag: {e}")
        
        logger.info(f"Invalidated cache entries with tag: {tag}")
    
    def warm_cache(self, key_value_pairs: List[Tuple[str, Any]], background: bool = True):
        """Warm cache with key-value pairs"""
        def _warm():
            for key, value in key_value_pairs:
                self.set(key, value)
            logger.info(f"Cache warmed with {len(key_value_pairs)} entries")
        
        if background:
            threading.Thread(target=_warm, daemon=True).start()
        else:
            _warm()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {}
        
        for level in CacheLevel:
            level_stats = self.stats[level]
            stats[level.value] = {
                'hits': level_stats.hits,
                'misses': level_stats.misses,
                'hit_rate': level_stats.hit_rate,
                'total_requests': level_stats.total_requests,
                'evictions': level_stats.evictions,
                'avg_response_time_ms': level_stats.average_response_time_ms
            }
        
        # Overall statistics
        total_hits = sum(s.hits for s in self.stats.values())
        total_requests = sum(s.total_requests for s in self.stats.values())
        
        stats['overall'] = {
            'total_hits': total_hits,
            'total_requests': total_requests,
            'overall_hit_rate': total_hits / total_requests if total_requests > 0 else 0.0,
            'l1_size_mb': self.l1_size_bytes / 1024 / 1024,
            'l2_size_mb': self.l2_size_bytes / 1024 / 1024,
            'l1_entries': len(self.l1_cache),
            'l2_entries': len(self.l2_cache),
        }
        
        return stats
    
    def optimize(self):
        """Optimize cache performance based on access patterns"""
        logger.info("Optimizing cache performance...")
        
        # Analyze access patterns
        stats = self.get_statistics()
        
        # Adjust cache sizes based on hit rates
        l1_hit_rate = stats[CacheLevel.L1_MEMORY.value]['hit_rate']
        l2_hit_rate = stats[CacheLevel.L2_DISK.value]['hit_rate']
        
        # If L1 hit rate is low but we have memory headroom, consider warming more data
        if l1_hit_rate < 0.5 and self.l1_size_bytes < self.max_memory_size * 0.8:
            self._promote_hot_items_to_l1()
        
        # Clean up expired entries
        self._cleanup_expired_entries()
        
        logger.info("Cache optimization completed")
    
    def _promote_hot_items_to_l1(self):
        """Promote frequently accessed items from L2/L3 to L1"""
        # This is a simplified implementation
        # In practice, you'd analyze access frequencies and promote the hottest items
        promoted_count = 0
        
        # Try to promote some items from L2
        with self.l2_lock:
            for key in list(self.l2_access_order.keys())[-10:]:  # Last 10 accessed items
                value = self._get_l2(key)
                if value is not None:
                    self._set_l1(key, value)
                    promoted_count += 1
        
        logger.info(f"Promoted {promoted_count} hot items to L1 cache")
    
    def _cleanup_expired_entries(self):
        """Clean up expired entries from all cache levels"""
        current_time = time.time()
        cleaned_count = 0
        
        # Clean L1
        with self.l1_lock:
            expired_keys = []
            for key, entry in self.l1_cache.items():
                if entry.ttl and current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._evict_l1(key)
                cleaned_count += 1
        
        # Clean L2
        with self.l2_lock:
            expired_keys = []
            for key in list(self.l2_cache.keys()):
                cache_file = self.l2_cache[key]
                try:
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        if cached_data.get('ttl') and current_time - cached_data['timestamp'] > cached_data['ttl']:
                            expired_keys.append(key)
                except Exception:
                    expired_keys.append(key)  # Remove corrupted entries
            
            for key in expired_keys:
                self._evict_l2(key)
                cleaned_count += 1
        
        # Clean L3
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'DELETE FROM cache_entries WHERE ttl IS NOT NULL AND (? - timestamp) > ttl',
                        (current_time,)
                    )
                    cleaned_count += cursor.rowcount
            except Exception as e:
                logger.error(f"Failed to clean L3 cache: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} expired cache entries")
    
    def clear(self, level: Optional[CacheLevel] = None):
        """Clear cache at specific level or all levels"""
        if level is None or level == CacheLevel.L1_MEMORY:
            with self.l1_lock:
                self.l1_cache.clear()
                self.l1_access_order.clear()
                self.l1_size_bytes = 0
        
        if level is None or level == CacheLevel.L2_DISK:
            with self.l2_lock:
                for cache_file in self.l2_cache.values():
                    if cache_file.exists():
                        cache_file.unlink()
                self.l2_cache.clear()
                self.l2_access_order.clear()
                self.l2_size_bytes = 0
        
        if level is None or level == CacheLevel.L3_DATABASE:
            with self.db_lock:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('DELETE FROM cache_entries')
                except Exception as e:
                    logger.error(f"Failed to clear L3 cache: {e}")
        
        level_str = level.value if level else "all levels"
        logger.info(f"Cleared cache: {level_str}")

# Global performance cache instance
global_performance_cache = PerformanceCache()