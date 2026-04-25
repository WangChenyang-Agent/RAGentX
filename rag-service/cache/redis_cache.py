# -*- coding: utf-8 -*-
import redis
import json
import hashlib
from typing import Optional, Dict, Any

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None

CACHE_TTL = 3600

_redis_client = None


def get_redis_client() -> Optional[redis.Redis]:
    """获取 Redis 客户端（单例模式）"""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5
            )
            _redis_client.ping()
            print(f"Redis connected successfully: {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            print(f"Redis connection failed: {e}")
            _redis_client = None
        except Exception as e:
            print(f"Redis error: {e}")
            _redis_client = None
    return _redis_client


def generate_cache_key(query: str, mode: str = "hybrid", top_k: int = 3) -> str:
    """生成缓存键"""
    key_str = f"{query}:{mode}:{top_k}"
    return f"rag:query:{hashlib.md5(key_str.encode('utf-8')).hexdigest()}"


def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """从缓存获取结果"""
    client = get_redis_client()
    if client is None:
        return None

    try:
        cached = client.get(cache_key)
        if cached:
            print(f"Cache hit: {cache_key}")
            return json.loads(cached)
    except Exception as e:
        print(f"Cache get error: {e}")
    return None


def set_cached_result(cache_key: str, result: Dict[str, Any], ttl: int = CACHE_TTL) -> bool:
    """设置缓存结果"""
    client = get_redis_client()
    if client is None:
        return False

    try:
        client.setex(cache_key, ttl, json.dumps(result, ensure_ascii=False))
        print(f"Cache set: {cache_key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        print(f"Cache set error: {e}")
        return False


def clear_all_cache() -> int:
    """清除所有 RAG 缓存"""
    client = get_redis_client()
    if client is None:
        return 0

    try:
        keys = client.keys("rag:query:*")
        if keys:
            count = client.delete(*keys)
            print(f"Cleared {count} cache keys")
            return count
    except Exception as e:
        print(f"Cache clear error: {e}")
    return 0


def is_redis_available() -> bool:
    """检查 Redis 是否可用"""
    return get_redis_client() is not None
