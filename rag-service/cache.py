import redis
import json
import time

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        self.ttl = 24 * 60 * 60  # 24小时
    
    def get(self, key):
        """从缓存获取数据"""
        try:
            cache_key = f"rag:query:{key}"
            data = self.redis_client.get(cache_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key, value):
        """设置缓存数据"""
        try:
            cache_key = f"rag:query:{key}"
            data = json.dumps(value, default=str)
            self.redis_client.setex(cache_key, self.ttl, data)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def delete(self, key):
        """删除缓存数据"""
        try:
            cache_key = f"rag:query:{key}"
            self.redis_client.delete(cache_key)
        except Exception as e:
            print(f"Cache delete error: {e}")
