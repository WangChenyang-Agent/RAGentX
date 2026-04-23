package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/go-redis/redis/v8"
	"time"
)

var (
	rdb     *redis.Client
	ctx     = context.Background()
	cacheTTL = 24 * time.Hour
)

type AskResponse struct {
	Answer  string   `json:"answer"`
	Sources []string `json:"sources"`
	Time    string   `json:"time"`
}

func InitRedis() {
	rdb = redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "",
		DB:       0,
	})

	// 测试连接
	pong, err := rdb.Ping(ctx).Result()
	if err != nil {
		fmt.Printf("Redis connection error: %v\n", err)
	} else {
		fmt.Printf("Redis connected: %s\n", pong)
	}
}

func GetFromCache(query string) (AskResponse, bool) {
	key := fmt.Sprintf("rag:query:%s", query)
	val, err := rdb.Get(ctx, key).Result()
	if err == redis.Nil {
		return AskResponse{}, false
	} else if err != nil {
		fmt.Printf("Redis get error: %v\n", err)
		return AskResponse{}, false
	}

	var resp AskResponse
	if err := json.Unmarshal([]byte(val), &resp); err != nil {
		fmt.Printf("JSON unmarshal error: %v\n", err)
		return AskResponse{}, false
	}

	return resp, true
}

func SetToCache(query string, resp AskResponse) {
	key := fmt.Sprintf("rag:query:%s", query)
	data, err := json.Marshal(resp)
	if err != nil {
		fmt.Printf("JSON marshal error: %v\n", err)
		return
	}

	err = rdb.Set(ctx, key, data, cacheTTL).Err()
	if err != nil {
		fmt.Printf("Redis set error: %v\n", err)
	}
}