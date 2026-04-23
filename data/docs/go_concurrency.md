# Go语言并发编程

## Goroutine（协程）

Goroutine是Go语言轻量级的线程，由Go运行时管理。

```go
// 启动一个goroutine
go func() {
    fmt.Println("这是另一个并发执行的函数")
}()

// 启动多个goroutine
for i := 0; i < 10; i++ {
    go func(id int) {
        fmt.Printf("Goroutine %d 执行\n", id)
    }(i)
}
```

## Channel（通道）

Channel是Go语言中用于goroutine之间通信的机制。

### Channel的创建和基本操作

```go
// 创建无缓冲channel
ch := make(chan int)

// 创建带缓冲的channel
ch := make(chan int, 10)

// 发送数据
ch <- 42

// 接收数据
value := <-ch
```

### Channel的3种状态

1. **nil**: 未初始化的channel
2. **active**: 正常的channel，可读可写
3. **closed**: 已关闭的channel

### Channel的3种操作

1. **读操作**: `<-ch`
2. **写操作**: `ch <- value`
3. **关闭操作**: `close(ch)`

### 使用for range读取channel

```go
// 遍历channel，直到channel关闭
for value := range ch {
    fmt.Println(value)
}
```

### 使用ok判断channel是否关闭

```go
// 手动判断channel是否关闭
for {
    value, ok := <-ch
    if !ok {
        fmt.Println("Channel已关闭")
        break
    }
    fmt.Println(value)
}
```

## Select语句

select语句用于处理多个channel的操作。

```go
select {
case msg1 := <-ch1:
    fmt.Println("收到ch1的消息:", msg1)
case msg2 := <-ch2:
    fmt.Println("收到ch2的消息:", msg2)
case <-time.After(time.Second):
    fmt.Println("超时")
default:
    fmt.Println("没有任何消息")
}
```

## 互斥锁（Mutex）

当多个goroutine访问共享资源时，需要使用锁来同步。

```go
import "sync"

var (
    mu    sync.Mutex
    count int
)

// 安全的计数器
func Increment() {
    mu.Lock()
    defer mu.Unlock()
    count++
}

func GetCount() int {
    mu.Lock()
    defer mu.Unlock()
    return count
}
```

## 读写锁（RWMutex）

读写锁可以提高读取密集型程序的性能。

```go
var (
    mu         sync.RWMutex
    data       map[string]string
)

func Read(key string) string {
    mu.RLock()
    defer mu.RUnlock()
    return data[key]
}

func Write(key, value string) {
    mu.Lock()
    defer mu.Unlock()
    data[key] = value
}
```

## WaitGroup（等待组）

WaitGroup用于等待一组goroutine执行完成。

```go
var wg sync.WaitGroup

for i := 0; i < 5; i++ {
    wg.Add(1)  // 增加计数器
    go func(id int) {
        defer wg.Done()  // 执行完成后减少计数器
        fmt.Printf("Goroutine %d 执行完成\n", id)
    }(i)
}

wg.Wait()  // 等待所有goroutine完成
fmt.Println("所有goroutine执行完成")
```

## Once（只执行一次）

sync.Once确保某个函数只执行一次。

```go
var once sync.Once
var instance *Singleton

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}
```

## Context（上下文）

Context用于在goroutine之间传递请求作用域的取消信号。

```go
import "context"

func worker(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("工作被取消")
            return
        default:
            fmt.Println("工作中...")
            time.Sleep(time.Second)
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    go worker(ctx)

    time.Sleep(3 * time.Second)
    cancel()  // 取消所有worker
}
```

## 并发安全的数据结构

Go标准库提供了一些并发安全的数据结构：

- **sync.Map**: 并发安全的Map
- **sync.Pool**: 对象池
- **sync.Cond**: 条件变量

```go
// sync.Map使用示例
var m sync.Map

// 存储键值对
m.Store("name", "Go")

// 读取值
value, ok := m.Load("name")

// 删除键值对
m.Delete("name")

// 遍历
m.Range(func(key, value interface{}) bool {
    fmt.Printf("%s: %s\n", key, value)
    return true
})
```