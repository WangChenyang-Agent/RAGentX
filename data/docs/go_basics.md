# Go语言基础

## 变量声明

Go语言中的变量声明有多种方式：

```go
// 声明单个变量
var name string = "Go语言"

// 声明多个变量
var (
    name string = "Go语言"
    age  int    = 10
)

// 简短声明（函数内部）
message := "Hello, Go!"
```

## 数据类型

Go语言的基本数据类型包括：

- **整型**: int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64
- **浮点型**: float32, float64
- **复数**: complex64, complex128
- **布尔型**: bool
- **字符串**: string

## 数组和切片

```go
// 数组 - 长度固定
var arr [5]int = [5]int{1, 2, 3, 4, 5}

// 切片 - 动态大小
slice := []int{1, 2, 3, 4, 5}
slice = append(slice, 6)
```

## Map（字典）

```go
// 声明Map
m := make(map[string]int)
m["Go"] = 2026
m["Python"] = 1991

// 访问Map
value, ok := m["Go"]
```

## 控制流

### if语句
```go
if x > 10 {
    fmt.Println("x大于10")
} else if x > 5 {
    fmt.Println("x大于5但小于等于10")
} else {
    fmt.Println("x小于等于5")
}
```

### for循环
```go
// 经典for循环
for i := 0; i < 10; i++ {
    fmt.Println(i)
}

// for range遍历切片
nums := []int{1, 2, 3, 4, 5}
for index, value := range nums {
    fmt.Printf("索引: %d, 值: %d\n", index, value)
}
```

## 函数

```go
// 普通函数
func add(a int, b int) int {
    return a + b
}

// 多返回值函数
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("除数不能为零")
    }
    return a / b, nil
}
```

## 结构体

```go
type Person struct {
    Name string
    Age  int
}

// 创建结构体实例
p := Person{
    Name: "张三",
    Age:  25,
}
```

## 接口

```go
type Speaker interface {
    Speak() string
}

type Dog struct{}

func (d Dog) Speak() string {
    return "汪汪汪"
}
```