# A6 CPython C-API 绑定原理

> 本附录对应主�?Ch8。Ch8 �?pybind11 �?C++ TensorImpl/ops 暴露�?Python。本附录深入 CPython C-API 的底层机制，揭示 pybind11 在做什么——它�?C-API �?RAII 封装�?

---

## A6.1 Python 对象的本�?

### A6.1.1 PyObject

Python �?*万物皆对�?*，C 层面每个对象都是一�?`PyObject*`�?

```c
// CPython 定义（简化）
typedef struct _object {
    int ob_refcnt;           // 引用计数
    PyTypeObject *ob_type;   // 类型对象
} PyObject;
```

**所�?Python 对象的前两个字段**�?
- `ob_refcnt`：引用计数（CPython 用引用计数做 GC�?
- `ob_type`：指向类型对象（决定对象的行为）

```c
// int 对象
PyObject *x = PyLong_FromLong(42);
// x->ob_refcnt == 1
// x->ob_type == &PyLong_Type

// list 对象
PyObject *lst = PyList_New(3);
// lst->ob_type == &PyList_Type
```

### A6.1.2 引用计数

**所有权规则**：谁持有 `PyObject*`，谁负责管理引用计数�?

```c
PyObject *x = PyLong_FromLong(42);  // refcnt = 1（你拥有�?

Py_INCREF(x);  // refcnt = 2（你又引用了一次）
// ... �?x ...
Py_DECREF(x);  // refcnt = 1（释放你的第二次引用�?

Py_DECREF(x);  // refcnt = 0 �?析构 �?free(x)
```

**常见 bug**�?
- 忘记 `Py_DECREF` �?内存泄漏
- 多余 `Py_DECREF` �?use-after-free（对象已析构但指针还在用�?

### A6.1.3 借用引用 vs 拥有引用

| 类型 | 规则 | 例子 |
|------|------|------|
| **拥有引用**（New reference�?| 你负�?DECREF | `PyLong_FromLong` 返回�?|
| **借用引用**（Borrowed reference�?| 不用 DECREF | `PyList_GetItem` 返回�?|

```c
PyObject *lst = PyList_New(3);       // 拥有
PyObject *item = PyList_GetItem(lst, 0);  // 借用！不�?DECREF
Py_DECREF(lst);  // �?DECREF lst，不 DECREF item
```

**这是 C-API 最容易出错的地�?*。pybind11 的核心价值就是自动管理引用计数�?

### A6.1.4 引用计数的循环引�?

引用计数无法处理循环引用（A→B, B→A），需要额外的 GC�?

```c
// 循环引用
a = MyClass();  # a.refcnt = 1
b = MyClass();  # b.refcnt = 1
a.ref = b;      # b.refcnt = 2
b.ref = a;      # a.refcnt = 2
del a;          # a.refcnt = 1 (b 还引�?a)
del b;          # b.refcnt = 1 (a 还引�?b)
# �?循环引用，refcnt 永远 > 0，但已无人可�?

// CPython 的解�? 分代 GC
// �?定期扫描容器对象，检测并打破循环引用
// �?PyTypeObject 中设�?Py_TPFLAGS_HAVE_GC 标志参与 GC
```

---

## A6.2 PyTypeObject：类型对�?

### A6.2.1 类型也是一种对�?

```c
// PyTypeObject（简化）
typedef struct {
    PyObject_HEAD
    const char *tp_name;           // 类型�?"minitorch.TensorImpl"
    int tp_basicsize;              // 实例大小
    destructor tp_dealloc;         // 析构函数
    PyCFunction tp_init;           // __init__
    PyObject *(*tp_call)(...);     // __call__
    // ... 几十个函数指�?...
} PyTypeObject;
```

### A6.2.2 定义一�?Python 类型（C-API�?

```c
// 1. 定义 C 结构体（继承 PyObject�?
typedef struct {
    PyObject_HEAD
    TensorImpl *cpp_tensor;  // 持有 C++ 对象
} PyTensor;

// 2. 定义 __init__
static int PyTensor_init(PyTensor *self, PyObject *args, PyObject *kwds) {
    PyObject *data;
    PyArg_ParseTuple(args, "O", &data);  // 解析参数
    // �?data 创建 C++ TensorImpl
    self->cpp_tensor = new TensorImpl(...);
    return 0;
}

// 3. 定义析构
static void PyTensor_dealloc(PyTensor *self) {
    delete self->cpp_tensor;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// 4. 定义方法�?
static PyMethodDef PyTensor_methods[] = {
    {"shape", (PyCFunction)PyTensor_shape, METH_NOARGS, "Get shape"},
    {NULL}
};

// 5. 定义类型对象
static PyTypeObject PyTensorType = {
    .tp_name = "minitorch.TensorImpl",
    .tp_basicsize = sizeof(PyTensor),
    .tp_init = (initproc)PyTensor_init,
    .tp_dealloc = (destructor)PyTensor_dealloc,
    .tp_methods = PyTensor_methods,
};

// 6. 模块初始�?
PyMODINIT_FUNC PyInit__C_ext(void) {
    PyType_Ready(&PyTensorType);  // 注册类型
    PyObject *m = PyModule_Create(&moduledef);
    PyModule_AddObject(m, "TensorImpl", (PyObject*)&PyTensorType);
    return m;
}
```

**这就是用 C-API 定义一�?Python 类的完整代码**——几十行样板代码，手动管理引用计数、类型注册、方法表。pybind11 把这些全部自动化�?

### A6.2.3 tp_getattro vs tp_getattr

```c
// 两种属性访问方�?
static PyTypeObject PyTensorType = {
    .tp_getattr = PyTensor_getattr,    // 旧式: 字符串查找（慢）
    .tp_getattro = PyTensor_getattro,  // 新式: 直接偏移（快�?
};

// 新式 (推荐): �?__dict__ �?descriptor
static PyObject *PyTensor_getattro(PyTensor *self, PyObject *name) {
    // 先查 descriptor (类型上的属�?
    PyObject *descr = _PyType_Lookup(Py_TYPE(self), name);
    if (descr && descr_is_data_descr(descr)) {
        return descr->ob_type->tp_descr_get(descr, self, ...);
    }
    // 再查实例 __dict__
    if (self->dict) {
        PyObject *v = PyDict_GetItem(self->dict, name);
        if (v) { Py_INCREF(v); return v; }
    }
    // 最后报 AttributeError
    return NULL;
}
```

---

## A6.3 PyArg_ParseTuple：参数解�?

### A6.3.1 格式字符�?

```c
static PyObject *my_add(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    // "OO" = 两个 Object
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;  // 解析失败，异常已设置

    // 转换�?C++ 类型
    PyTensor *ta = (PyTensor*)a;
    PyTensor *tb = (PyTensor*)b;

    // 调用 C++
    TensorImpl *result = ops::add(ta->cpp_tensor, tb->cpp_tensor);

    // 包装�?Python
    PyTensor *py_result = PyObject_New(PyTensor, &PyTensorType);
    py_result->cpp_tensor = result;
    return (PyObject*)py_result;
}
```

### A6.3.2 常用格式

| 格式 | C 类型 | Python 类型 |
|------|--------|------------|
| `O` | `PyObject*` | 任意对象 |
| `i` | `int` | int |
| `f` | `float` | float |
| `s` | `char*` | str |
| `O!` | 特定类型 | 检查类�?|
| `|` | 可选参�?| 分隔必�?可�?|
| `O&` | 转换函数 | 自定义转�?|

```c
// 带可选参数和类型检�?
static PyObject *tensor_init(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *data;
    int requires_grad = 0;  // 可选，默认 0
    static char *kwlist[] = {"data", "requires_grad", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", kwlist,
                                     &data, &requires_grad))
        return NULL;
}
```

### A6.3.3 类型检查格�?

```c
// O! 格式: 检查对象类�?
static PyObject *my_fn(PyObject *self, PyObject *args) {
    PyTensor *t;
    if (!PyArg_ParseTuple(args, "O!", &PyTensorType, &t))
        return NULL;
    // �?如果 args[0] 不是 TensorImpl，抛 TypeError
    // �?t 已是 PyTensor*，无需手动 cast
}

// O& 格式: 自定义转�?
static int convert_to_tensor(PyObject *o, TensorImpl **out) {
    // 自定义转换逻辑
    if (PyTensor_Check(o)) {
        *out = ((PyTensor*)o)->cpp_tensor;
        return 1;  // 成功
    }
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return 0;  // 失败
}

static PyObject *my_fn(PyObject *self, PyObject *args) {
    TensorImpl *t;
    if (!PyArg_ParseTuple(args, "O&", convert_to_tensor, &t))
        return NULL;
}
```

---

## A6.4 pybind11：C-API �?RAII 封装

### A6.4.1 同样的功能，pybind11 �?

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_C_ext, m) {
    py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
        .def(py::init<const std::vector<double>&, std::vector<int64_t>, bool>(),
             py::arg("data"), py::arg("shape"),
             py::arg("requires_grad") = false)
        .def_property_readonly("shape", [](const TensorImplPtr& t) {
            return t->shape();
        });

    m.def("add", [](const TensorImplPtr& a, const TensorImplPtr& b) {
        return add(a, b);
    });
}
```

**对比 C-API**：pybind11 �?10 行，C-API �?50+ 行。pybind11 自动处理�?
- 类型注册（`PyType_Ready`�?
- 方法表（`PyMethodDef`�?
- 参数解析（`PyArg_ParseTuple`�?
- 引用计数（`Py_INCREF`/`Py_DECREF`�?
- 异常转换（C++ exception �?Python exception�?
- 类型转换（`std::vector<double>` �?Python list�?

### A6.4.2 pybind11 的引用计数管�?

```cpp
// pybind11 �?py::object �?RAII 包装
{
    py::object x = py::cast(42);  // PyLong_FromLong(42), refcnt=1
    // 使用 x...
}  // 析构 �?Py_DECREF �?refcnt=0 �?free
```

```cpp
// 借用引用�?py::handle（不增减引用计数�?
py::handle borrowed = some_list.attr("key");  // 借用，不 DECREF
```

| pybind11 类型 | 引用语义 | 对应 C-API |
|--------------|---------|-----------|
| `py::object` | 拥有（RAII�?| New reference |
| `py::handle` | 借用 | Borrowed reference |
| `py::str` | 拥有，类型安�?| `PyUnicode_*` |
| `py::list` | 拥有，类型安�?| `PyList_*` |
| `py::tuple` | 拥有，类型安�?| `PyTuple_*` |
| `py::dict` | 拥有，类型安�?| `PyDict_*` |
| `py::array_t<T>` | 拥有，numpy 数组 | `PyArray_*` |

### A6.4.3 类型转换

pybind11 自动�?C++ 类型�?Python 类型之间转换�?

```cpp
// C++ �?Python
py::object x = py::cast(42);              // int �?PyLong
py::object s = py::cast("hello");         // const char* �?PyUnicode
py::object v = py::cast(std::vector<int>{1,2,3});  // vector �?PyList
py::object m = py::cast(std::map<std::string,int>{{"a",1}});  // map �?PyDict

// Python �?C++
int i = x.cast<int>();                    // PyLong �?int
std::string s = py::str("hello");         // PyUnicode �?std::string
auto v = py::list(x).cast<std::vector<int>>();  // PyList �?vector
```

**对比 C-API**�?

```c
// C-API 版：手动转换
PyObject *x = PyLong_FromLong(42);        // int �?PyLong
long i = PyLong_AsLong(x);                // PyLong �?long

// vector �?list
PyObject *lst = PyList_New(v.size());
for (size_t i = 0; i < v.size(); i++)
    PyList_SET_ITEM(lst, i, PyLong_FromLong(v[i]));
```

### A6.4.4 自定义类型转�?

```cpp
// 为自定义 C++ 类型注册 Python 转换
namespace pybind11 { namespace detail {
    template<>
    struct type_caster<MyType> {
    public:
        PYBIND11_TYPE_CASTER(MyType, _("MyType"));

        // Python �?C++
        bool load(handle src, bool convert) {
            if (!src.attr("my_attr").is_none()) {
                value = MyType(src.attr("my_attr").cast<int>());
                return true;
            }
            return false;
        }

        // C++ �?Python
        static handle cast(MyType src, return_value_policy, handle) {
            return py::module_::import("mymodule")
                .attr("MyType")(src.value).release();
        }
    };
}}
```

### A6.4.5 异常转换

```cpp
// pybind11 自动�?C++ 异常转成 Python 异常
m.def("div", [](double a, double b) {
    if (b == 0) throw std::runtime_error("division by zero");
    return a / b;
});

// Python �?
try:
    div(1, 0)
except RuntimeError as e:
    print(e)  # "division by zero"
```

```c
/* C-API 版：手动设置异常 */
static PyObject *my_div(PyObject *self, PyObject *args) {
    double a, b;
    PyArg_ParseTuple(args, "dd", &a, &b);
    if (b == 0) {
        PyErr_SetString(PyExc_RuntimeError, "division by zero");
        return NULL;  // 返回 NULL 表示异常
    }
    return PyFloat_FromDouble(a / b);
}
```

### A6.4.6 注册自定义异�?

```cpp
// pybind11 注册自定义异�?
static py::exception<MyError> my_exc;

PYBIND11_MODULE(_C_ext, m) {
    my_exc = py::exception<MyError>(m, "MyError");
    // �?Python 中可�? from _C_ext import MyError

    m.def("f", []() {
        throw MyError("something went wrong");
        // �?自动转成 MyError 异常
    });
}
```

---

## A6.5 GIL：全局解释器锁

### A6.5.1 什么是 GIL

CPython �?GIL（Global Interpreter Lock）保证同一时刻只有一个线程执�?Python 字节码。C++ 扩展�?Python 交互时必须持�?GIL�?

### A6.5.2 何时需�?GIL

| 场景 | 需�?GIL�?|
|------|-----------|
| 调用 Python C-API（创建对象等�?| **�?* |
| 操作 Python 对象（属性、方法） | **�?* |
| �?C++ 计算（不�?Python�?| **�?* |
| 多线�?C++ 计算 | **�?*（释�?GIL 可并行） |
| 修改 Python 对象�?C++ 字段 | **�?*（Python 可能在读�?|

### A6.5.3 pybind11 �?GIL 管理

```cpp
// pybind11 自动在函数入口获�?GIL
m.def("my_func", [](py::object x) {
    // GIL 已持有（pybind11 自动�?
    return x.attr("method")();  // �?Python 方法，OK
});

// 手动释放 GIL（C++ 多线程计算）
m.def("heavy_compute", [](py::array_t<double> arr) {
    // 1. 拷贝数据�?C++（持�?GIL�?
    auto data = arr.unchecked();

    // 2. 释放 GIL，C++ 多线程计�?
    py::gil_scoped_release release;
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        result[i] = compute(data[i]);

    // 3. 重新获取 GIL（返�?Python�?
    return py::cast(result);
});
```

### A6.5.4 minitorch 中的 GIL

```cpp
// binding/module.cpp 中的 hook 绑定
.def("register_hook", [](TensorImplPtr& t, py::object fn) {
    t->register_hook([fn](TensorImplPtr grad) -> TensorImplPtr {
        py::gil_scoped_acquire acquire;  // �?�?Python callable 需要持�?GIL
        py::object result = fn(grad);
        return result.cast<TensorImplPtr>();
    });
})
```

### A6.5.5 GIL 死锁场景

```cpp
// 反模�? 持有 GIL 时等待另一个线程（那个线程也需�?GIL�?
m.def("deadlock", []() {
    py::object future = start_async_work();  // 启动异步任务
    future.attr("wait")();  // �?持有 GIL 等待，异步任务需�?GIL �?死锁
});

// 修复: 释放 GIL 再等�?
m.def("no_deadlock", []() {
    py::object future = start_async_work();
    py::gil_scoped_release release;
    future_wait(future);  // �?不持�?GIL 等待
    py::gil_scoped_acquire acquire;
    return future.attr("result")();
});
```

---

## A6.6 Buffer Protocol

### A6.6.1 零拷贝数组共�?

Python �?buffer protocol 让不同库共享数组数据而不拷贝�?

```c
// C-API 实现 buffer protocol
static int PyTensor_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
    PyTensor *self = (PyTensor*)obj;
    view->obj = obj;
    view->buf = self->cpp_tensor->data_ptr();  // 指向 C++ 数据
    view->len = self->cpp_tensor->nbytes();
    view->itemsize = sizeof(float);
    view->readonly = 0;
    view->ndim = self->cpp_tensor->ndim();
    view->shape = self->cpp_tensor->shape_ptr();
    view->strides = self->cpp_tensor->strides_ptr();
    Py_INCREF(obj);  // view 持有引用
    return 0;
}

static void PyTensor_releasebuffer(PyObject *obj, Py_buffer *view) {
    Py_DECREF(view->obj);
}

// 注册
static PyTypeObject PyTensorType = {
    .tp_as_buffer = &PyTensor_as_buffer,
    // ...
};
```

### A6.6.2 pybind11 �?buffer protocol

```cpp
// pybind11 �?
py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
    .def_buffer([](TensorImplPtr &t) -> py::buffer_info {
        return py::buffer_info(
            t->data_ptr(),        // 数据指针
            sizeof(float),        // itemsize
            py::format_descriptor<float>::format(),  // 格式
            t->ndim(),            // 维度�?
            t->shape(),           // shape
            t->strides()          // strides
        );
    });

// Python �? 零拷贝转 numpy
import numpy as np
t = minitorch.TensorImpl(...)
arr = np.array(t, copy=False)  # �?共享内存，不拷贝
```

**minitorch �?TensorImpl 支持 buffer protocol**，所以可以直接与 numpy 互操作�?

---

## A6.7 Capsule：传�?C++ 指针

### A6.7.1 PyCapsule

Capsule �?Python �?不透明指针"包装，用于在 Python 间传�?C++ 指针�?

```c
// 创建 capsule
MyClass *obj = new MyClass();
PyObject *cap = PyCapsule_New(obj, "my_module.MyClass", destructor);
// �?Python 持有指针，不关心内部

// 取出指针
void *ptr = PyCapsule_GetPointer(cap, "my_module.MyClass");
MyClass *obj = (MyClass*)ptr;

// 析构函数（Python GC 时调用）
void destructor(PyObject *cap) {
    void *ptr = PyCapsule_GetPointer(cap, "my_module.MyClass");
    delete (MyClass*)ptr;
}
```

### A6.7.2 pybind11 �?capsule

```cpp
// pybind11 �?capsule 传�?C++ 对象
m.def("get_raw_ptr", []() {
    auto *ptr = new MyClass();
    return py::capsule(ptr, [](void *p) { delete (MyClass*)p; });
});

m.def("use_raw_ptr", [](py::capsule cap) {
    auto *ptr = cap.get_pointer<MyClass>();
    ptr->do_something();
});
```

**用�?*�?
- C 扩展间传�?C++ 对象（不经过 Python 类型转换�?
- torch �?`THPVariable_Wrap` 内部�?capsule
- CUDA stream、event 等句柄传�?

---

## A6.8 多阶段模块初始化

### A6.8.1 PEP 489

Python 3.5+ 支持多阶段模块初始化，加速导入：

```c
// 传统: 单阶�?
PyMODINIT_FUNC PyInit__C_ext(void) {
    // 一次性创建所有类型、注册所有方�?
    // �?导入慢（大模块可能几百毫秒）
}

// PEP 489: 多阶�?
static PyModuleDef _C_ext_def = {
    PyModuleDef_HEAD_INIT, "_C_ext", NULL, -1, ...
};

PyMODINIT_FUNC PyInit__C_ext(void) {
    PyObject *m = PyModule_Create(&_C_ext_def);
    return m;
    // �?快速返回，类型延迟创建
}

// 第一次访问类型时才创�?
static PyObject *get_tensor_type() {
    if (!PyTensorType.tp_bases) {
        PyType_Ready(&PyTensorType);  // 延迟初始�?
    }
    return (PyObject*)&PyTensorType;
}
```

### A6.8.2 pybind11 的模块初始化

```cpp
// pybind11 �?PYBIND11_MODULE �?
PYBIND11_MODULE(_C_ext, m) {
    // 所有绑定代�?
    py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl") ...;
    m.def("add", &add);
    // ...
}
// �?pybind11 自动处理多阶段初始化
```

**pybind11 的优�?*�?
- 类型定义延迟到第一次访�?
- 方法注册用高效的 hash table
- 避免导入时的大循�?

---

## A6.9 minitorch �?pybind11 绑定（Ch8 回顾�?

### A6.9.1 结构

```cpp
// binding/module.cpp
PYBIND11_MODULE(_C_ext, m) {
    // Storage �?
    py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage")
        .def(py::init<size_t>())
        .def("__len__", &Storage::size);

    // TensorImpl �?
    py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
        .def(py::init<const std::vector<double>&, std::vector<int64_t>, bool>())
        .def_property_readonly("shape", &TensorImpl::shape)
        .def("backward", &TensorImpl::backward);

    // 算子
    m.def("add", [](const TensorImplPtr& a, const TensorImplPtr& b) {
        return add(a, b);
    });
}
```

### A6.9.2 shared_ptr �?pybind11

```cpp
// pybind11 自动管理 shared_ptr 的引用计�?
py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
//                                  �?
//                    告诉 pybind11 �?shared_ptr<TensorImpl>
```

Python 端持�?`TensorImpl` 对象时，pybind11 持有一�?`shared_ptr<TensorImpl>`。Python 对象析构时，`shared_ptr` 析构，引用计数减 1�?*双重引用计数**（Python refcnt + C++ shared_ptr refcnt），�?pybind11 自动同步，用户无感�?

### A6.9.3 minitorch 绑定的统�?

```
binding/module.cpp �?330 行，绑定:
  - 4 个类 (Storage, TensorImpl, Allocator, Profiler)
  - ~40 个算�?(add, mul, matmul, conv2d, ...)
  - ~10 �?autograd 算子
  - ~5 �?hook/异常/检查点函数

如果�?C-API:
  - 每个�?~60 �?(类型定义 + 方法�?+ 初始�?
  - 每个算子 ~15 �?(参数解析 + 调用 + 返回)
  - 估计 2000+ �?
  �?pybind11 节省 ~85% 代码
```

---

## A6.10 C-API vs pybind11 对比

| 维度 | C-API | pybind11 |
|------|-------|---------|
| 代码�?| 50+ �?�?| 5-10 �?�?|
| 引用计数 | 手动 INCREF/DECREF | 自动（RAII�?|
| 参数解析 | `PyArg_ParseTuple` 格式�?| 自动类型推导 |
| 异常 | `PyErr_SetString` + return NULL | `throw` 自动转换 |
| 类型转换 | 手动 `PyLong_AsLong` �?| 自动 `cast<T>()` |
| 编译依赖 | �?Python.h | pybind11 头文�?|
| 运行时开销 | 最�?| 略高（模�?+ 封装�?|
| 学习曲线 | 陡峭（引用计数陷阱） | 平缓 |
| buffer protocol | 手动实现 | `.def_buffer()` |
| 异常注册 | 手动 | `py::exception<T>` |
| 多线�?| 手动 GIL | `gil_scoped_acquire/release` |

### A6.10.1 为什�?PyTorch 历史上用 C-API

PyTorch 1.0 之前大量使用 C-API（`torch/csrc/`），原因�?
- 当时 pybind11 不够成熟
- 极致性能（C-API 零开销，pybind11 有模板开销�?
- 需要精细控制引用计�?

现代 PyTorch 逐步迁移�?pybind11，新算子绑定�?pybind11，旧代码保留 C-API�?

### A6.10.2 性能对比

```
C-API:
  函数调用开销: ~50 ns (直接 PyCFunction 调用)
  参数解析:     ~100 ns (PyArg_ParseTuple)
  总计:         ~150 ns

pybind11:
  函数调用开销: ~80 ns (模板分发 + 类型检�?
  参数解析:     ~150 ns (自动转换 + 类型检�?
  总计:         ~230 ns

�?pybind11 �?~80 ns/调用
�?对于算子（内部计�?us 级），这 80 ns 可忽�?
�?对于极轻量函数（如获�?shape），C-API 有优�?
```

---

## A6.11 与真�?PyTorch 对照

| 概念 | C-API | pybind11 | PyTorch 文件 |
|------|-------|---------|------------|
| 类型定义 | `PyTypeObject` | `py::class_` | `torch/csrc/autograd/python_variable.cpp` |
| 方法注册 | `PyMethodDef` | `.def()` | 同上 |
| 参数解析 | `PyArg_ParseTuple` | 自动 | - |
| 引用计数 | `Py_INCREF/DECREF` | RAII | - |
| GIL | `PyGILState_Ensure` | `gil_scoped_acquire` | - |
| 模块初始�?| `PyInit_xxx` | `PYBIND11_MODULE` | `torch/csrc/Module.cpp` |
| Buffer | `tp_as_buffer` | `.def_buffer()` | `torch/csrc/autograd/python_variable.cpp` |
| Capsule | `PyCapsule_New` | `py::capsule` | `torch/csrc/autograd/python_variable.cpp` |
| 异常 | `PyErr_SetString` | `throw` | `torch/csrc/autograd/python_variable.cpp` |

---

## A6.12 minitorch 绑定的常见模�?

### A6.12.1 包装 C++ 异常

```cpp
// minitorch 的异常处�?
PYBIND11_MODULE(_C_ext, m) {
    py::register_exception<BadShape>(m, "BadShape");
    py::register_exception<OutOfMemory>(m, "OutOfMemory");

    m.def("matmul", [](const TensorImplPtr& a, const TensorImplPtr& b) {
        if (a->shape().back() != b->shape().front())
            throw BadShape("matmul shape mismatch");
        // ...
    });
}

// Python �?
try:
    _C_ext.matmul(a, b)
except _C_ext.BadShape as e:
    print(e)  # "matmul shape mismatch"
```

### A6.12.2 返回值策�?

```cpp
// pybind11 的返回值策�?
m.def("get_storage", [](TensorImplPtr& t) -> StoragePtr {
    return t->storage();
}, py::return_value_policy::reference_internal);
// �?返回�?Storage �?TensorImpl 共享生命周期

// 策略类型:
// copy:                  返回拷贝
// take_ownership:        转移所有权（C++ �?Python�?
// reference:             借用引用（不�?refcnt，危险）
// reference_internal:    借用但绑定到 self 的生命周�?
```

### A6.12.3 重载方法

```cpp
// pybind11 的重�?
py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
    .def("sum", [](const TensorImplPtr& t) { return sum(t, -1, false); })
    .def("sum", [](const TensorImplPtr& t, int dim) { return sum(t, dim, false); })
    .def("sum", [](const TensorImplPtr& t, int dim, bool keepdim) {
        return sum(t, dim, keepdim);
    });
// �?pybind11 根据参数数量/类型自动分派
```

---

## A6.13 小结

CPython C-API �?Python C 扩展�?*最底层接口**，直接操�?`PyObject*`、手动管理引用计数。功能完整但极易出错�?

pybind11 �?C-API �?**RAII 封装**�?
- `py::object` 自动管理引用计数（构�?INCREF，析�?DECREF�?
- `py::class_` 自动生成 `PyTypeObject` + 方法�?+ 初始化代�?
- 类型转换自动推导（C++ `std::vector` �?Python `list`�?
- 异常自动转换（C++ `throw` �?Python `except`�?
- GIL 自动管理（`gil_scoped_acquire/release`�?
- Buffer protocol 一行注册（`.def_buffer()`�?

minitorch �?pybind11 �?C++ TensorImpl/ops 暴露�?Python，约 330 行绑定代码（`binding/module.cpp`）就完成了所有类、算子、autograd 的绑定。如果用 C-API，同样功能需�?2000+ 行�?

**关键概念**�?
- **PyObject**：万物皆对象，前两个字段�?refcnt �?type
- **引用计数**：拥�?vs 借用，最容易出错的地�?
- **GIL**：Python 线程锁，C++ 多线程时需释放
- **Buffer protocol**：零拷贝数组共享
- **Capsule**：传递不透明 C++ 指针
- **RAII**：pybind11 的核心思想，构造获取、析构释�?
