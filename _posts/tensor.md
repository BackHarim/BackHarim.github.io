# 텐서소개


```python
import tensorflow as tf
import numpy as np
```

텐서는 일관된 유형을 가진 다차원 배열입니다. 모든 dtypes는 tf.dtypes.DType에서 볼수 있습니다.
Numpy에 더 익숙하다면 , 텐서는 np.arrays와 같습니다.
모든 텐서는 Python 숫자 및 문자열로 변경할 수 없습니다. 텐서의 내용을 업테이트 할 수 없고, 새로운 텐서를 만들수만 있습니다.

## 기초


"스칼라" 또는 "랭크-0" 의 텐서입니다. 스칼라는 단일값을 포함하고 축은 없습니다.


```python
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
```

"벡터" 또는 "랭크-1"의 텐서입니다. 벡터는 하나의 축이 존재합니다.


```python
# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
```

"행렬"또는 "랭크-2"의 텐서는 두개의 축이 존재합니다.


```python
# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

<table>
<tr>
  <th>A scalar, shape: <code>[]</code></th>
  <th>A vector, shape: <code>[3]</code></th>
  <th>A matrix, shape: <code>[3, 2]</code></th>
</tr>
<tr>
  <td>
   <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/scalar.png?raw=1" alt="A scalar, the number 4" />
  </td>

  <td>
   <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/vector.png?raw=1" alt="The line with 3 sections, each one containing a number."/>
  </td>
  <td>
   <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/matrix.png?raw=1" alt="A 3x2 grid, with each cell containing a number.">
  </td>
</tr>
</table>


텐서는 더 많은 축이 존재 할수 있고, 밑은 3개의 축이 있는 텐서입니다.


```python
# There can be an arbitrary number of
# axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)
```

축이 2개 이상인 텐서는 시각화를 할 수 있는 방법이 다양합니다.

<table>
<tr>
  <th colspan=3>A 3-axis tensor, shape: <code>[3, 2, 5]</code></th>
<tr>
<tr>
  <td>
   <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/3-axis_numpy.png?raw=1"/>
  </td>
  <td>
   <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/3-axis_front.png?raw=1"/>
  </td>

  <td>
   <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/3-axis_block.png?raw=1"/>
  </td>
</tr>

</table>

텐서를 np.array , tensor.numpy 메서드를 이용하여서 numpy배열로 변환할수 있습니다.


```python
#
np.array(rank_2_tensor)
```


```python
#
rank_2_tensor.numpy()
```

텐서는 float 및 int를 포함하지만 다음을 비롯한 많은 유형이 존재합니다.
 복소수, 문자열
 기본 tf.Tensor 클래스에서 텐서는 직사각형 입니다. 
 각 축을 따라 모든 요소의 크기가 동일해아합니다. 그러나 다양한 모리를 처리할수 있는 특수한 유형의 텐서가 있습니다.
  -비정형 텐서
  -희소 텐서

덧셈, 요소별 곱셈 및 행렬 곱셈을 포함하여 텐서에 대한 수학을 수행 가능합니다.


```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")
```


```python
print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication
```

텐서는 모든 종류의 연산에 사용됩니다.


```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))
```

## 모양 정보

텐서에는 모양이 있습니다. 
 -Shape : 텐서읭 각 축의 길이
 -Rank: 위에서 나타냇듯이 텐서 축의 수, 스칼라는 0순위, 벡터는 1순위, 행렬은 2순위
 -축 또는 차원 : 텐서의 특정 차원
 -크기: 텐서의 총 항목 수, 제품 모양 벡터

텐서와 tf.TensorShape 객체는 편리한 속성이 있습니다.


```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

<table>
<tr>
  <th colspan=2>A rank-4 tensor, shape: <code>[3, 2, 4, 5]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/shape.png?raw=1" alt="A tensor shape is like a vector.">
    <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/4-axis_block.png?raw=1" alt="A 4-axis tensor">
  </td>
  </tr>
</table>



```python
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())
```

축은 인덱스로 참조되지만 각각 의미를 추적해야 합나디.축은 전역에서 로컬로 정렬됩니다. 배치축이 먼저이고 공간 차원이 다음에 오고, 각 위치에 대한 기능이
마지막에 옵니다. 이런식으로 특징 벡터는 메모리의 연속영역입니다.

<table>
<tr>
<th>Typical axis order</th>
</tr>
<tr>
    <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/shape2.png?raw=1" alt="Keep track of what each axis is. A 4-axis tensor might be: Batch, Width, Height, Features">
  </td>
</tr>
</table>

## 인덱싱

### 단일 축 인덱싱

텐서플로우는 파이썬의 [목록또는 문자열 인덱싱](https://docs.python.org/3/tutorial/introduction.html#strings)과 유사한 표준 파이썬 인덱싱 규칙 및 Numpy 인덱싱의 기본 규칙을 따릅니다.


- 인덱스 시작은 0
- 음수 인덱스는 끝에서 거꾸로 계산
- 콜론 : 은 슬라이스에서 사용. start:stop:step



```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
```

스칼라로 인덱싱을 하면 축이 제거 됩니다.


```python
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
```

: 슬라이스로 인덱싱을 하면 축이 유지 됩니다.


```python
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())
```

### 다축 인덱싱

상위 텐서는 여러 인덱스를 전달하여 인덱싱됩니다.
단일 축의 경우와 똑같은 규칙이 각 축마다 독립적으로 적용됩니다.


```python
print(rank_2_tensor.numpy())
```

각 인덱스에 정수를 전달하게 되면 결과는 스칼라입니다.


```python
# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())
```

정수와 슬라이스를 조합하여 인덱싱을 할 수 있습니다.


```python
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")
```

밑의 내용은 3축 텐서를 사용한 예시입니다.


```python
print(rank_3_tensor[:, :, 4])
```

<table>
<tr>
<th colspan=2>Selecting the last feature across all locations in each example in the batch </th>
</tr>
<tr>
    <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/index1.png?raw=1" alt="A 3x2x5 tensor with all the values at the index-4 of the last axis selected.">
  </td>
      <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/index2.png?raw=1" alt="The selected values packed into a 2-axis tensor.">
  </td>
</tr>
</table>

## 형상 조작하기


```python
# Shape returns a `TensorShape` object that shows the size along each axis
x = tf.constant([[1], [2], [3]])
print(x.shape)
```


```python
# You can convert this object into a Python list, too
print(x.shape.as_list())
```

텐서를 새로운 형태로 변형 가능합니다. tf.reshape 기본 데이터가 중복 될 필요가 없기 때문에 작업을 빠르게 할 수 있습니다.


```python
# You can reshape a tensor to a new shape.
# Note that you're passing in a list
reshaped = tf.reshape(x, [1, 3])
```


```python
print(x.shape)
print(reshaped.shape)
```

데이터는 메모리에서 레이아웃을 유지, 동일한 데이터를 가리키는 요청된 모양으로 새로운 텐서가 나타납니다.
텐서플로우는 C스타일의 "행 중심"메모리 순서를 사용합니다. 가장 오른쪽 인덱스를 증가시키면 메모리의 단일 단계에 해당합니다


```python
print(rank_3_tensor)
```

텐서를 평면화하면 메모리에 어떤 순서로 배치되있는지를 알 수 있습니다.


```python
# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))
```

tf.reshape의 가장 좋은 용도는 인접한 축을 결합하거나 분할하는 것입니다.
3*2*5 텐서의 경우, 슬라이스가 혼합하지 않으므로 (3*2)*5 또는  3*(2*5)으로 재구성하는 것이 합리적입니다.


```python
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
```

<table>
<th colspan=3>
Some good reshapes.
</th>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/reshape-before.png?raw=1" alt="A 3x2x5 tensor">
  </td>
  <td>
  <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/reshape-good1.png?raw=1" alt="The same data reshaped to (3x2)x5">
  </td>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/reshape-good2.png?raw=1" alt="The same data reshaped to 3x(2x5)">
  </td>
</tr>
</table>


형상을 변경하면 총 요소 수를 갖는 새로운 형상에 의해 작동하지만, 축의 순서를 고려하지 않으면 사용가치가 없습니다.
tf.reshape에서 축 교환이 작동하지 않는다면, tf.transpose를 사용하여야합니다.


```python
# Bad examples: don't do this

# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

<table>
<th colspan=3>
Some bad reshapes.
</th>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/reshape-bad.png?raw=1" alt="You can't reorder axes, use tf.transpose for that">
  </td>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/reshape-bad4.png?raw=1" alt="Anything that mixes the slices of data together is probably wrong.">
  </td>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/reshape-bad2.png?raw=1" alt="The new shape must fit exactly.">
  </td>
</tr>
</table>

완전히 지정되지 않은 형상 전체에 걸쳐 실행 가능합니다. None이 포함되거나 전체 형상이 None입니다.

##  DTypes에 대한 추가 정보

tf.Tensor의 데이터 유형을 검사하려면, Tensor.dtype의 속성을 사용
파이썬 객체에서 tf.Tensor를 만들때 선택적으로 데이터 유형을 지정가능합니다.

그렇지 않으면, 텐서플로우는 데이터를 나타낼수 있는 데이터 유형을 선택하게 됩니다.
텐서플로우는 파이선 정수를 tf.int32로 파이썬 부동 소수점을 tf.float32로 변환합니다.
그렇지 텐서플로우는 Numpy가 배열로 변환할때 사용하는 것과 같은 규칙을 사용하게 됩니다.


```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
```

## 브로드캐스팅

브로드캐스팅은 Numpy의 해당특성에서 가져온 개념입니다. 특정 조건에서 작은 텐서는 결합된 연산을 실행할 때 더 큰 텐서에 맞게 자동으로 확장됩니다.
가장 간단하고 일반적인 경우는 스칼라에 텐서를 곱하기 하거나 추가할때 입니다.
스칼라는 다른 인수와 같은 형상으로 브로드캐스트 됩니다.


```python
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

크기가1인 축은 다른 인수와 일치하도록 확장 가능합니다. 두 인수 모두 같은 계산으로 확장 가능합니다.
3*1의 행렬에 요소별로 1*4 행렬을 곱하여 3*4 행렬을 생성합니다. 선행1이 선택사항인 것에 유의하여아합니다. y의 형상은 [4]입니다.


```python
# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

<table>
<tr>
  <th>A broadcasted add: a <code>[3, 1]</code> times a <code>[1, 4]</code> gives a <code>[3,4]</code> </th>
</tr>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/broadcasting.png?raw=1" alt="Adding a 3x1 matrix to a 4x1 matrix results in a 3x4 matrix">
  </td>
</tr>
</table>


브로드캐스팅이 없는 연산입니다.


```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading
```

대부분 브로드캐스팅은 브로드캐스트 연산으로 메모리에서 확장된 텐서를 구체화하지 않아서 시간과 공간에 효율적입니다.
tf.broadcast_to를 이용하여 브로드 캐스팅이 어떤 모습인지 확인 가능합니다.


```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```

예시로, broadcast_to는 수학적인 op와 다르게 메모리를 절약하기 위해 특별한 연산을 수행하지 않습니다. 여기에서 텐서를 구체화합니다.

훨씬 복잡해 줄수 있습니다. 

## tf.convert_to_tensor

tf.matmul , tf.reshape와 같은 대부분의 ops는 클래스 tf.Tensor의 인수를 사용합니다. 하지만 이러한 경우 텐서 형상의 파이썬 객체가 수용되있음을 알 수 있습니다.

대부분의 ops는 텐서가 아닌 인수에 대해 conver_to_tensor 를 호출합니다. 변환 레지스트리가 있어서 Numpy의 ndarray, TensorShape, 파이썬 목록 및 tf.Variable과 
같은 대부분의 객체 클래스는 모두 자동변환됩니다.

## 비정형 텐서

어떤 축을 따라 다양한 수의 요소를 가진 텐서를 "비정형 텐서"라고 합니다. 비정형 데이터에는 tf.ragged.RaggedTensor를 사용합니다.
비정형 텐서는 정규 텐서로 표현 불가 합니다.

<table>
<tr>
  <th>A `tf.RaggedTensor`, shape: <code>[4, None]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/ragged.png?raw=1" alt="A 2-axis ragged tensor, each row can have a different length.">
  </td>
</tr>
</table>


```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
```


```python
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

tf.ragged.constant를 사용하여 tf.RaggedTensor를 작성합니다.


```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```

tf.RaggedTensor의 형상에는 알 수 없는 길이의 일부 축이 포함됩니다.


```python
print(ragged_tensor.shape)
```

## 문자열 텐서

tf.string은 dtype이며 텐서에서 문자열과 같은 데이터를 나타낼 수 있습니다.

문자열은 원자성이므로 파이썬 문자열과 같은 방식으로 인덱싱이 불가능합니다. 문자열의 길이는 텐서 축의 일부가 아닙니다.

밑은 스칼라 문자열 텐서입니다.


```python
# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
```

문자열의 벡터는 다음과 같습니다.

<table>
<tr>
  <th>A vector of strings, shape: <code>[3,]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/strings.png?raw=1" alt="The string length is not one of the tensor's axes.">
  </td>
</tr>
</table>


```python
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
print(tensor_of_strings)
```

위의 출력에서 b 접두사는 tf.string dtype이 유니코드 문자열이 아니라 바이트 문자열인것을 보여줍니다.

유니코드 문자를 전달하면 UTF-8으로 인코딩 됩니다.


```python
tf.constant("🥳👍")
```

문자열이 있는 기본 함수는 tf.strings을 포함하여 tf.strings.split에서 찾을 수 있습니다.


```python
# You can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))
```


```python
# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))
```

<table>
<tr>
  <th>Three strings split, shape: <code>[3, None]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/string-split.png?raw=1" alt="Splitting multiple strings returns a tf.RaggedTensor">
  </td>
</tr>
</table>

And `tf.string.to_number`:


```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

tf.cast를 사용하여 문자열 텐서를 숫자 변환은 할 수 없지만, 바이트로 변환한 다음 숫자로 변환 가능합니다.


```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
```


```python
# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)
```

## 희소 텐서

매우 넓은 임베드 공간과 같이 데이터가 희소합니다. 텐서플로우는 tf.sparse.SparseTensor 및 관련 연산을 지원하여 희소 데이터를 효율적으로 저장합니다.

<table>
<tr>
  <th>A `tf.SparseTensor`, shape: <code>[3, 4]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/tensor/sparse.png?raw=1" alt="An 3x4 grid, with values in only two of the cells.">
  </td>
</tr>
</table>


```python
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))
```
