import numpy as np
import mindspore
from mindspore import Tensor, mint, ops, Parameter, tensor
from mindspore import dtype as mstype
from mindspore import numpy as msnp
import unittest
from packaging import version

mindspore.set_context(pynative_synchronize=True)


# Tensor: Creation Operations
class TensorCreationOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Tensor Creation Operations '''

    # mindspore.mint.arange
    def test_arange(self):
        output = mint.arange(1, 6, dtype=mindspore.int32)
        print(output)
        print(output.dtype)
        output = mint.arange(0, 3, 1.2)
        print(output)
        print(output.dtype)
        output = mint.arange(7, 1, -2, dtype=mindspore.int32)
        print(output)
        print(output.dtype)
        output = mint.arange(12, 2, -1, dtype=mindspore.float16)
        print(output)
        print(output.dtype)


    # mindspore.mint.bernoulli
    def test_bernoulli(self):
        input_x = Tensor(np.ones((3, 3)), mindspore.float16)
        output = mint.bernoulli(input_x)
        print(output)
        input_x = Tensor(np.zeros((3, 3)), mindspore.float16)
        output = mint.bernoulli(input_x)
        print(output)


    # mindspore.mint.bincount
    def test_bincount(self):
        print(mint.bincount(Tensor(np.arange(5))))
        print(mint.bincount(Tensor(np.array([0, 1, 1, 3, 2, 1, 7]))))
        w = Tensor(np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6])) # weights
        x = Tensor(np.array([0, 1, 1, 2, 2, 2]))
        print(mint.bincount(x,  weights=w, minlength=5))


    # mindspore.mint.clone
    def test_clone(self):
        input = Tensor(np.ones((3,3)).astype("float16"))
        output = mint.clone(input)
        print(output)


    # mindspore.mint.eye
    def test_eye(self):
        output = mint.eye(2, 2, mindspore.int32)
        print(output)
        print(output.dtype)
        output = mint.eye(1, 2, mindspore.float16)
        print(output)
        print(output.dtype)
        output = mint.eye(2, dtype=mindspore.int32)
        print(output)
        print(output.dtype)
        output = mint.eye(2)
        print(output)
        print(output.dtype)


    # mindspore.mint.einsum
    def test_einsum(self):
        x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        equation = "i->"
        output = mint.einsum(equation, x)
        print(output)
        x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float16)
        equation = "i,i->i"
        output = mint.einsum(equation, x, y)
        print(output)
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float16)
        y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float16)
        equation = "ij,jk->ik"
        output = mint.einsum(equation, x, y)
        print(output)
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float16)
        equation = "ij->ji"
        output = mint.einsum(equation, x)
        print(output)
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float16)
        equation = "ij->j"
        output = mint.einsum(equation, x)
        print(output)
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float16)
        equation = "...->"
        output = mint.einsum(equation, x)
        print(output)
        x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float16)
        y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float16)
        equation = "j,i->ji"
        output = mint.einsum(equation, x, y)
        print(output)
        x = mindspore.Tensor([1, 2, 3, 4], mindspore.float16)
        y = mindspore.Tensor([1, 2], mindspore.float16)
        output = mint.einsum(x, [..., 1], y, [..., 2], [..., 1, 2])
        print(output)


    # mindspore.mint.empty
    def test_empty(self):
        output = mint.empty((2, 3), dtype=mindspore.float16)
        print(output) # accuracy may be not OK on orange_pi


    # mindspore.mint.empty_like
    def test_empty_like(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        output1 = mint.empty_like(x)
        print(output1) # accuracy may be not OK on orange_pi
        output2 = mint.empty_like(x, dtype=mindspore.float16)
        print(output2) # accuracy may be not OK on orange_pi


    # mindspore.mint.full
    def test_full(self):
        output = mint.full((2, 2), 1)
        print(output)
        output = mint.full((3, 3), 0)
        print(output)


    # mindspore.mint.full_like
    def test_full_like(self):
        input = Tensor([[0, 1], [2, 1]], dtype=mindspore.int32)
        output = mint.full_like(input, 1)
        print(output)
        input = Tensor([[0, 1, 1], [2, 1, 2], [1, 3, 4]], dtype=mindspore.int32)
        output = mint.full_like(input, 0, dtype=mindspore.float16)
        print(output)


    # mindspore.mint.linspace
    def test_linspace(self):
        start = 1
        end = 10
        steps = 5
        output = mint.linspace(start, end, steps, dtype=mindspore.float16)
        print(output)


    # mindspore.mint.ones
    def test_ones(self):
        output = mint.ones((2, 2), dtype=mindspore.float16)
        print(output)


    # mindspore.mint.ones_like
    def test_ones_like(self):
        x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        output = mint.ones_like(x)
        print(output)


    # mindspore.mint.randint
    def test_randint(self):
        print(mint.randint(0, 5, (2, 3)).shape)


    # mindspore.mint.randint_like
    def test_randint_like(self):
        a = Tensor([[2, 3, 4], [1, 2, 3]])
        low = 0
        high = 5
        print(mint.randint_like(a, low, high, dtype=mindspore.int32).shape)


    # mindspore.mint.randn
    def test_randn(self):
        print(mint.randn(2, 3).shape)


    # mindspore.mint.randn_like
    def test_randn_like(self):
        a = Tensor([[2, 3, 4], [1, 2, 3]])
        print(mint.randn_like(a, dtype=mindspore.float16).shape)


    # mindspore.mint.randperm
    def test_randperm(self):
        n = 4
        output = mint.randperm(n, dtype=mstype.int32)
        print(output.shape)


    # mindspore.mint.zeros
    def test_zeros(self):
        output = mint.zeros((2, 2), dtype=mindspore.float16)
        print(output)


    # mindspore.mint.zeros_like
    def test_zeros_like(self):
        x = Tensor(np.arange(4).reshape(2, 2))
        output = mint.zeros_like(x, dtype=mindspore.float16)
        print(output)


# Tensor: Indexing, Slicing, Joining, Mutating Operations
class TensorOtherOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Tensor Indexing, Slicing, Joining, Mutating Operations '''

    # mindspore.mint.cat
    def test_cat(self):
        input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float16))
        input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float16))
        output = mint.cat((input_x1, input_x2))
        print(output)
        output = mint.cat((input_x1, input_x2), 1)
        print(output)


    # mindspore.mint.chunk
    def test_chunk(self):
        input_x = np.arange(9).astype("float16")
        output = mindspore.mint.chunk(Tensor(input_x), 3)
        print(output)


    # mindspore.mint.concat
    def test_concat(self):
        input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float16))
        input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float16))
        output = mint.concat((input_x1, input_x2))
        print(output)
        output = mint.concat((input_x1, input_x2), 1)
        print(output)


    # mindspore.mint.count_nonzero
    def test_count_nonzero(self):
        # case 1: each value specified.
        x = Tensor(np.array([[0, 1, 0], [1, 1, 0]]).astype(np.float16))
        nonzero_num = mint.count_nonzero(input=x, dim=[0, 1])
        print(nonzero_num)
        # case 2: all value is default.
        nonzero_num = mint.count_nonzero(input=x)
        print(nonzero_num)
        # case 3: dim value was specified 0.
        nonzero_num = mint.count_nonzero(input=x, dim=[0,])
        print(nonzero_num)
        # case 4: dim value was specified 1.
        nonzero_num = mint.count_nonzero(input=x, dim=[1,])
        print(nonzero_num)


    # mindspore.mint.gather
    def test_gather(self):
        input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float16)
        index = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        output = mint.gather(input, 1, index)
        print(output)


    # mindspore.mint.index_add
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_index_add(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float16)
        index = Tensor(np.array([0, 2]), mindspore.int32)
        y = Tensor(np.array([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]]), mindspore.float16)
        output = mint.index_add(x, 1, index, y, alpha=1)
        print(output)


    # mindspore.mint.index_select
    def test_index_select(self):
        input = Tensor(np.arange(16).astype(np.float16).reshape(2, 2, 4))
        print(input)
        index = Tensor([0,], mindspore.int32)
        y = mint.index_select(input, 1, index)
        print(y)


    # mindspore.mint.masked_select
    def test_masked_select(self):
        x = Tensor(np.array([1, 2, 3, 4]), mindspore.int32)
        mask = Tensor(np.array([1, 0, 1, 0]), mindspore.bool_)
        output = mint.masked_select(x, mask)
        print(output)


    # mindspore.mint.permute
    def test_permute(self):
        input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float16)
        input_perm = (0, 2, 1)
        print(mint.permute(input_x, input_perm))


    # mindspore.mint.reshape
    def test_reshape(self):
        input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float16)
        output = mint.reshape(input, (3, 2))
        print(output)


    # mindspore.mint.scatter
    def test_scatter(self):
        input = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=mindspore.float16)
        src = Tensor(np.array([[8, 8]]), dtype=mindspore.float16)
        index = Tensor(np.array([[2, 4]]), dtype=mindspore.int32)
        out = mint.scatter(input=input, dim=1, index=index, src=src)
        print(out)
        input = Tensor(np.zeros((5, 5)), dtype=mindspore.float16)
        src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=mindspore.float16)
        index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=mindspore.int32)
        out = mint.scatter(input=input, dim=0, index=index, src=src)
        print(out)
        input = Tensor(np.zeros((5, 5)), dtype=mindspore.float16)
        src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=mindspore.float16)
        index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=mindspore.int32)
        out = mint.scatter(input=input, dim=1, index=index, src=src)
        print(out)


    # mindspore.mint.scatter_add
    def test_scatter_add(self):
        input = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=mindspore.float16)
        src = Tensor(np.array([[8, 8]]), dtype=mindspore.float16)
        index = Tensor(np.array([[2, 4]]), dtype=mindspore.int32)
        out = mint.scatter_add(input=input, dim=1, index=index, src=src)
        print(out)
        input = Tensor(np.zeros((5, 5)), dtype=mindspore.float16)
        src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=mindspore.float16)
        index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=mindspore.int32)
        out = mint.scatter_add(input=input, dim=0, index=index, src=src)
        print(out)
        input = Tensor(np.zeros((5, 5)), dtype=mindspore.float16)
        src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=mindspore.float16)
        index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=mindspore.int32)
        out = mint.scatter_add(input=input, dim=1, index=index, src=src)
        print(out)


    # mindspore.mint.split
    def test_split(self):
        input_x = np.arange(9).astype("float16")
        output = mint.split(Tensor(input_x), 3)
        print(output)


    # mindspore.mint.narrow
    def test_narrow(self):
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.int32)
        output = mint.narrow(x, 0, 0, 2)
        print(output)
        output = mint.narrow(x, 1, 1, 2)
        print(output)


    # mindspore.mint.nonzero
    def test_nonzero(self):
        x = Tensor(np.array([[[1,  0], [-5, 0]]]), mindspore.int32)
        output = mint.nonzero(x)
        print(output)
        x = Tensor(np.array([1, 0, 2, 0, 3]), mindspore.int32)
        output = mint.nonzero(x, as_tuple=False)
        print(output)
        x = Tensor(np.array([[[1,  0], [-5, 0]]]), mindspore.int32)
        output = mint.nonzero(x, as_tuple=True)
        print(output)
        x = Tensor(np.array([1, 0, 2, 0, 3]), mindspore.int32)
        output = mint.nonzero(x, as_tuple=True)
        print(output)


    # mindspore.mint.tile
    def test_tile(self):
        input = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float16)
        dims = (2, 3)
        output = mint.tile(input, dims)
        print(output)
        dims = (2, 3, 2)
        output = mint.tile(input, dims)
        print(output)


    # mindspore.mint.tril
    def test_tril(self):
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.tril(x)
        print(result)
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.tril(x, diagonal=1)
        print(result)
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.tril(x, diagonal=-1)
        print(result)


    # mindspore.mint.select
    def test_select(self):
        input = Tensor([[2, 3, 4, 5],[3, 2, 4, 5]])
        y = mint.select(input, 0, 0)
        print(y)


    # mindspore.mint.squeeze
    def test_squeeze(self):
        input = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float16)
        output = mint.squeeze(input, 2)
        print(output)


    # mindspore.mint.stack
    def test_stack(self):
        data1 = Tensor(np.array([0, 1]).astype(np.float16))
        data2 = Tensor(np.array([2, 3]).astype(np.float16))
        output = mint.stack([data1, data2], 0)
        print(output)


    # mindspore.mint.swapaxes
    def test_swapaxes(self):
        input = Tensor(np.ones((2,3,4), dtype=np.float16))
        output = mint.swapaxes(input, 0, 2)
        print(output.shape)


    # mindspore.mint.transpose
    def test_transpose(self):
        input = Tensor(np.ones((2,3,4), dtype=np.float16))
        output = mint.transpose(input, 0, 2)
        print(output.shape)


    # mindspore.mint.triu
    def test_triu(self):
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.triu(x)
        print(result)
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.triu(x, diagonal=1)
        print(result)
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.triu(x, diagonal=-1)
        print(result)


    # mindspore.mint.unbind
    def test_unbind(self):
        input = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        output = mint.unbind(input, dim=0)
        print(output)


    # mindspore.mint.unique_consecutive
    def test_unique_consecutive(self):
        x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]), mstype.int32)
        output, inverse_indices, counts = mint.unique_consecutive(x, True, True, None)
        print(output)
        print(inverse_indices)
        print(counts)


    # mindspore.mint.unsqueeze
    def test_unsqueeze(self):
        input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float16)
        output = mint.unsqueeze(input_tensor, dim=0)
        print(output)


    # mindspore.mint.where
    def test_where(self):
        a = tensor(np.arange(4).reshape((2, 2)), mstype.float16)
        b = tensor(np.ones((2, 2)), mstype.float16)
        condition = a < 3
        output = mint.where(condition, a, b)
        print(output)


# Random Sampling
class RandomSamplingTest(unittest.TestCase):
    ''' mindspore.mint API about Random Sampling '''
    
    # mindspore.mint.multinomial
    def test_multinomial(self):
        # case 1: The output is random, and the length of the output is the same as num_sample.
        # replacement is False.
        input1 = Tensor([90 / 100, 10 / 100, 0], mindspore.float16)
        input2 = Tensor([90, 10, 0], mindspore.float16)
        # input1 and input2 have the same meaning.
        output1 = mint.multinomial(input1, 2)
        output2 = mint.multinomial(input2, 2)
        # print(output1)
        # [0 1]
        # print(output2)
        # [0 1]
        print(len(output1))
        print(len(output2))
        # case 2: The output is random, and the length of the output is the same as num_sample.
        # replacement is True.
        output3 = mint.multinomial(input1, 10, replacement=True)
        # print(output3)
        # [0 0 1 0 0 0 0 0 0 0]
        print(len(output3))
        # case 3: The output is random, and the length of the output is the same as num_sample.
        # replacement is True.
        # rank is 2
        input4 = Tensor([[90, 10, 0], [10, 90, 0]], mstype.float16)
        output4 = mint.multinomial(input4, 10, replacement=True)
        # print(output4)
        # [[0 0 0 0 0 0 0 0 1 0]
        #  [1 1 1 1 1 0 1 1 1 1]]


    # mindspore.mint.normal
    def test_normal(self):
        mean = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float16)
        std = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float16)
        output = mint.normal(mean, std)
        print(output.shape)
        mean = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float16)
        output = mint.normal(mean, 1.0)
        print(output.shape)
        output = mint.normal(1.0, 2.0, (2, 4))
        print(output.shape)


    # mindspore.mint.rand_like
    def test_rand_like(self):
        a = Tensor([[2, 3, 4], [1, 2, 3]])
        print(mint.rand_like(a, dtype=mindspore.float16).shape)


    # mindspore.mint.rand
    def test_rand(self):
        print(mint.rand(2, 3).shape)


# Math Operations: Pointwise Operations
class MathPointwiseOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Math Pointwise Operations '''

    # mindspore.mint.abs
    def test_abs(self):
        input = Tensor(np.array([-1.0, 1.0, 0.0]), mindspore.float16)
        output = mint.abs(input)
        print(output)


    # mindspore.mint.add
    def test_add(self):
        x = Tensor(1, mindspore.int32)
        y = Tensor(np.array([4, 5, 6]).astype(np.float16))
        alpha = 0.5
        output1 = mint.add(x, y)
        print(output1)
        print(output1.dtype)
        output2 = mint.add(x, y, alpha=alpha)
        print(output2)
        # the data type of x is int32, the data type of y is float16,
        # alpha is a float, and the output is the data format of higher precision float16.
        print(output2.dtype)


    # mindspore.mint.addmv
    def test_addmv(self):
        input = Tensor(np.array([2., 3.]).astype(np.float16))
        mat = Tensor(np.array([[2., 5., 3.], [4., 2., 2.]]).astype(np.float16))
        vec = Tensor(np.array([3., 2., 4.]).astype(np.float16))
        output = mint.addmv(input, mat, vec)
        print(output)


    # mindspore.mint.acos
    def test_acos(self):
        input = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float16)
        output = mint.acos(input)
        print(output)


    # mindspore.mint.acosh
    def test_acosh(self):
        input = Tensor(np.array([1.0, 1.5, 3.0, 100.0]), mindspore.float16)
        output = mint.acosh(input)
        print(output)


    # mindspore.mint.arccos
    def test_arccos(self):
        input = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float16)
        output = mint.arccos(input)
        print(output)


    # mindspore.mint.arccosh
    def test_arccosh(self):
        input = Tensor(np.array([1.0, 1.5, 3.0, 100.0]), mindspore.float16)
        output = mint.arccosh(input)
        print(output)


    # mindspore.mint.arcsin
    def test_arcsin(self):
        input = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float16)
        output = mint.arcsin(input)
        print(output)


    # mindspore.mint.arcsinh
    def test_arcsinh(self):
        input = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mindspore.float16)
        output = mint.arcsinh(input)
        print(output)


    # mindspore.mint.arctan
    def test_arctan(self):
        input = Tensor(np.array([1.0, 0.0]), mindspore.float16)
        output = mint.arctan(input)
        print(output)


    # mindspore.mint.arctan2
    def test_arctan2(self):
        x = Tensor(np.array([0, 1]), mindspore.float16)
        y = Tensor(np.array([1, 1]), mindspore.float16)
        output = mint.arctan2(x, y)
        print(output)


    # mindspore.mint.atan2
    def test_atan2(self):
        input = Tensor(np.array([0, 1]), mindspore.float16)
        other = Tensor(np.array([1, 1]), mindspore.float16)
        output = mint.atan2(input, other)
        print(output)


    # mindspore.mint.arctanh
    def test_arctanh(self):
        input = Tensor(np.array([0, -0.5]), mindspore.float16)
        output = mint.arctanh(input)
        print(output)


    # mindspore.mint.asin
    def test_asin(self):
        input = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float16)
        output = mint.asin(input)
        print(output)


    # mindspore.mint.asinh
    def test_asinh(self):
        input = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mindspore.float16)
        output = mint.asinh(input)
        print(output)


    # mindspore.mint.atan
    def test_atan(self):
        input = Tensor(np.array([1.0, 0.0]), mindspore.float16)
        output = mint.atan(input)
        print(output)


    # mindspore.mint.atanh
    def test_atanh(self):
        input = Tensor(np.array([0, -0.5]), mindspore.float16)
        output = mint.atanh(input)
        print(output)


    # mindspore.mint.bitwise_and
    def test_bitwise_and(self):
        input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int32)
        other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int32)
        output = mint.bitwise_and(input, other)
        print(output)


    # mindspore.mint.bitwise_or
    def test_bitwise_or(self):
        input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int32)
        other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int32)
        output = mint.bitwise_or(input, other)
        print(output)


    # mindspore.mint.bitwise_xor
    def test_bitwise_xor(self):
        input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        output = mint.bitwise_xor(input, other)
        print(output)


    # mindspore.mint.ceil
    def test_ceil(self):
        input = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float16)
        output = mint.ceil(input)
        print(output)
        input = Tensor(2.1, mindspore.float16)
        output = mint.ceil(input)
        print(output)


    # mindspore.mint.clamp
    def test_clamp(self):
        # case 1: the data type of input is Tensor
        min_value = Tensor(5, mindspore.float16)
        max_value = Tensor(20, mindspore.float16)
        input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float16)
        output = mint.clamp(input, min_value, max_value)
        print(output)
        # case 2: the data type of input is number
        min_value = 5
        max_value = 20
        input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float16)
        output = mint.clamp(input, min_value, max_value)
        print(output)


    # mindspore.mint.cos
    def test_cos(self):
        input = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float16)
        output = mint.cos(input)
        print(output)


    # mindspore.mint.cosh
    def test_cosh(self):
        x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float16)
        output = mint.cosh(x)
        print(output)
        x = Tensor(2.1, mindspore.float16)
        output = mint.cosh(x)
        print(output)


    # mindspore.mint.cross
    def test_cross(self):
        # case 1: dim=None.
        x = Tensor([[1, 2, 3], [1, 2, 3]])
        other = Tensor([[4, 5, 6], [4, 5, 6]])
        output = mint.cross(x, other)
        print(output)
        # case 2: dim=1.
        x = Tensor([[1, 2, 3], [1, 2, 3]])
        other = Tensor([[4, 5, 6], [4, 5, 6]])
        output = mint.cross(x, other, dim=1)
        print(output)


    # mindspore.mint.diff
    def test_diff(self):
        x = Tensor([1, 3, -1, 0, 4])
        out = mint.diff(x)
        print(out.asnumpy())


    # mindspore.mint.div
    def test_div(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float16)
        y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float16)
        output = mint.div(x, y)
        print(output)


    # mindspore.mint.erf
    def test_erf(self):
        input = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float16)
        output = mint.erf(input)
        print(output)


    # mindspore.mint.erfc
    def test_erfc(self):
        input = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float16)
        output = mint.erfc(input)
        print(output)


    # mindspore.mint.erfinv
    def test_erfinv(self):
        input = Tensor(np.array([0, 0.5, -0.9]), mindspore.float16)
        output = mint.erfinv(input)
        print(output)


    # mindspore.mint.exp
    def test_exp(self):
        input = Tensor(np.array([0.0, 1.0, 3.0]), mindspore.float16)
        output = mint.exp(input)
        print(output)


    # mindspore.mint.exp2
    def test_exp2(self):
        x = Tensor(np.array([0.0, 1.0, 2.0, 4.0]), mindspore.float16)
        output = mint.exp2(x)
        print(output)


    # mindspore.mint.expm1
    def test_expm1(self):
        x = Tensor(np.array([0.0, 1.0, 2.0, 4.0]), mindspore.float16)
        output = mint.expm1(x)
        print(output)


    # mindspore.mint.trunc
    def test_trunc(self):
        x = Tensor(np.array([3.4742, 0.5466, -0.8008, -3.9079]),mindspore.float16)
        output = mint.trunc(x)
        print(output)


    # mindspore.mint.float_power
    def test_float_power(self):
        input = Tensor([1, 2, 3])
        mint.float_power(input, 2)
        
        exp = Tensor([2, -3, -4])
        mint.float_power(input, exp)


    # mindspore.mint.floor
    def test_floor(self):
        input = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float16)
        output = mint.floor(input)
        print(output)


    # mindspore.mint.fmod
    def test_fmod(self):
        input = Tensor(np.array([-4., -3.5, 0, 3.5, 4]), mindspore.float16)
        output = mint.fmod(input, 2.5)
        print(output)


    # mindspore.mint.frac
    def test_frac(self):
        x = Tensor([2, 4.2, -2.5], mindspore.float16)
        output = mint.frac(x)
        print(output)


    # mindspore.mint.lerp
    def test_lerp(self):
        start = Tensor(np.array([1., 2., 3., 4.]), mindspore.float16)
        end = Tensor(np.array([10., 10., 10., 10.]), mindspore.float16)
        output = mint.lerp(start, end, Tensor(0.5, mindspore.float16)) # weight must be float16 on orange_pi
        print(output)


    # mindspore.mint.log
    def test_log(self):
        x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        output = mint.log(x)
        print(output)


    # mindspore.mint.log1p
    def test_log1p(self):
        x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        output = mint.log1p(x)
        print(output)


    # mindspore.mint.log2
    def test_log2(self):
        x = Tensor(np.array([3.0, 5.0, 7.0]), mindspore.float16)
        output = mint.log2(x)
        print(output)


    # mindspore.mint.log10
    def test_log10(self):
        x = Tensor(np.array([3.0, 5.0, 7.0]), mindspore.float16)
        output = mint.log10(x)
        print(output)


    # mindspore.mint.logaddexp
    def test_logaddexp(self):
        x1 = Tensor(np.array([1, 2, 3]).astype(np.float16))
        x2 = Tensor(np.array(2).astype(np.float16))
        output = mint.logaddexp(x1, x2)
        print(output)


    # mindspore.mint.logaddexp2
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_logaddexp2(self):
        x1 = Tensor(np.array([1, 2, 3]).astype(np.float16))
        x2 = Tensor(np.array(2).astype(np.float16))
        output = mint.logaddexp2(x1, x2)
        print(output)


    # mindspore.mint.logical_and
    def test_logical_and(self):
        x = Tensor(np.array([True, False, True]), mindspore.bool_)
        y = Tensor(np.array([True, True, False]), mindspore.bool_)
        output = mint.logical_and(x, y)
        print(output)
        x = Tensor(1, mindspore.bool_)
        y = Tensor(0, mindspore.bool_)
        output = mint.logical_and(x, y)
        print(output)
        x = True
        y = Tensor(0, mindspore.bool_)
        output = mint.logical_and(x, y)
        print(output)
        x = True
        y = Tensor(np.array([True, False]), mindspore.bool_)
        output = mint.logical_and(x, y)
        print(output)


    # mindspore.mint.logical_not
    def test_logical_not(self):
        x = Tensor(np.array([True, False, True]), mindspore.bool_)
        output = mint.logical_not(x)
        print(output)


    # mindspore.mint.logical_or
    def test_logical_or(self):
        x = Tensor(np.array([True, False, True]), mindspore.bool_)
        y = Tensor(np.array([True, True, False]), mindspore.bool_)
        output = mint.logical_or(x, y)
        print(output)
        x = Tensor(1, mindspore.bool_)
        y = Tensor(0, mindspore.bool_)
        output = mint.logical_or(x, y)
        print(output)
        x = True
        y = Tensor(0, mindspore.bool_)
        output = mint.logical_or(x, y)
        print(output)
        x = True
        y = Tensor(np.array([True, False]), mindspore.bool_)
        output = mint.logical_or(x, y)
        print(output)


    # mindspore.mint.logical_xor
    def test_logical_xor(self):
        x = Tensor(np.array([True, False, True]), mindspore.bool_)
        y = Tensor(np.array([True, True, False]), mindspore.bool_)
        output = mint.logical_xor(x, y)
        print(output)
        x = Tensor(1, mindspore.bool_)
        y = Tensor(0, mindspore.bool_)
        output = mint.logical_xor(x, y)
        print(output)
        x = True
        y = Tensor(0, mindspore.bool_)
        output = mint.logical_xor(x, y)
        print(output)
        x = True
        y = Tensor(np.array([True, False]), mindspore.bool_)
        output = mint.logical_xor(x, y)
        print(output)


    # mindspore.mint.mul
    def test_mul(self):
        x = Tensor(np.array([2, 6, 9]).astype(np.int32))
        y = Tensor(np.array([4, 5, 6]).astype(np.float16))
        output = mint.mul(x, y)
        print(output)
        # the data type of x is int32, the data type of y is float16,
        # and the output is the data format of higher precision float16.
        print(output.dtype)


    # mindspore.mint.mv
    def test_mv(self):
        input = Tensor(np.array([[3., 4.], [1., 6.], [1., 3.]]).astype(np.float16))
        vec = Tensor(np.array([1., 2.]).astype(np.float16))
        output = mint.mv(input, vec)
        print(output)


    # mindspore.mint.nansum
    def test_nansum(self):
        x = Tensor(np.array([[float("nan"), 2, 3], [1, 2, float("nan")]]), mindspore.float16)
        output1 = mint.nansum(x, dim=0, keepdim=False, dtype=mindspore.float16)
        output2 = mint.nansum(x, dim=0, keepdim=True, dtype=mindspore.float16)
        print(output1)
        print(output2)


    # mindspore.mint.nan_to_num
    def test_nan_to_num(self):
        input = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 5.0]), mindspore.float16)
        output = mint.nan_to_num(input, 1.0, 2.0, 3.0)
        print(output)


    # mindspore.mint.neg
    def test_neg(self):
        input = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float16)
        output = mint.neg(input)
        print(output)


    # mindspore.mint.pow
    def test_pow(self):
        input = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        exponent = 3.0
        output = mint.pow(input, exponent)
        print(output)
        
        input = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        exponent = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float16)
        output = mint.pow(input, exponent)
        print(output)


    # mindspore.mint.polar float32
    def test_polar_float32(self):
        abs = Tensor(np.array([1, 2]), mindspore.float32)
        angle = Tensor(np.array([np.pi / 2, 5 * np.pi / 4]), mindspore.float32)
        output = mint.polar(abs, angle)
        print(output)


    # mindspore.mint.polar float16
    def test_polar_float16(self):
        abs = Tensor(np.array([1, 2]), mindspore.float16)
        angle = Tensor(np.array([np.pi / 2, 5 * np.pi / 4]), mindspore.float16)
        output = mint.polar(abs, angle) # float32 may be OK
        print(output)


    # mindspore.mint.ravel
    def test_ravel(self):
        x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float16))
        output = mint.ravel(x)
        print(output)
        print(output.shape)


    # mindspore.mint.reciprocal
    def test_reciprocal(self):
        input = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float16)
        output = mint.reciprocal(input)
        print(output)


    # mindspore.mint.remainder
    def test_remainder(self):
        x = Tensor(np.array([-4.0, 5.0, 6.0]).astype(np.float16))
        y = Tensor(np.array([3.0, 2.0, 3.0]).astype(np.float16))
        output = mint.remainder(x, y)
        print(output)


    # mindspore.mint.roll
    def test_roll(self):
        input = Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float16))
        output = mint.roll(input, shifts=2, dims=0)
        print(output)


    # mindspore.mint.round
    def test_round(self):
        input = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), mindspore.float16)
        output = mint.round(input)
        print(output)
        input = Tensor(np.array([0.81, 1.52, 2.35, 2.53, -4.57]), mindspore.float16)
        output = mint.round(input, decimals=1)
        print(output)


    # mindspore.mint.rsqrt
    def test_rsqrt(self):
        input = mindspore.Tensor([-0.0370,  0.2970,  1.5420, -0.9105])
        output = mint.rsqrt(input)
        print(output)


    # mindspore.mint.sigmoid
    def test_sigmoid(self):
        input = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.sigmoid(input)
        print(output)


    # mindspore.mint.sign
    def test_sign(self):
        input = mindspore.Tensor([[-1, 0, 2, 4, 6], [2, 3, 5, -6, 0]])
        output = mint.sign(input)
        print(output)
        x = mindspore.Tensor([[-1, 0, float('inf'), 4, float('nan')], [2, 3, float('-inf'), -6, 0]])
        output = mint.sign(x)
        print(output)


    # mindspore.mint.sin
    def test_sin(self):
        input = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float16)
        output = mint.sin(input)
        print(output)


    # mindspore.mint.sinc
    def test_sinc(self):
        input = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float16)
        output = mint.sinc(input)
        print(output)


    # mindspore.mint.sinh
    def test_sinh(self):
        input = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float16)
        output = mint.sinh(input)
        print(output)


    # mindspore.mint.nn.functional.softmax
    def test_functional_softmax(self):
        input = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.nn.functional.softmax(input)
        print(output)


    # mindspore.mint.sqrt
    def test_sqrt(self):
        input = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float16)
        output = mint.sqrt(input)
        print(output)


    # mindspore.mint.square
    def test_square(self):
        input = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float16)
        output = mint.square(input)
        print(output)


    # mindspore.mint.sub
    def test_sub(self):
        x = Tensor(np.array([4, 5, 6]).astype(np.float16))
        y = Tensor(1, mindspore.int32)
        alpha = 0.5
        output1 = mint.sub(x, y)
        print(output1)
        # the data type of x is float16, the data type of y is int32.
        print(output1.dtype)
        output2 = mint.sub(x, y, alpha=alpha)
        print(output2)
        # the data type of x is float16, the data type of y is int32,
        # alpha is a float, and the output is the data format of higher precision float16.
        print(output2.dtype)


    # mindspore.mint.t
    def test_t(self):
        input = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.float16)
        output = mint.t(input)
        print(output)


    # mindspore.mint.tan
    def test_tan(self):
        input = Tensor(np.array([-1.0, 0.0, 1.0]), mindspore.float16)
        output = mint.tan(input)
        print(output)


    # mindspore.mint.tanh
    def test_tanh(self):
        input = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.tanh(input)
        print(output)


    # mindspore.mint.xlogy
    def test_xlogy(self):
        input = Tensor(np.array([-5, 0, 4]), mindspore.float16)
        other = Tensor(np.array([2, 2, 2]), mindspore.float16)
        output = mint.xlogy(input, other)
        print(output)


# Math Operations: Reduction Operations
class MathReductionOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Math Reduction Operations '''
    
    # mindspore.mint.amax
    def test_amax(self):
        x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float16))
        output = mint.amax(x, 1, keepdim=True)
        result = output.shape
        print(result)
        # case 1: Reduces a dimension by the maximum value of all elements in the dimension.
        x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]),
        mindspore.float16)
        output = mint.amax(x)
        print(output)
        print(output.shape)
        # case 2: Reduces a dimension along axis 0.
        output = mint.amax(x, 0, True)
        print(output)
        # case 3: Reduces a dimension along axis 1.
        output = mint.amax(x, 1, True)
        print(output)
        # case 4: Reduces a dimension along axis 2.
        output = mint.amax(x, 2, True)
        print(output)


    # mindspore.mint.amin
    def test_amin(self):
        x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float16))
        output = mint.amin(x, 1, keepdim=True)
        result = output.shape
        print(result)
        # case 1: Reduces a dimension by the minimum value of all elements in the dimension.
        x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]),
        mindspore.float16)
        output = mint.amin(x)
        print(output)
        print(output.shape)
        # case 2: Reduces a dimension along axis 0.
        output = mint.amin(x, 0, True)
        print(output)
        # case 3: Reduces a dimension along axis 1.
        output = mint.amin(x, 1, True)
        print(output)
        # case 4: Reduces a dimension along axis 2.
        output = mint.amin(x, 2, True)
        print(output)


    # mindspore.mint.argmax
    def test_argmax(self):
        x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float16))
        output = mint.argmax(x)
        print(output)
        x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float16))
        output = mint.argmax(x, dim=-1)
        print(output)


    # mindspore.mint.argmin
    def test_argmin(self):
        x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float16))
        output = mint.argmin(x, dim=-1)
        print(output)


    # mindspore.mint.argsort
    def test_argsort(self):
        x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        sort = mint.argsort(x)
        print(sort)


    # mindspore.mint.all
    def test_all(self):
        x = Tensor(np.array([[True, False], [True, True]]))
        # case 1: Reduces a dimension by the "logicalAND" of all elements in the dimension.
        output = mint.all(x)
        print(output)
        x = Tensor(np.array([[True, False], [True, True]]))
        # case 1: Reduces a dimension along axis 0.
        output = mint.all(x, dim=0)
        print(output)
        # case 2: Reduces a dimension along axis 1.
        output = mint.all(x, dim=1)
        print(output)


    # mindspore.mint.any
    def test_any(self):
        x = Tensor(np.array([[True, False], [True, True]]))
        # case 1: Reduces a dimension by the "logical OR" of all elements in the dimension.
        output = mint.any(x, keepdim=True)
        print(output)
        print(output.shape)
        # case 2: Reduces a dimension along dim 0.
        output = mint.any(x, dim=0)
        print(output)
        # case 3: Reduces a dimension along dim 1.
        output = mint.any(x, dim=1)
        print(output)


    # mindspore.mint.cumprod
    def test_cumprod(self):
        x = Tensor(np.array([1, 2, 3], np.float16))
        output = mint.cumprod(x, 0)
        print(output)


    # mindspore.mint.histc
    def test_histc(self):
        x = Tensor([1., 2, 1])
        y = mint.histc(x, bins=4, min=0, max=3)
        print(y)


    # mindspore.mint.logsumexp
    def test_logsumexp(self):
        x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float16))
        output = mint.logsumexp(x, 1, keepdim=True)
        print(output.shape)


    # mindspore.mint.max
    def test_max(self):
        x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float16)
        output = mint.max(x)
        print(output)
        x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float16)
        output, index = mint.max(x, 0, keepdim=True)
        print(output, index)


    # mindspore.mint.mean
    def test_mean(self):
        x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        mindspore.float16)
        output = mint.mean(x)
        print(output)
        print(output.shape)
        x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        mindspore.float16)
        output = mint.mean(x, 0, True)
        print(output)


    # mindspore.mint.median
    def test_median(self):
        x = Tensor(np.array([[0.57, 0.11, 0.21],[0.38, 0.50, 0.57], [0.36, 0.16, 0.44]]).astype(np.float16))
        y = mint.median(x, dim=0, keepdim=False)
        print(y)


    # mindspore.mint.min
    def test_min(self):
        x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float16)
        output = mint.min(x)
        print(output)
        x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float16)
        output, index = mint.min(x, 0, keepdim=True)
        print(output, index)


    # mindspore.mint.norm
    def test_norm(self):
        data_range = ops.arange(-13, 13, dtype=mindspore.float16)
        x = data_range[data_range != 0]
        y = x.reshape(5, 5)
        print(mint.norm(x, 2.0))


    # mindspore.mint.prod
    def test_prod(self):
        x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                             [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                             [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float16)
        output = mint.prod(x)
        print(output)
        print(output.shape)
        x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                             [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                             [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float16)
        output = mint.prod(x, 0, True)
        print(output)


    # mindspore.mint.sum
    def test_sum(self):
        x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                             [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                             [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float16)
        out = mint.sum(x)
        print(out)
        x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                             [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                             [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float16)
        out = mint.sum(x)
        print(out)
        out = mint.sum(x, dim=2)
        print(out)
        out = mint.sum(x, dim=2, keepdim=True)
        print(out)


    # mindspore.mint.std
    def test_std(self):
        input = Tensor(np.array([[1, 2, 3], [-1, 1, 4]]).astype(np.float16))
        output = mint.std(input, dim=1, correction=1, keepdim=False)
        print(output)


    # mindspore.mint.std_mean
    def test_std_mean(self):
        input = mindspore.Tensor([[1, 2, 3, 4], [-1, 1, 4, -10]], mindspore.float16)
        output_std, output_mean = mindspore.mint.std_mean(input, 1, correction=2, keepdim=True)
        print(output_std)
        print(output_mean)


    # mindspore.mint.unique
    def test_unique(self):
        x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        output = mint.unique(x, return_inverse=True)
        print(output)
        y = output[0]
        print(y)
        idx = output[1]
        print(idx)


    # mindspore.mint.var
    def test_var(self):
        input = Tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], mindspore.float16)
        output = mint.var(input, dim=0, correction=1, keepdim=True)
        print(output)


    # mindspore.mint.var_mean
    def test_var_mean(self):
        input = mindspore.Tensor([[1, 2, 3, 4], [-1, 1, 4, -10]], mindspore.float16)
        output_var, output_mean = mindspore.mint.var_mean(input, 1, correction=2, keepdim=True)
        print(output_var)
        print(output_mean)


# Math Operations: Comparison Operations
class MathComparisonOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Math Comparison Operations '''
    
    # mindspore.mint.allclose
    def test_allclose(self):
        input = Tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        other = Tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        output = mint.allclose(input, other)
        print(output)


    # mindspore.mint.argsort
    def test_argsort(self):
        x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        sort = mint.argsort(x)
        print(sort)


    # mindspore.mint.eq
    def test_eq(self):
        # case 1: The shape of two inputs are different
        x = Tensor([1, 2, 3], mindspore.float16)
        output = mint.eq(x, 2.0)
        print(output)
        # case 2: The shape of two inputs are the same
        x = Tensor([1, 2, 3], mindspore.int32)
        y = Tensor([1, 2, 4], mindspore.int32)
        output = mint.eq(x, y)
        print(output)


    # mindspore.mint.equal
    def test_equal(self):
        x = Tensor([1, 2, 3], mindspore.int32)
        y = Tensor([1, 2, 4], mindspore.int32)
        output = mint.equal(x, y)
        print(output)


    # mindspore.mint.greater
    def test_greater(self):
        input = Tensor(np.array([1, 2, 3]), mindspore.int32)
        other = Tensor(np.array([1, 1, 4]), mindspore.int32)
        output = mint.greater(input, other)
        print(output)


    # mindspore.mint.greater_equal
    def test_greater_equal(self):
        input = Tensor(np.array([1, 2, 3]), mindspore.int32)
        other = Tensor(np.array([1, 1, 4]), mindspore.int32)
        output = mint.greater_equal(input, other)
        print(output)


    # mindspore.mint.gt
    def test_gt(self):
        x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        output = mint.gt(x, y)
        print(output)


    # mindspore.mint.isclose
    def test_isclose(self):
        input = Tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        other = Tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        output = mint.isclose(input, other)
        print(output)


    # mindspore.mint.isfinite
    def test_isfinite(self):
        x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float16)
        output = mint.isfinite(x)
        print(output)
        x = Tensor(2.1, mindspore.float16)
        output = mint.isfinite(x)
        print(output)


    # mindspore.mint.isinf
    def test_isinf(self):
        x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float16)
        output = mint.isinf(x)
        print(output)
        x = Tensor(2.1, mindspore.float16)
        output = mint.isinf(x)
        print(output)


    # mindspore.mint.isneginf
    def test_isneginf(self):
        output = mint.isneginf(Tensor([[-float("inf"), float("inf")], [1, -float("inf")]], mstype.float16))
        print(output)


    # mindspore.mint.le
    def test_le(self):
        x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        output = mint.le(x, y)
        print(output)


    # mindspore.mint.less
    def test_less(self):
        input = Tensor(np.array([1, 2, 3]), mindspore.int32)
        other = Tensor(np.array([1, 1, 4]), mindspore.int32)
        output = mint.less(input, other)
        print(output)


    # mindspore.mint.less_equal
    def test_less_equal(self):
        x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        other = Tensor(np.array([1, 1, 4]), mindspore.int32)
        output = mint.less_equal(x, other)
        print(output)


    # mindspore.mint.maximum
    def test_maximum(self):
        # case 1 : same data type
        input = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float16)
        other = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float16)
        output = mint.maximum(input, other)
        print(output)
        # case 2 : different data type
        input = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.int32)
        other = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float16)
        output = mint.maximum(input, other)
        print(output.dtype)


    # mindspore.mint.minimum
    def test_minimum(self):
        # case 1 : same data type
        input = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float16)
        other = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float16)
        output = mint.minimum(input, other)
        print(output)
        # case 2 : different data type
        input = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.int32)
        other = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float16)
        output = mint.minimum(input, other)
        print(output.dtype)


    # mindspore.mint.ne
    def test_ne(self):
        x = Tensor([1, 2, 3], mindspore.float16)
        output = mint.ne(x, 2.0)
        print(output)
        
        x = Tensor([1, 2, 3], mindspore.int32)
        y = Tensor([1, 2, 4], mindspore.int32)
        output = mint.ne(x, y)
        print(output)


    # mindspore.mint.topk
    def test_topk(self):
        x = mindspore.Tensor([[0.5368, 0.2447, 0.4302, 0.9673],
                       [0.4388, 0.6525, 0.4685, 0.1868],
                       [0.3563, 0.5152, 0.9675, 0.8230]], dtype=mindspore.float16)
        output = mint.topk(x, 2, dim=1)
        print(output)
        output2 = mint.topk(x, 2, dim=1, largest=False)
        print(output2)


    # mindspore.mint.sort
    def test_sort(self):
        x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        output = mint.sort(x)
        # The output below is based on the Ascend platform.
        print(output)


# Math Operations: BLAS and LAPACK Operations
class MathBlasAndLapackOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Math BLAS and LAPACK Operations '''
    
    # mindspore.mint.addbmm
    def test_addbmm(self):
        m = np.ones((3, 3)).astype(np.float16)
        arr1 = np.arange(24).astype(np.float16).reshape((2, 3, 4))
        arr2 = np.arange(24).astype(np.float16).reshape((2, 4, 3))
        a = Tensor(arr1)
        b = Tensor(arr2)
        c = Tensor(m)
        output = mint.addbmm(c, a, b)
        print(output)


    # mindspore.mint.addmm
    def test_addmm(self):
        input = Tensor(np.ones([3, 3]).astype(np.float16))
        mat1 = Tensor(np.ones([3, 4]).astype(np.float16))
        mat2 = Tensor(np.ones([4, 3]).astype(np.float16))
        output =  mint.addmm(input, mat1, mat2)
        print(output)


    # mindspore.mint.baddbmm
    def test_baddbmm(self):
        input = Tensor(np.ones([1, 3, 3]).astype(np.float16))
        batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float16))
        batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float16))
        output = mint.baddbmm(input, batch1, batch2)
        print(output)


    # mindspore.mint.bmm
    def test_bmm(self):
        a = Tensor(np.ones(shape=[2, 3, 4]), mindspore.float16)
        b = Tensor(np.ones(shape=[2, 4, 5]), mindspore.float16)
        output = mint.bmm(a, b)
        print(output)


    # mindspore.mint.dot
    def test_dot(self):
        x = Tensor([2.0, 3.0], mindspore.float16)
        y = Tensor([2.0, 1.0], mindspore.float16)
        # dot = mint.dot()
        output = mint.dot(x, y)
        print(output)
        print(output.dtype)


    # mindspore.mint.inverse float16
    def test_inverse_float16(self):
        x = Tensor([[1., 2.], [3., 4.]], mstype.float16)
        print(mint.inverse(x)) # float32 may be OK


    # mindspore.mint.inverse float32
    def test_inverse_float32(self):
        x = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        print(mint.inverse(x))


    # mindspore.mint.matmul
    def test_matmul(self):
        # case 1 : Reasonable application of broadcast mechanism
        input = Tensor(np.arange(2*3*4).reshape(2, 3, 4), mindspore.float16)
        other = Tensor(np.arange(4*5).reshape(4, 5), mindspore.float16)
        output = mint.matmul(input, other)
        print(output)
        print(output.shape)
        # case 2 : the rank of `input` is 1
        input = Tensor(np.ones([1, 2]), mindspore.float16)
        other = Tensor(np.ones([2,]), mindspore.float16)
        output = mint.matmul(input, other)
        print(output)
        print(output.shape)


    # mindspore.mint.meshgrid
    def test_meshgrid(self):
        x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
        y = Tensor(np.array([5, 6, 7]).astype(np.int32))
        z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
        output = mint.meshgrid(x, y, z, indexing='xy')
        print(output)


    # mindspore.mint.mm
    def test_mm(self):
        x1 = mindspore.Tensor(np.random.rand(2, 3), mindspore.float16)
        x2 = mindspore.Tensor(np.random.rand(3, 4), mindspore.float16)
        out = mint.mm(x1, x2)
        print(out.shape)


    # mindspore.mint.outer
    def test_outer(self):
        input = Tensor(np.array([7, 8, 9]), mindspore.int32)
        vec2 = Tensor(np.array([7, 10, 11]), mindspore.int32)
        out = mint.outer(input, vec2)
        print(out)


    # mindspore.mint.trace
    def test_trace(self):
        input = Tensor(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]), mindspore.float32)
        output = mint.trace(input) # float16 may be core dumped
        print(output)
        input = Tensor(np.arange(1, 13).reshape(3, 4), mindspore.float32)
        output = mint.trace(input)
        print(output)
        input = Tensor(np.arange(12, 0, -1).reshape(4, 3), mindspore.float32)
        output = mint.trace(input)
        print(output)


# Math Operations: Other Operations
class MathOtherOperationsTest(unittest.TestCase):
    ''' mindspore.mint API about Math Other Operations '''

    # mindspore.mint.broadcast_to
    def test_broadcast_to(self):
        shape = (2, 3)
        x = Tensor(np.array([1, 2, 3]).astype(np.float16))
        output = mint.broadcast_to(x, shape)
        print(output)
        shape = (-1, 2)
        x = Tensor(np.array([[1], [2]]).astype(np.float16))
        output = mint.broadcast_to(x, shape)
        print(output)


    # mindspore.mint.cdist
    def test_cdist(self):
        x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float16))
        y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float16))
        output = mint.cdist(x, y, 2.0)
        print(output)


    # mindspore.mint.cummax
    def test_cummax(self):
        x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float16))
        output = mint.cummax(x, dim=0)
        print(output[0])
        print(output[1])


    # mindspore.mint.cummin
    def test_cummin(self):
        a = Tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220], mindspore.float16)
        output = mint.cummin(a, dim=0)
        print(output[0])
        print(output[1])


    # mindspore.mint.cumsum
    def test_cumsum(self):
        x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float16))
        # case 1: along the dim 0
        y = mint.cumsum(x, 0)
        print(y)
        # case 2: along the dim 1
        y = mint.cumsum(x, 1)
        print(y)


    # mindspore.mint.diag
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_diag(self):
        input = Tensor([1, 2, 3, 4]).astype('int32')
        output = mint.diag(input)
        print(output)


    # mindspore.mint.flatten
    def test_flatten(self):
        input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float16)
        output = mint.flatten(input_x)
        print(output.shape)


    # mindspore.mint.flip
    def test_flip(self):
        input = mindspore.Tensor(np.arange(1, 9).reshape((2, 2, 2)))
        output = mint.flip(input, (0, 2))
        print(output)


    # mindspore.mint.repeat_interleave
    def test_repeat_interleave(self):
        input = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        output = mint.repeat_interleave(input, repeats=2, dim=0)
        print(output)


    # mindspore.numpy.repeat
    def test_numpy_repeat(self):
        output = msnp.repeat(msnp.array(3), 4)
        print(output)
        x = msnp.array([[1,2],[3,4]])
        output = msnp.repeat(x, 2)
        print(output)
        output = msnp.repeat(x, 3, axis=1)
        print(output)
        output = msnp.repeat(x, [1, 2], axis=0)
        print(output)


    # mindspore.mint.searchsorted
    def test_searchsorted(self):
        sorted_sequence = Tensor(np.array([[0, 1, 3, 5, 7], [2, 4, 6, 8, 10]]), mindspore.float16)
        values = Tensor(np.array([[3, 6, 9], [3, 6, 9]]), mindspore.float16)
        output = mint.searchsorted(sorted_sequence, values)
        print(output)


    # mindspore.mint.tril
    def test_tril(self):
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.tril(x)
        print(result)
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.tril(x, diagonal=1)
        print(result)
        x = Tensor(np.array([[ 1,  2,  3,  4],
                             [ 5,  6,  7,  8],
                             [10, 11, 12, 13],
                             [14, 15, 16, 17]]))
        result = mint.tril(x, diagonal=-1)
        print(result)


    # mindspore.mint.triangular_solve float16
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_triangular_solve_float16(self):
        b = Tensor(np.ones((2, 3, 4), dtype=np.float16))
        A = Tensor(np.ones((2, 3, 3), dtype=np.float16))
        output = mint.triangular_solve(b, A)
        print(output[0])


    # mindspore.mint.triangular_solve float32
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_triangular_solve_float32(self):
        b = Tensor(np.ones((2, 3, 4), dtype=np.float32))
        A = Tensor(np.ones((2, 3, 3), dtype=np.float32))
        output = mint.triangular_solve(b, A)
        print(output[0])


# mindspore.mint.nn.functional: Convolution functions
class FunctionalConvolutionTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Convolution functions '''
    
    # mindspore.mint.nn.functional.conv2d
    def test_functional_conv2d(self):
        x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float16)
        weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float16)
        output = mint.nn.functional.conv2d(x, weight)
        print(output.shape)


    # mindspore.mint.nn.functional.conv3d
    def test_functional_conv3d(self):
        x = mindspore.Tensor(np.random.randn(12, 1, 60, 50, 8), mindspore.float16)
        w = mindspore.Tensor(np.random.randn(26, 1, 2, 4, 4), mindspore.float16)
        out = mint.nn.functional.conv3d(x, w)
        print(out.shape)


    # mindspore.mint.nn.functional.conv_transpose2d
    def test_functional_conv_transpose2d(self):
        x = Tensor(np.ones([1, 4, 5, 5]), mindspore.float16)
        weight = Tensor(np.ones([4, 8, 3, 3]), mindspore.float16)
        output = mint.nn.functional.conv_transpose2d(x, weight)
        print(output.shape)


    # mindspore.mint.nn.functional.fold
    def test_functional_fold(self):
        x = Tensor(np.random.rand(16, 64, 25).astype(np.float16))
        output = mint.nn.functional.fold(x, (8, 8), [2, 2], [2, 2], [2, 2], [2, 2])
        print(output.shape)


    # mindspore.mint.nn.functional.unfold
    def test_functional_unfold(self):
        x = Tensor(np.random.rand(4, 4, 32, 32), mindspore.float16)
        output = mint.nn.functional.unfold(x, kernel_size=3, dilation=1, stride=1)
        print(output.shape)


# mindspore.mint.nn.functional: Pooling functions
class FunctionalPoolingTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Pooling functions '''
    
    # mindspore.mint.nn.functional.adaptive_avg_pool1d
    def test_functional_adaptive_avg_pool1d(self):
        input = Tensor([[2,3],[3,4]],dtype=mindspore.float16)
        output = mint.nn.functional.adaptive_avg_pool1d(input, 3)
        print(output)


    # mindspore.mint.nn.functional.adaptive_avg_pool2d
    def test_functional_adaptive_avg_pool2d(self):
        # case 1: output_size=(3, 2)
        input = Tensor(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), mindspore.float16)
        output = mint.nn.functional.adaptive_avg_pool2d(input, (3, 2))
        print(output)


    # mindspore.mint.nn.functional.adaptive_avg_pool3d
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_adaptive_avg_pool3d(self):
        # case 1: output_size=(3, 3, 4)
        output_size=(3, 3, 4)
        input_val = np.random.randn(4, 3, 5, 6, 7)
        input = Tensor(input_val, mindspore.float16)
        output = mint.nn.functional.adaptive_avg_pool3d(input, output_size)
        print(output.shape)

        # case 2: output_size=4
        output_size=5
        input_val = np.random.randn(2, 3, 8, 6, 12)
        input = Tensor(input_val, mindspore.float16)
        output = mint.nn.functional.adaptive_avg_pool3d(input, output_size)
        print(output.shape)

        # case 3: output_size=(None, 4, 5)
        output_size=(None, 4, 5)
        input_val = np.random.randn(4, 1, 9, 10, 8)
        input = Tensor(input_val, mindspore.float16)
        output = mint.nn.functional.adaptive_avg_pool3d(input, output_size)
        print(output.shape)


    # mindspore.mint.nn.functional.adaptive_max_pool1d
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_adaptive_max_pool1d(self):
        input = Tensor([[2,3],[3,4]],dtype=mindspore.float16)
        output = mint.nn.functional.adaptive_max_pool1d(input, 3)
        print(output)


    # mindspore.mint.nn.functional.avg_pool1d
    def test_functional_avg_pool1d(self):
        input_x = Tensor(np.random.randint(0, 10, [1, 3, 6]), mindspore.float16)
        output = mint.nn.functional.avg_pool1d(input_x, kernel_size=6, stride=1)
        print(output.shape)


    # mindspore.mint.nn.functional.avg_pool2d
    def test_functional_avg_pool2d(self):
        x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), mindspore.float16)
        output = mint.nn.functional.avg_pool2d(x, kernel_size=2, stride=1)
        print(output)


    # mindspore.mint.nn.functional.avg_pool3d
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_avg_pool3d(self):
        input_x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)), mindspore.float16)
        output = mint.nn.functional.avg_pool3d(input_x, kernel_size=2, stride=1)
        print(output)


    # mindspore.mint.nn.functional.max_pool2d
    def test_functional_max_pool2d(self):
        input = Tensor(np.arange(20 * 16 * 50 * 32).reshape((20, 16, 50, 32)), mindspore.float16)
        output_tensor, argmax = mint.nn.functional.max_pool2d(input, kernel_size=(3, 2), stride=(2, 1),
                                                                    ceil_mode=False, return_indices=True)
        print(output_tensor.shape)
        print(argmax.shape)


    # mindspore.mint.nn.functional.max_unpool2d
    def test_functional_max_unpool2d(self):
        input = Tensor(np.array([[[[0, 1], [8, 9]]]]).astype(np.float16))
        indices = Tensor(np.array([[[[0, 1], [2, 3]]]]).astype(np.int32))
        output = mint.nn.functional.max_unpool2d(input, indices, 1, stride=1, padding=0)
        print(output.asnumpy())


# mindspore.mint.nn.functional: Non-linear activation functions
class FunctionalNonLinearActivationTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Non-linear activation functions '''
    
    # mindspore.mint.nn.functional.batch_norm
    def test_functional_batch_norm(self):
        input_x = Tensor([[1.0, 2.0], [3.0, 4.0]], mindspore.float16)
        running_mean = Tensor([0.5, 1.5], mindspore.float16)
        running_var = Tensor([0.1, 0.2], mindspore.float16)
        weight = Tensor([2.0, 2.0], mindspore.float16)
        bias = Tensor([-1.0, -1.0], mindspore.float16)
        output = mint.nn.functional.batch_norm(input_x, running_mean, running_var, weight, bias)
        print(output)


    # mindspore.mint.nn.functional.elu
    def test_functional_elu(self):
        x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        output = mint.nn.functional.elu(x)
        print(output)


    # mindspore.mint.nn.functional.elu_
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_elu_(self):
        input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        mint.nn.functional.elu_(input)
        print(input)


    # mindspore.mint.nn.functional.gelu
    def test_functional_gelu(self):
        x = Tensor([1.0, 2.0, 3.0], mindspore.float16)
        result = mint.nn.functional.gelu(x, approximate='none')
        print(result)
        result = mint.nn.functional.gelu(x, approximate="tanh")
        print(result)


    # mindspore.mint.nn.functional.glu
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_glu(self):
        input = Tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        output = mint.nn.functional.glu(input)
        print(output)


    # mindspore.mint.nn.functional.group_norm
    def test_functional_group_norm(self):
        x = mindspore.Tensor(np.ones([1, 2, 4, 4], np.float16))
        output = mint.nn.functional.group_norm(x, 2)
        print(output)


    # mindspore.mint.nn.functional.hardshrink
    def test_functional_hardshrink(self):
        input = Tensor(np.array([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]), mindspore.float16)
        output = mint.nn.functional.hardshrink(input)
        print(output)


    # mindspore.mint.nn.functional.hardsigmoid
    def test_functional_hardsigmoid(self):
        input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        output = mint.nn.functional.hardsigmoid(input)
        print(output)


    # mindspore.mint.nn.functional.hardswish
    def test_functional_hardswish(self):
        input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        output = mint.nn.functional.hardswish(input)
        print(output)


    # mindspore.mint.nn.functional.layer_norm
    def test_functional_layer_norm(self):
        input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float16)
        normalized_shape = (3,)
        gamma = Tensor(np.ones(normalized_shape), mindspore.float16)
        beta = Tensor(np.zeros(normalized_shape), mindspore.float16)
        eps = 1e-7
        output = mint.nn.functional.layer_norm(input_x, normalized_shape, gamma, beta, eps)
        print(output)


    # mindspore.mint.nn.functional.leaky_relu
    def test_functional_leaky_relu(self):
        input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        print(mint.nn.functional.leaky_relu(input, negative_slope=0.2))


    # mindspore.mint.nn.functional.log_softmax
    def test_functional_log_softmax(self):
        logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.nn.functional.log_softmax(logits, dim=-1)
        print(output)


    # mindspore.mint.nn.functional.logsigmoid
    def test_functional_logsigmoid(self):
        input = Tensor([1.0, 2.0, 3.0], mindspore.float16)
        output = mint.nn.functional.logsigmoid(input)
        print(output)


    # mindspore.mint.nn.functional.mish
    def test_functional_mish(self):
        x = Tensor(np.array([[-1.1, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        output = mint.nn.functional.mish(x)
        print(output)


    # mindspore.mint.nn.functional.prelu
    def test_functional_prelu(self):
        x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)), mindspore.float16)
        weight = Tensor(np.array([0.1, 0.6, -0.3]), mindspore.float16)
        output = mint.nn.functional.prelu(x, weight)
        print(output)


    # mindspore.mint.nn.functional.relu
    def test_functional_relu(self):
        input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        output = mint.nn.functional.relu(input)
        print(output)


    # mindspore.mint.nn.functional.relu6
    def test_functional_relu6(self):
        x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        result = mint.nn.functional.relu6(x)
        print(result)


    # mindspore.mint.nn.functional.relu_
    def test_functional_relu_(self):
        input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        mint.nn.functional.relu_(input)
        print(input)


    # mindspore.mint.nn.functional.selu
    def test_functional_selu(self):
        input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float16)
        output = mint.nn.functional.selu(input)
        print(output)


    # mindspore.mint.nn.functional.sigmoid
    def test_functional_sigmoid(self):
        input = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.nn.functional.sigmoid(input)
        print(output)


    # mindspore.mint.nn.functional.silu
    def test_functional_silu(self):
        input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        output = mint.nn.functional.silu(input)
        print(output)


    # mindspore.mint.nn.functional.softmax
    def test_functional_softmax(self):
        input = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.nn.functional.softmax(input)
        print(output)


    # mindspore.mint.nn.functional.softplus
    def test_functional_softplus(self):
        input = Tensor(np.array([0.1, 0.2, 30, 25]), mindspore.float16)
        output = mint.nn.functional.softplus(input)
        print(output)


    # mindspore.mint.nn.functional.softshrink
    def test_functional_softshrink(self):
        x = Tensor(np.array([[ 0.5297,  0.7871,  1.1754], [ 0.7836,  0.6218, -1.1542]]), mindspore.float16)
        output = mint.nn.functional.softshrink(x)
        print(output)


    # mindspore.mint.nn.functional.tanh
    def test_functional_tanh(self):
        input = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float16)
        output = mint.nn.functional.tanh(input)
        print(output)


# mindspore.mint.nn.functional: Normalization functions
class FunctionalNormalizationTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Normalization functions '''
    
    # mindspore.mint.nn.functional.normalize
    def test_functional_normalize(self):
        tensor = Tensor(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), mindspore.float16)
        output = mint.nn.functional.normalize(tensor)
        print(output)


# mindspore.mint.nn.functional: Linear functions
class FunctionalLinearTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Linear functions '''
    
    # mindspore.mint.nn.functional.linear
    def test_functional_linear(self):
        input = Tensor([[-1., 1., 2.], [-3., -3., 1.]], mindspore.float16)
        weight = Tensor([[-2., -2., -2.], [0., -1., 0.]], mindspore.float16)
        bias = Tensor([0., 1.], mindspore.float16)
        output = mint.nn.functional.linear(input, weight, bias)
        print(output)


# mindspore.mint.nn.functional: Dropout functions
class FunctionalDropoutTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Dropout functions '''
    
    # mindspore.mint.nn.functional.dropout
    def test_functional_dropout(self):
        input = Tensor(((20, 16), (50, 50)), mindspore.float16)
        output = mint.nn.functional.dropout(input, p=0.5)
        print(output.shape)


    # mindspore.mint.nn.functional.dropout2d
    def test_functional_dropout2d(self):
        input = Tensor(np.ones([2, 1, 2, 3]), mindspore.float16)
        output = mint.nn.functional.dropout2d(input, 0.5)
        print(output.shape)


# mindspore.mint.nn.functional: Sparse functions
class FunctionalSparseTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Sparse functions '''
    
    # mindspore.mint.nn.functional.embedding
    def test_functional_embedding(self):
        input = Tensor([[1, 0, 1, 1], [0, 0, 1, 0]])
        weight = Parameter(np.random.randn(3, 3).astype(np.float16))
        output = mint.nn.functional.embedding(input, weight, max_norm=0.4)
        print(output)


    # mindspore.mint.nn.functional.one_hot
    def test_functional_one_hot(self):
        tensor = Tensor(np.array([0, 1, 2]), mindspore.int32)
        num_classes = 3
        output = mint.nn.functional.one_hot(tensor, num_classes)
        print(output)


# mindspore.mint.nn.functional: Loss Functions
class FunctionalLossTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Loss Functions '''
    
    # mindspore.mint.nn.functional.cross_entropy
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_cross_entropy(self):
        # Case 1: Indices labels
        inputs = Tensor(np.random.randn(3, 5), mindspore.float16)
        target = Tensor(np.array([1, 0, 4]), mindspore.int32)
        output = mint.nn.functional.cross_entropy(inputs, target)
        # Case 2: Probability labels
        inputs = Tensor(np.random.randn(3, 5), mindspore.float16)
        target = Tensor(np.random.randn(3, 5), mindspore.float16)
        output = mint.nn.functional.cross_entropy(inputs, target)


    # mindspore.mint.nn.functional.binary_cross_entropy
    def test_functional_binary_cross_entropy(self):
        input = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float16)
        target = Tensor(np.array([0., 1., 0.]), mindspore.float16)
        weight = Tensor(np.array([1, 2, 2]), mindspore.float16)
        output = mint.nn.functional.binary_cross_entropy(input, target, weight)
        print(output)


    # mindspore.mint.nn.functional.binary_cross_entropy_with_logits
    def test_functional_binary_cross_entropy_with_logits(self):
        input = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float16)
        target = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float16)
        weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float16)
        pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float16)
        output = mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, 'mean', pos_weight)
        print(output)


    # mindspore.mint.nn.functional.kl_div
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_kl_div(self):
        input = Tensor(np.array([[0.5, 0.5], [0.4, 0.6]]), mindspore.float16)
        target = Tensor(np.array([[0., 1.], [1., 0.]]), mindspore.float16)
        output = mint.nn.functional.kl_div(input, target, reduction='mean', log_target=False)
        print(output)


    # mindspore.mint.nn.functional.l1_loss
    def test_functional_l1_loss(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float16)
        target = Tensor([[6, 5, 4], [3, 2, 1]], mstype.float16)
        output = mint.nn.functional.l1_loss(x, target, reduction="mean")
        print(output)


    # mindspore.mint.nn.functional.mse_loss
    def test_functional_mse_loss(self):
        logits = Tensor(np.array([1, 2, 3]), mindspore.float16)
        labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float16)
        output = mint.nn.functional.mse_loss(logits, labels, reduction='none')
        print(output)


    # mindspore.mint.nn.functional.nll_loss
    def test_functional_nll_loss(self):
        input = mindspore.Tensor(np.random.randn(3, 5), mindspore.float16)
        target = mindspore.Tensor(np.array([1, 0, 4]), mindspore.int32)
        output = mint.nn.functional.nll_loss(input, target)


    # mindspore.mint.nn.functional.smooth_l1_loss
    def test_functional_smooth_l1_loss(self):
        input = Tensor(np.array([2, 2, 3]), mindspore.float16)
        target = Tensor(np.array([2, 2, 2]), mindspore.float16)
        beta = 1.0
        reduction_1 = 'none'
        output = mint.nn.functional.smooth_l1_loss(input, target, reduction_1, beta)
        print(output)
        reduction_2 = 'mean'
        output = mint.nn.functional.smooth_l1_loss(input, target, reduction_2, beta)
        print(output)
        reduction_3 = 'sum'
        output = mint.nn.functional.smooth_l1_loss(input, target, reduction_3, beta)
        print(output)


# mindspore.mint.nn.functional: Vision functions
class FunctionalVisionTest(unittest.TestCase):
    ''' mindspore.mint.nn.functional API about Vision functions '''
    
    # mindspore.mint.nn.functional.interpolate
    def test_functional_interpolate(self):
        input = Tensor([[[1, 2, 3], [4, 5, 6]]], mindspore.float16)
        output = mint.nn.functional.interpolate(input, size=(6,), mode='nearest')
        print(output)


    # mindspore.mint.nn.functional.grid_sample float16
    def test_functional_grid_sample_float16(self):
        input_x = Tensor(np.arange(16).reshape((2, 2, 2, 2)).astype(np.float16))
        grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(np.float16))
        output = mint.nn.functional.grid_sample(input_x, grid, mode='bilinear', padding_mode='zeros',
                                 align_corners=True)
        print(output)


    # mindspore.mint.nn.functional.grid_sample float32
    def test_functional_grid_sample_float32(self):
        input_x = Tensor(np.arange(16).reshape((2, 2, 2, 2)).astype(np.float32))
        grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(np.float32))
        output = mint.nn.functional.grid_sample(input_x, grid, mode='bilinear', padding_mode='zeros',
                                 align_corners=True)
        print(output)


    # mindspore.mint.nn.functional.pad
    def test_functional_pad(self):
        x = mindspore.Tensor(np.arange(1 * 2 * 2 * 2).reshape((1, 2, 2, 2)), dtype=mindspore.float16)
        output = mint.nn.functional.pad(x, [1, 0, 0, 1], mode='constant', value=6.0)
        print(output)


    # mindspore.mint.nn.functional.pixel_shuffle
    @unittest.skipUnless(version.parse(mindspore.__version__) >= version.parse("2.6.0"), "version at least 2.6.0 is required")
    def test_functional_pixel_shuffle(self):
        input = mint.randn(1, 9, 4, 4)
        output = mint.nn.functional.pixel_shuffle(input, 3)
        print(output.shape)
