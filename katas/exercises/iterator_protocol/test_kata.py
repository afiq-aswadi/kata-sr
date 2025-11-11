"""Tests for iterator protocol kata."""



try:
    from user_kata import Countdown
    from user_kata import InfiniteSequence, take
    from user_kata import BatchIterator
    from user_kata import fibonacci_generator, take
    from user_kata import take
except ImportError:
    from .reference import Countdown
    from .reference import InfiniteSequence, take
    from .reference import BatchIterator
    from .reference import fibonacci_generator, take
    from .reference import take


def test_countdown():

    cd = Countdown(5)
    result = list(cd)
    assert result == [5, 4, 3, 2, 1, 0]


def test_countdown_zero():

    cd = Countdown(0)
    result = list(cd)
    assert result == [0]


def test_infinite_sequence():

    seq = InfiniteSequence(0, 1)
    result = take(seq, 5)
    assert result == [0, 1, 2, 3, 4]


def test_infinite_sequence_custom_start():

    seq = InfiniteSequence(10, 2)
    result = take(seq, 5)
    assert result == [10, 12, 14, 16, 18]


def test_batch_iterator():

    items = [1, 2, 3, 4, 5, 6, 7]
    batches = list(BatchIterator(items, 3))

    assert batches == [[1, 2, 3], [4, 5, 6], [7]]


def test_batch_iterator_exact_size():

    items = [1, 2, 3, 4]
    batches = list(BatchIterator(items, 2))

    assert batches == [[1, 2], [3, 4]]


def test_fibonacci_generator():

    fib = fibonacci_generator()
    result = take(fib, 10)

    assert result == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_take_exhausted():

    it = iter([1, 2, 3])
    result = take(it, 10)

    assert result == [1, 2, 3]


def test_countdown_reusable():

    cd = Countdown(3)
    first = list(cd)
    second = list(cd)

    # Iterator is exhausted after first use
    assert first == [3, 2, 1, 0]
    assert second == []
