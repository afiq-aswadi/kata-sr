"""Tests for Task Scheduler kata."""

try:
    from user_kata import least_interval
except ImportError:
    from .reference import least_interval


def test_task_scheduler_example1():
    assert least_interval(["A","A","A","B","B","B"], 2) == 8

def test_task_scheduler_example2():
    assert least_interval(["A","A","A","B","B","B"], 0) == 6

def test_task_scheduler_no_cooldown():
    assert least_interval(["A","B","C","D","E","F"], 1) == 6

def test_task_scheduler_single_task():
    assert least_interval(["A","A","A","A"], 2) == 10
