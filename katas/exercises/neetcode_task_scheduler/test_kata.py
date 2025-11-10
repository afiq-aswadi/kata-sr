"""Tests for Task Scheduler kata."""

def test_task_scheduler_example1():
    from template import least_interval
    assert least_interval(["A","A","A","B","B","B"], 2) == 8

def test_task_scheduler_example2():
    from template import least_interval
    assert least_interval(["A","A","A","B","B","B"], 0) == 6

def test_task_scheduler_no_cooldown():
    from template import least_interval
    assert least_interval(["A","B","C","D","E","F"], 1) == 6

def test_task_scheduler_single_task():
    from template import least_interval
    assert least_interval(["A","A","A","A"], 2) == 10
