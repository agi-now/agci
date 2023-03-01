import pytest as pytest

from agci import Interpreter


@pytest.mark.parametrize("code", [
    """
def main():
    return 1
""", """
def main():
    return
""", """
def main():
    if 1 > 2:
        return 1
    else:
        return 2
""", """
def main():
    for i in [0, 1, 2,]:
        return i
    return -1
""", """
def main():
    for i in [0, 1, 2,]:
        if i == 3:
            return i
    return -1
""", """
def main():
    for i in [0, 1, 2,]:
        if i == 0:
            break
        if i == 1:
            return i
    return -1
""", """
def main():
    i = 0
    while i < 3:
        if i == 0:
            break
        if i == 1:
            return i
        i += 1
    return -1
""", """
def main():
    for i in [1, 2, 3]:
        for j in [2, 3, 4]:
            if i == j:
                return i
""", """
def main():
    total_sum = 0 
    for i in [1, 2, 3, 5]:
        if i == 3:
            continue
        for j in [2, 3, 4, 5, 6, 7]:
            if i == j:
                continue
            if j == 5:
                continue
            if i == 5 and j == 6 or False:
                return total_sum
            total_sum += i * j
            if j == 7:
                break
        for j in [2, 3, 4, 5, 6, 7]:
            if i == j:
                break
            total_sum += i * j
    return total_sum
""",
])
def test_value(code):
    expected_result = eval(compile(code, '<string>', 'exec').co_consts[0])

    interpreter = Interpreter({'range': range})
    interpreter.load_code(code)
    result = interpreter.run_main()
    assert result == expected_result

