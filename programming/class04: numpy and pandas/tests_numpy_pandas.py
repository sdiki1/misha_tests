"""
Автотесты к ноутбуку «NumPy и pandas: числа и таблицы в Python» (numpy_pandas.ipynb).

Файл должен лежать в одной папке с ноутбуком. Использование — как в прошлых ноутбуках:

    from tests_numpy_pandas import test_task1

    def task1(numbers):
        return np.array(numbers)

    test_task1(task1)

test_taskN(func) вызывает функцию с подготовленными массивами и таблицами и сравнивает
результат с ожидаемым. Массивы сравниваются через numpy.testing, таблицы и столбцы —
через pandas.testing (типы чисел и имена столбцов-результатов не придираются).
Отчёт печатается прямо в ячейке:

    Задача 9. passing_scores — тестов: 3
      ✅ тест 1: passing_scores(np.array([45, 70, 88, 69]), 70) — OK
      ❌ тест 2: passing_scores(np.array([10, 20]), 50) — неверный ответ
         ожидалось: np.array([])
         получено: np.array([10, 20])
    Итог: пройдено 2 из 3 ❌

Тесты не читают data/students.csv: все маленькие массивы и таблицы созданы здесь.
Если функция изменила переданную таблицу или массив, тест сообщит об этом отдельно.
"""

import contextlib
import io
import math
import traceback

import numpy as np
import pandas as pd

__all__ = ["Call", "FLOAT_TOLERANCE"] + [f"test_task{i}" for i in range(1, 21)]

FLOAT_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Описание тестов
# ---------------------------------------------------------------------------

class Call:
    """Тест-вызов: Call(ожидаемый_результат, *аргументы, desc="как показать вызов")."""

    def __init__(self, expected, *args, desc=None, **kwargs):
        self.expected, self.args, self.kwargs, self.desc = expected, args, kwargs, desc

    def describe(self, name):
        if self.desc:
            return self.desc
        parts = [_fmt_arg(a) for a in self.args] + [f"{k}={_fmt_arg(v)}" for k, v in self.kwargs.items()]
        return f"{name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def _fmt_arg(value):
    if isinstance(value, np.ndarray):
        text = np.array2string(value, separator=", ", threshold=20).replace("\n", "")
        return f"np.array({' '.join(text.split())})"
    if isinstance(value, pd.DataFrame):
        return "df"
    if isinstance(value, pd.Series):
        return "series"
    return repr(value)


def _fmt(value):
    if isinstance(value, pd.DataFrame):
        if len(value) <= 12:
            return "DataFrame:\n" + value.to_string()
        return f"DataFrame {value.shape[0]}×{value.shape[1]}:\n" + value.head(8).to_string() + "\n..."
    if isinstance(value, pd.Series):
        if len(value) <= 12:
            return "Series:\n" + value.to_string()
        return f"Series из {len(value)} значений:\n" + value.head(8).to_string() + "\n..."
    if isinstance(value, np.ndarray):
        return _fmt_arg(value)
    if isinstance(value, BaseException):
        return f"исключение {type(value).__name__}: {value}"
    try:
        text = repr(value)
    except Exception:
        text = f"<объект {type(value).__name__}>"
    return text if len(text) <= 300 else text[:300] + "…"


# ---------------------------------------------------------------------------
# Сравнение результатов
# ---------------------------------------------------------------------------

def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _equal(got, expected):
    """Возвращает (совпало, подсказка)."""
    if isinstance(expected, pd.DataFrame):
        if not isinstance(got, pd.DataFrame):
            return False, f"ожидался DataFrame, а получен {type(got).__name__}"
        try:
            pd.testing.assert_frame_equal(got, expected, check_dtype=False, check_names=False,
                                          check_index_type=False, check_column_type=False)
            return True, None
        except AssertionError:
            if list(got.columns) != list(expected.columns):
                return False, f"столбцы {list(got.columns)}, а ожидались {list(expected.columns)}"
            if len(got) != len(expected):
                return False, f"строк {len(got)}, а ожидалось {len(expected)}"
            if list(got.index) != list(expected.index):
                return False, "строки те же, но их порядок или индексы отличаются"
            return False, None
    if isinstance(expected, pd.Series):
        if not isinstance(got, pd.Series):
            return False, f"ожидался столбец (Series), а получен {type(got).__name__}"
        try:
            pd.testing.assert_series_equal(got, expected, check_dtype=False, check_names=False,
                                           check_index_type=False)
            return True, None
        except AssertionError:
            if len(got) != len(expected):
                return False, f"значений {len(got)}, а ожидалось {len(expected)}"
            if list(got.index) != list(expected.index):
                return False, "значения те же, но порядок или индексы отличаются"
            return False, None
    if isinstance(expected, np.ndarray):
        if not isinstance(got, np.ndarray):
            return False, f"ожидался массив NumPy (np.ndarray), а получен {type(got).__name__}"
        if got.shape != expected.shape:
            return False, f"форма массива {got.shape}, а ожидалась {expected.shape}"
        try:
            if expected.dtype.kind in "biuf" and got.dtype.kind in "biuf":
                np.testing.assert_allclose(got, expected, rtol=1e-7, atol=FLOAT_TOLERANCE)
            else:
                np.testing.assert_array_equal(got, expected)
            return True, None
        except AssertionError:
            return False, None
    if isinstance(expected, tuple):
        if not isinstance(got, tuple):
            return False, f"ожидался кортеж, а получен {type(got).__name__}"
        if len(got) != len(expected):
            return False, f"в кортеже {len(got)} элемента(ов), а ожидалось {len(expected)}"
        return all(_equal(g, e)[0] for g, e in zip(got, expected)), None
    if isinstance(expected, list):
        if not isinstance(got, list):
            return False, f"ожидался список, а получен {type(got).__name__}"
        return len(got) == len(expected) and all(_equal(g, e)[0] for g, e in zip(got, expected)), None
    if isinstance(expected, bool):
        return isinstance(got, (bool, np.bool_)) and bool(got) == expected, None
    if _is_number(expected):
        if not _is_number(got):
            return False, f"ожидалось число, а получен {type(got).__name__}"
        if math.isnan(expected) or math.isnan(got):
            return math.isnan(expected) and math.isnan(got), None
        return math.isclose(float(got), float(expected), rel_tol=1e-9, abs_tol=FLOAT_TOLERANCE), None
    try:
        return bool(got == expected), None
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Запуск с контролем неизменности входных данных
# ---------------------------------------------------------------------------

def _snapshot(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy(deep=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    return None


def _changed(value, snapshot):
    if snapshot is None:
        return False
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return not value.equals(snapshot)
    return not (value.shape == snapshot.shape and np.array_equal(value, snapshot, equal_nan=True))


def _run_case(func, case):
    snapshots = [_snapshot(a) for a in case.args]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        value = func(*case.args, **case.kwargs)
    changed = []
    for arg, snap in zip(case.args, snapshots):
        if _changed(arg, snap):
            kind = "DataFrame" if isinstance(arg, pd.DataFrame) else "массив" if isinstance(arg, np.ndarray) else "Series"
            changed.append(kind)
    return value, buffer.getvalue(), changed


def _describe_exception(exc):
    where = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if frame.filename != __file__ and "site-packages" not in frame.filename:
            code = f": {frame.line.strip()}" if frame.line else ""
            where = f" (строка {frame.lineno}{code})"
            break
    text = f"{type(exc).__name__}: {exc}{where}"
    if isinstance(exc, KeyError):
        text += "\n     подсказка: проверьте название столбца — KeyError значит, что такого столбца нет"
    return text


def _show(label, text):
    lines = text.split("\n")
    print(f"     {label}: {lines[0]}")
    for line in lines[1:]:
        print(f"       {line}")


def _check(number, func):
    task = _TASKS[number]
    name, cases = task["name"], task["cases"]
    if not callable(func):
        print(f"❌ test_task{number}: нужно передать саму функцию, а не результат вызова — "
              f"пишите test_task{number}(task{number}) без скобок после task{number}.")
        return

    print(f"Задача {number}. {name} — тестов: {len(cases)}")
    passed = 0
    for i, case in enumerate(cases, 1):
        desc = case.describe(name)
        try:
            got, printed, changed = _run_case(func, case)
        except Exception as exc:
            print(f"  ❌ тест {i}: {desc} — ошибка выполнения — {_describe_exception(exc)}")
            continue
        ok, hint = _equal(got, case.expected)
        if ok and not changed:
            passed += 1
            print(f"  ✅ тест {i}: {desc} — OK")
            continue
        if changed:
            print(f"  ❌ тест {i}: {desc} — функция изменила переданный {changed[0]}")
            print("     подсказка: работайте с копией — df.copy() для таблицы, arr.copy() для массива — "
                  "или создавайте новый объект")
            if not ok:
                _show("ожидалось", _fmt(case.expected))
                _show("получено", _fmt(got))
            continue
        print(f"  ❌ тест {i}: {desc} — неверный ответ")
        _show("ожидалось", _fmt(case.expected))
        _show("получено", _fmt(got))
        if got is None and printed.strip():
            print("     подсказка: функция напечатала результат через print(), а должна вернуть его через return")
        elif hint:
            print(f"     подсказка: {hint}")

    if passed == len(cases):
        print(f"Итог: все {len(cases)} тестов пройдены ✅")
    else:
        print(f"Итог: пройдено {passed} из {len(cases)} ❌")


# ---------------------------------------------------------------------------
# Учебные таблицы для тестов (маленькие, не связаны с data/students.csv)
# ---------------------------------------------------------------------------

def _small_df():
    return pd.DataFrame({
        "student_id": [1, 2, 3, 4, 5, 6],
        "name": ["Аня", "Боря", "Вера", "Гоша", "Даша", "Егор"],
        "grade": [7, 8, 7, 9, 8, 7],
        "city": ["Москва", "Казань", "Москва", "Тверь", "Казань", "Москва"],
        "math_score": [85.0, np.nan, 72.0, 90.0, 64.0, np.nan],
        "python_score": [78.0, 88.0, np.nan, 95.0, 70.0, 81.0],
        "hours_per_week": [4, 6, 3, 8, 5, 4],
    })


def _empty_df():
    return _small_df().iloc[0:0]


def _one_row_df():
    return _small_df().iloc[[3]]


def _both_missing_df():
    return pd.DataFrame({
        "student_id": [10, 11],
        "name": ["Ким", "Лев"],
        "grade": [10, 10],
        "city": ["Пермь", "Пермь"],
        "math_score": [np.nan, 50.0],
        "python_score": [np.nan, 60.0],
        "hours_per_week": [2, 7],
    })


def _no_missing_df():
    return pd.DataFrame({
        "student_id": [21, 22, 23],
        "name": ["Оля", "Петя", "Рита"],
        "grade": [11, 11, 10],
        "city": ["Самара", "Самара", "Пермь"],
        "math_score": [91.0, 67.0, 80.0],
        "python_score": [89.0, 74.0, 85.0],
        "hours_per_week": [9, 4, 6],
    })


def _grades_with_gaps_df():
    return pd.DataFrame({
        "student_id": [31, 32, 33],
        "name": ["Саша", "Таня", "Федя"],
        "grade": [7, 7, 8],
        "city": ["Тверь", "Тверь", "Тверь"],
        "math_score": [np.nan, np.nan, 50.0],
        "python_score": [60.0, 70.0, 80.0],
        "hours_per_week": [3, 3, 3],
    })


def _with_average(df):
    result = df.copy()
    result["average_score"] = df[["math_score", "python_score"]].mean(axis=1)
    return result


def _with_filled_math(df):
    result = df.copy()
    result["math_score"] = df["math_score"].fillna(df["math_score"].median())
    return result


_DF = _small_df()

_TASKS = {
    # ---------------- NumPy ----------------
    1: {"name": "make_array", "cases": [
        Call(np.array([1, 2, 3]), [1, 2, 3]),
        Call(np.array([2.5, 3.5]), [2.5, 3.5]),
        Call(np.array([]), []),
        Call(np.array([42]), [42]),
    ]},
    2: {"name": "make_range", "cases": [
        Call(np.array([0, 1, 2, 3, 4]), 0, 5),
        Call(np.array([-2, -1, 0, 1, 2]), -2, 3),
        Call(np.array([], dtype=int), 3, 3),
        Call(np.array([7]), 7, 8),
    ]},
    3: {"name": "first_and_last", "cases": [
        Call((4, 16), np.array([4, 8, 15, 16])),
        Call((7, 7), np.array([7])),
        Call((2.5, -1.0), np.array([2.5, 0.0, -1.0])),
    ]},
    4: {"name": "middle_values", "cases": [
        Call(np.array([2, 3, 4]), np.array([1, 2, 3, 4, 5])),
        Call(np.array([], dtype=int), np.array([1, 2])),
        Call(np.array([], dtype=int), np.array([1])),
        Call(np.array([20]), np.array([10, 20, 30])),
    ]},
    5: {"name": "increase_scores", "cases": [
        Call(np.array([55, 65, 75]), np.array([50, 60, 70]), 5),
        Call(np.array([], dtype=int), np.array([], dtype=int), 10),
        Call(np.array([97, 100]), np.array([100, 103]), -3),
        Call(np.array([1.5, 2.5]), np.array([1.0, 2.0]), 0.5),
    ]},
    6: {"name": "square_numbers", "cases": [
        Call(np.array([1, 4, 9]), np.array([1, -2, 3])),
        Call(np.array([], dtype=int), np.array([], dtype=int)),
        Call(np.array([0.25, 0.0]), np.array([0.5, 0.0])),
        Call(np.array([100]), np.array([10])),
    ]},
    7: {"name": "total_score", "cases": [
        Call(60, np.array([10, 20, 30])),
        Call(0, np.array([], dtype=int)),
        Call(7, np.array([7])),
        Call(1.5, np.array([0.5, 0.25, 0.75])),
    ]},
    8: {"name": "average_score", "cases": [
        Call(90.0, np.array([80, 90, 100])),
        Call(73.0, np.array([73])),
        Call(2.5, np.array([1, 2, 3, 4])),
        Call(0.0, np.array([-5, 5])),
    ]},
    9: {"name": "passing_scores", "cases": [
        Call(np.array([70, 88]), np.array([45, 70, 88, 69]), 70),
        Call(np.array([], dtype=int), np.array([10, 20]), 50),
        Call(np.array([90, 95, 100]), np.array([90, 95, 100]), 60),
        Call(np.array([50.5]), np.array([49.9, 50.5]), 50),
    ]},
    10: {"name": "subject_means", "cases": [
        Call(np.array([70.0, 80.0]), np.array([[80, 90], [60, 70]])),
        Call(np.array([55.0, 65.0, 75.0]), np.array([[55, 65, 75]])),
        Call(np.array([2.0, 5.0, 8.0]), np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])),
    ]},
    # ---------------- pandas ----------------
    11: {"name": "get_names", "cases": [
        Call(_DF["name"], _small_df(), desc="get_names(df)"),
        Call(_empty_df()["name"], _empty_df(), desc="get_names(пустая таблица)"),
    ]},
    12: {"name": "get_scores", "cases": [
        Call(_DF[["math_score", "python_score"]], _small_df(), desc="get_scores(df)"),
        Call(_one_row_df()[["math_score", "python_score"]], _one_row_df(), desc="get_scores(таблица из одной строки)"),
    ]},
    13: {"name": "first_students", "cases": [
        Call(_DF.iloc[:2], _small_df(), 2, desc="first_students(df, 2)"),
        Call(_DF.iloc[:0], _small_df(), 0, desc="first_students(df, 0)"),
        Call(_DF.iloc[:6], _small_df(), 10, desc="first_students(df, 10) — больше, чем строк"),
    ]},
    14: {"name": "students_in_grade", "cases": [
        Call(_DF[_DF["grade"] == 7], _small_df(), 7, desc="students_in_grade(df, 7)"),
        Call(_DF[_DF["grade"] == 11], _small_df(), 11, desc="students_in_grade(df, 11) — таких нет"),
        Call(_DF[_DF["grade"] == 9], _small_df(), 9, desc="students_in_grade(df, 9)"),
    ]},
    15: {"name": "high_math_scores", "cases": [
        Call(_DF[_DF["math_score"] >= 70], _small_df(), 70, desc="high_math_scores(df, 70)"),
        Call(_DF[_DF["math_score"] >= 100], _small_df(), 100, desc="high_math_scores(df, 100) — таких нет"),
        Call(_DF[_DF["math_score"] >= 0], _small_df(), 0, desc="high_math_scores(df, 0) — пропуски не проходят"),
    ]},
    16: {"name": "add_average_score", "cases": [
        Call(_with_average(_small_df()), _small_df(), desc="add_average_score(df)"),
        Call(_with_average(_both_missing_df()), _both_missing_df(), desc="add_average_score(df с двумя пропусками в одной строке)"),
        Call(_with_average(_empty_df()), _empty_df(), desc="add_average_score(пустая таблица)"),
    ]},
    17: {"name": "sort_by_python", "cases": [
        Call(_DF.sort_values("python_score", ascending=False), _small_df(), desc="sort_by_python(df)"),
        Call(_empty_df().sort_values("python_score", ascending=False), _empty_df(), desc="sort_by_python(пустая таблица)"),
        Call(_no_missing_df().sort_values("python_score", ascending=False), _no_missing_df(), desc="sort_by_python(df без пропусков)"),
    ]},
    18: {"name": "students_per_city", "cases": [
        Call(_DF["city"].value_counts(), _small_df(), desc="students_per_city(df)"),
        Call(_one_row_df()["city"].value_counts(), _one_row_df(), desc="students_per_city(таблица из одной строки)"),
    ]},
    19: {"name": "fill_math_scores", "cases": [
        Call(_with_filled_math(_small_df()), _small_df(), desc="fill_math_scores(df)"),
        Call(_with_filled_math(_no_missing_df()), _no_missing_df(), desc="fill_math_scores(df без пропусков)"),
        Call(_with_filled_math(_both_missing_df()), _both_missing_df(), desc="fill_math_scores(df, где медиана — единственное значение)"),
    ]},
    20: {"name": "mean_score_by_grade", "cases": [
        Call(_DF.groupby("grade")["math_score"].mean(), _small_df(), desc="mean_score_by_grade(df)"),
        Call(_grades_with_gaps_df().groupby("grade")["math_score"].mean(), _grades_with_gaps_df(),
             desc="mean_score_by_grade(df, где у класса одни пропуски)"),
        Call(_one_row_df().groupby("grade")["math_score"].mean(), _one_row_df(), desc="mean_score_by_grade(таблица из одной строки)"),
    ]},
}


def _make_test(number):
    def test(func):
        _check(number, func)
    test.__name__ = f"test_task{number}"
    test.__doc__ = f"Задача {number}. {_TASKS[number]['name']}"
    return test


test_task1 = _make_test(1)
test_task2 = _make_test(2)
test_task3 = _make_test(3)
test_task4 = _make_test(4)
test_task5 = _make_test(5)
test_task6 = _make_test(6)
test_task7 = _make_test(7)
test_task8 = _make_test(8)
test_task9 = _make_test(9)
test_task10 = _make_test(10)
test_task11 = _make_test(11)
test_task12 = _make_test(12)
test_task13 = _make_test(13)
test_task14 = _make_test(14)
test_task15 = _make_test(15)
test_task16 = _make_test(16)
test_task17 = _make_test(17)
test_task18 = _make_test(18)
test_task19 = _make_test(19)
test_task20 = _make_test(20)
