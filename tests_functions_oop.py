"""
Автотесты к ноутбуку «Функции и ООП в Python» (python_functions_oop.ipynb).

Файл должен лежать в одной папке с ноутбуком. Использование в ячейке задачи —
такое же, как в ноутбуке про контейнеры:

    from tests_functions_oop import test_clamp

    def clamp(value, low, high):
        return max(low, min(value, high))

    test_clamp(clamp)

Отличие от прошлого ноутбука: теперь проверяется не вывод программы, а
функции и классы. test_<имя>(объект) вызывает вашу функцию с разными
аргументами (или прогоняет сценарий с вашим классом) и сравнивает результат
с ожидаемым. Отчёт печатается прямо в ячейке:

    Задача 1. clamp — тестов: 8
      ✅ тест 1: clamp(5, 0, 10) — OK
      ❌ тест 2: clamp(-3, 0, 10) — неверный ответ
         ожидалось: 0
         получено: -3
    Итог: пройдено 7 из 8 ❌

Правила сравнения:
  * числа сравниваются с точностью FLOAT_TOLERANCE, всё остальное — точно;
    список и кортеж — разные типы, True и 1 — разные значения;
  * если задача требует исключения, проверяется его тип (подкласс подходит);
  * если условие требует или запрещает конструкции («используйте рекурсию»,
    «нельзя sorted()»), проверка заглядывает в исходный код решения.
"""

import ast
import contextlib
import importlib
import inspect
import io
import math
import sys
import textwrap
import time
import traceback
import types

FLOAT_TOLERANCE = 1e-6
_MAX_SHOWN = 110


# ---------------------------------------------------------------------------
# Описание тестов
# ---------------------------------------------------------------------------

class Raises:
    """Ожидаемый результат — исключение указанного типа (или с таким именем)."""

    def __init__(self, exc, contains=None):
        self.exc = exc              # класс исключения или его имя строкой
        self.contains = contains    # подстрока, которая должна быть в сообщении

    @property
    def name(self):
        return self.exc if isinstance(self.exc, str) else self.exc.__name__

    def matches(self, exc_obj):
        if isinstance(self.exc, str):
            ok = any(cls.__name__ == self.exc for cls in type(exc_obj).__mro__)
        else:
            ok = isinstance(exc_obj, self.exc)
        if ok and self.contains is not None:
            ok = self.contains in str(exc_obj)
        return ok

    def __repr__(self):
        return f"исключение {self.name}"


class Call:
    """Тест-вызов функции: Call(ожидаемый_результат, *аргументы, **именованные)."""

    def __init__(self, expected, *args, **kwargs):
        self.expected, self.args, self.kwargs = expected, args, kwargs

    def describe(self, name):
        parts = [repr(a) for a in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{name}({', '.join(parts)})"

    def run(self, func, *rest):
        return func(*self.args, **self.kwargs)


class Scenario:
    """Тест-сценарий: Scenario(описание, функция_от_проверяемых_объектов, ожидаемый_результат)."""

    def __init__(self, description, run, expected):
        self.description, self._run, self.expected = description, run, expected

    def describe(self, name):
        return self.description

    def run(self, *objs):
        return self._run(*objs)


# ---------------------------------------------------------------------------
# Сравнение результатов
# ---------------------------------------------------------------------------

def _equal(got, expected):
    if isinstance(expected, bool) or isinstance(got, bool):
        return isinstance(got, bool) and isinstance(expected, bool) and got == expected
    if isinstance(expected, (int, float)):
        return (isinstance(got, (int, float))
                and math.isclose(got, expected, rel_tol=1e-9, abs_tol=FLOAT_TOLERANCE))
    if isinstance(expected, (list, tuple)):
        return (type(got) is type(expected) and len(got) == len(expected)
                and all(_equal(g, e) for g, e in zip(got, expected)))
    if isinstance(expected, dict):
        return (isinstance(got, dict) and got.keys() == expected.keys()
                and all(_equal(got[k], expected[k]) for k in expected))
    if isinstance(expected, (set, frozenset)):
        return isinstance(got, (set, frozenset)) and got == expected
    try:
        return bool(got == expected)
    except Exception:
        return False


def _fmt(value):
    if isinstance(value, Raises):
        return repr(value)
    if isinstance(value, BaseException):
        return f"исключение {type(value).__name__}: {value}"
    try:
        text = repr(value)
    except Exception:
        text = f"<объект {type(value).__name__}>"
    if len(text) > _MAX_SHOWN:
        text = text[:_MAX_SHOWN] + "…"
    return text


def _hint(got, expected, printed):
    if got is None and expected is not None and printed.strip():
        return "подсказка: функция напечатала результат через print(), а должна вернуть его через return"
    if isinstance(expected, list) and isinstance(got, tuple):
        return "подсказка: ожидался список, а получен кортеж"
    if isinstance(expected, tuple) and isinstance(got, list):
        return "подсказка: ожидался кортеж, а получен список"
    if isinstance(expected, (int, float)) and isinstance(got, str):
        return "подсказка: получена строка, а ожидалось число"
    if isinstance(expected, bool) and isinstance(got, int):
        return "подсказка: ожидалось True/False, а получено число"
    if isinstance(expected, str) and got is not None and not isinstance(got, str):
        return "подсказка: ожидалась строка, а получено значение другого типа"
    return None


# ---------------------------------------------------------------------------
# Проверка ограничений условия по исходному коду
# ---------------------------------------------------------------------------

def _source_tree(obj):
    """AST исходника функции или класса (для класса — из исходников его методов)."""
    if inspect.isclass(obj):
        body = []
        for member in vars(obj).values():
            if isinstance(member, (staticmethod, classmethod, property)):
                member = getattr(member, "__func__", None) or getattr(member, "fget", None)
            if inspect.isfunction(member):
                try:
                    body += ast.parse(textwrap.dedent(inspect.getsource(member))).body
                except Exception:
                    pass
        return ast.Module(body=body, type_ignores=[]) if body else None
    try:
        source = textwrap.dedent(inspect.getsource(obj))
        return ast.parse(source)
    except Exception:
        return None


def _function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _structural_checks(obj, cfg):
    """Возвращает список нарушений (строк) для объекта obj по настройкам cfg."""
    problems = []
    name = getattr(obj, "__name__", "объект")
    tree = _source_tree(obj)

    if callable(obj) and not inspect.isclass(obj):
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            params = {}
        kinds = [p.kind for p in params.values()]
        if cfg.get("require_varargs") and inspect.Parameter.VAR_POSITIONAL not in kinds:
            problems.append(f"функция {name} должна принимать произвольное число позиционных "
                            f"аргументов через *args (параметр вида *имя)")
        if cfg.get("require_varkw") and inspect.Parameter.VAR_KEYWORD not in kinds:
            problems.append(f"функция {name} должна принимать произвольные именованные "
                            f"аргументы через **kwargs (параметр вида **имя)")
        for pname, default in cfg.get("require_defaults", {}).items():
            param = params.get(pname)
            if param is None:
                problems.append(f"у функции {name} должен быть параметр {pname}")
            elif param.default is inspect.Parameter.empty:
                problems.append(f"параметр {pname} функции {name} должен иметь значение по умолчанию")
            elif param.default != default:
                problems.append(f"значение по умолчанию параметра {pname} должно быть {default!r}, "
                                f"а не {param.default!r}")
        if cfg.get("require_yield") and not inspect.isgeneratorfunction(inspect.unwrap(obj)):
            problems.append(f"{name} должна быть функцией-генератором: используйте yield")

    if tree is None:
        return problems   # исходник недоступен — проверки по коду пропускаем

    names_used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs_used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = set(cfg.get("forbid_names", ()))
    if cfg.get("forbid_print"):
        forbidden.add("print")
    for bad in sorted(forbidden & names_used):
        problems.append(f"в решении используется запрещённое условием: {bad}()")
    for bad in sorted(set(cfg.get("forbid_attrs", ())) & attrs_used):
        problems.append(f"в решении используется запрещённое условием: .{bad}()")
    for needed in cfg.get("require_names", ()):
        if needed not in names_used and needed not in attrs_used:
            problems.append(f"по условию в решении нужно использовать {needed}()")
    if cfg.get("forbid_negative_step"):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Slice) and isinstance(node.step, ast.UnaryOp)
                    and isinstance(node.step.op, ast.USub)):
                problems.append("в решении используется срез с отрицательным шагом")
                break
    if cfg.get("forbid_listcomp") and any(isinstance(n, ast.ListComp) for n in ast.walk(tree)):
        problems.append("по условию списковые включения использовать нельзя — "
                        "используйте map/filter/sorted")
    if cfg.get("require_lambda") and not any(isinstance(n, ast.Lambda) for n in ast.walk(tree)):
        problems.append("по условию в решении должна быть lambda-функция")
    if cfg.get("require_nonlocal") and not any(isinstance(n, ast.Nonlocal) for n in ast.walk(tree)):
        problems.append("по условию нужно использовать nonlocal")
    if cfg.get("require_try") and not any(isinstance(n, ast.Try) for n in ast.walk(tree)):
        problems.append("по условию нужно обработать исключение через try/except")
    if cfg.get("require_return"):
        node = _function_node(tree, name) or tree
        if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)):
            problems.append(f"функция {name} должна возвращать результат через return")
    if cfg.get("require_recursion"):
        node = _function_node(tree, name)
        calls_itself = node is not None and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
            for n in ast.walk(node))
        if not calls_itself:
            problems.append(f"в решении нет рекурсии: функция {name} должна вызывать сама себя")
    if cfg.get("require_super") and "super" not in names_used:
        problems.append(f"в {name} нужно вызвать конструктор родителя через super()")
    if cfg.get("require_yield_in_source") and not any(
            isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(tree)):
        problems.append(f"в {name} должен быть генератор (yield)")
    return problems


# ---------------------------------------------------------------------------
# Запуск и отчёт
# ---------------------------------------------------------------------------

def _run_case(case, objs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        value = case.run(*objs)
    return value, buffer.getvalue()


def _describe_exception(exc):
    where = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if frame.filename != __file__:
            code = f": {frame.line.strip()}" if frame.line else ""
            where = f" (строка {frame.lineno}{code})"
            break
    text = f"{type(exc).__name__}: {exc}{where}"
    if isinstance(exc, AttributeError):
        text += "\n     подсказка: проверьте, что метод или атрибут с таким именем определён"
    elif isinstance(exc, TypeError) and "positional argument" in str(exc):
        text += "\n     подсказка: проверьте число параметров функции или метода (не забыли self?)"
    return text


def _show(label, text):
    print(f"     {label}: {text}")


def _check(key, *objs):
    task = _TASKS[key]
    title, name, cases = task["title"], task.get("name", "?"), task["cases"]
    expected_objs = task.get("objects", 1)
    if len(objs) != expected_objs:
        print(f"❌ {title}: нужно передать {expected_objs} объект(а/ов), а передано {len(objs)}. "
              f"Смотрите строку вызова в шаблоне.")
        return
    if not all(callable(o) for o in objs):
        print(f"❌ {title}: нужно передать саму функцию или класс, а не результат вызова "
              f"(пишите без скобок: test_...({name}))")
        return

    print(f"{title} — тестов: {len(cases)}")

    checks = task.get("checks", {})
    if checks and not any(isinstance(k, int) for k in checks):
        checks = {0: checks}
    violations = []
    for idx, cfg in checks.items():
        violations += _structural_checks(objs[idx], cfg)
    for problem in violations:
        print(f"  ⛔ {problem}")

    passed = 0
    printed_something = False
    for i, case in enumerate(cases, 1):
        desc = case.describe(name)
        expected = case.expected
        try:
            got, printed = _run_case(case, objs)
        except Exception as exc:
            if isinstance(expected, Raises) and expected.matches(exc):
                passed += 1
                print(f"  ✅ тест {i}: {desc} — OK")
            elif isinstance(expected, Raises):
                print(f"  ❌ тест {i}: {desc} — не то исключение")
                _show("ожидалось", _fmt(expected))
                _show("получено", _fmt(exc))
            else:
                print(f"  ❌ тест {i}: {desc} — ошибка выполнения — {_describe_exception(exc)}")
            continue
        if printed.strip():
            printed_something = True
        if isinstance(expected, Raises):
            print(f"  ❌ тест {i}: {desc} — исключения не было")
            _show("ожидалось", _fmt(expected))
            _show("получено", _fmt(got))
            continue
        if _equal(got, expected):
            passed += 1
            print(f"  ✅ тест {i}: {desc} — OK")
            continue
        print(f"  ❌ тест {i}: {desc} — неверный ответ")
        _show("ожидалось", _fmt(expected))
        _show("получено", _fmt(got))
        hint = _hint(got, expected, printed)
        if hint:
            print(f"     {hint}")

    if task.get("no_print") and printed_something:
        violations.append("печать")
        print("  ⛔ функция печатает через print(), а по условию должна только возвращать результат")

    if passed == len(cases) and not violations:
        print(f"Итог: все {len(cases)} тестов пройдены ✅")
    elif passed == len(cases):
        print("Итог: тесты пройдены, но нарушено ограничение условия ❌")
    else:
        print(f"Итог: пройдено {passed} из {len(cases)} ❌")


def _is_generator(obj):
    return isinstance(obj, types.GeneratorType)


_TASKS = {}

# ===========================================================================
# ЧАСТЬ I. ФУНКЦИИ — мини-задачи из теории
# ===========================================================================

_TASKS["rectangle_area"] = {
    "title": "Мини-задача 1. rectangle_area",
    "name": "rectangle_area",
    "checks": {"require_return": True, "forbid_print": True},
    "no_print": True,
    "cases": [
        Call(12, 3, 4),
        Call(25, 5, 5),
        Call(10.0, 2.5, 4),
        Call(0, 0, 7),
        Call(1, 1, 1),
        Call(30, 10, 3),
    ],
}


def _counter_cases():
    def three_calls(make_counter):
        counter = make_counter()
        return counter(), counter(), counter()

    def independent(make_counter):
        first, second = make_counter(), make_counter()
        first(); first()
        second()
        return first(), second()

    def returns_function(make_counter):
        return callable(make_counter())

    def long_run(make_counter):
        counter = make_counter()
        for _ in range(99):
            counter()
        return counter()

    return [
        Scenario("три вызова счётчика подряд дают 1, 2, 3", three_calls, (1, 2, 3)),
        Scenario("два счётчика считают независимо", independent, (3, 2)),
        Scenario("make_counter() возвращает функцию", returns_function, True),
        Scenario("сотый вызов возвращает 100", long_run, 100),
    ]


_TASKS["make_counter"] = {
    "title": "Мини-задача 2. make_counter",
    "name": "make_counter",
    "checks": {"require_nonlocal": True},
    "cases": _counter_cases(),
}


def _add_bonus_cases():
    def original_untouched(add_bonus):
        scores = [70, 80, 90]
        add_bonus(scores, 5)
        return scores

    def returns_new_object(add_bonus):
        scores = [1, 2]
        return add_bonus(scores, 0) is scores

    return [
        Call([80, 85, 95], [70, 75, 85], 10),
        Call([], [], 5),
        Call([-4, 6], [1, 11], -5),
        Call([100], [100], 0),
        Scenario("исходный список не меняется", original_untouched, [70, 80, 90]),
        Scenario("возвращается новый список, а не тот же объект", returns_new_object, False),
    ]


_TASKS["add_bonus"] = {
    "title": "Мини-задача 3. add_bonus",
    "name": "add_bonus",
    "checks": {"forbid_attrs": ("append", "extend", "insert")},
    "cases": _add_bonus_cases(),
}

_TASKS["join_words"] = {
    "title": "Мини-задача 4. join_words",
    "name": "join_words",
    "checks": {"require_varargs": True},
    "cases": [
        Call("a-b-c", "-", "a", "b", "c"),
        Call("hello world", " ", "hello", "world"),
        Call("", ", "),
        Call("one", "/", "one"),
        Call("1+2+3+4", "+", "1", "2", "3", "4"),
        Call("xy", "", "x", "y"),
    ],
}


def _compose_cases():
    return [
        Scenario("compose(str, abs)(-5) == '5'", lambda compose: compose(str, abs)(-5), "5"),
        Scenario("compose(lambda x: x + 1, lambda x: x * 2)(3) == 7",
                 lambda compose: compose(lambda x: x + 1, lambda x: x * 2)(3), 7),
        Scenario("compose(len, str)(12345) == 5", lambda compose: compose(len, str)(12345), 5),
        Scenario("порядок важен: compose(lambda x: x * 2, lambda x: x + 1)(3) == 8",
                 lambda compose: compose(lambda x: x * 2, lambda x: x + 1)(3), 8),
        Scenario("compose возвращает функцию", lambda compose: callable(compose(abs, abs)), True),
        Scenario("результат можно вызывать много раз",
                 lambda compose: [compose(str.upper, str.strip)(s) for s in (" a ", "b ")], ["A", "B"]),
    ]


_TASKS["compose"] = {
    "title": "Мини-задача 5. compose",
    "name": "compose",
    "cases": _compose_cases(),
}


def _sort_by_length_cases():
    def original_untouched(sort_by_length):
        words = ["ccc", "a", "bb"]
        sort_by_length(words)
        return words

    return [
        Call(["a", "bb", "ccc"], ["bb", "a", "ccc"]),
        Call(["b", "aa", "ab"], ["ab", "aa", "b"]),
        Call([], []),
        Call(["x", "x", "y"], ["y", "x", "x"]),
        Call(["як", "кот", "пёс", "слон"], ["слон", "пёс", "як", "кот"]),
        Call(["c", "bb", "dd", "aaa"], ["aaa", "dd", "bb", "c"]),
        Scenario("исходный список не меняется", original_untouched, ["ccc", "a", "bb"]),
    ]


_TASKS["sort_by_length"] = {
    "title": "Мини-задача 6. sort_by_length",
    "name": "sort_by_length",
    "checks": {"require_lambda": True, "forbid_attrs": ("sort",)},
    "cases": _sort_by_length_cases(),
}

_TASKS["sum_digits"] = {
    "title": "Мини-задача 7. sum_digits",
    "name": "sum_digits",
    "checks": {"require_recursion": True},
    "cases": [
        Call(6, 123),
        Call(0, 0),
        Call(9, 9),
        Call(45, 99999),
        Call(1, 1000),
        Call(28, 1234567),
        Call(10, 19),
    ],
}


def _double_result_cases():
    def add(a, b):
        return a + b

    def mul(a, b=1):
        return a * b

    def decorated_add(double_result):
        return double_result(add)(2, 3)

    def keywords_supported(double_result):
        return double_result(mul)(4, b=5)

    def name_preserved(double_result):
        return double_result(add).__name__

    def original_untouched(double_result):
        double_result(add)
        return add(2, 3)

    def is_function(double_result):
        return callable(double_result(add))

    def with_decorator_syntax(double_result):
        @double_result
        def square(x):
            return x * x
        return square(3), square(10)

    return [
        Scenario("double_result(add)(2, 3) == 10", decorated_add, 10),
        Scenario("именованные аргументы передаются в исходную функцию", keywords_supported, 40),
        Scenario("__name__ обёрнутой функции сохранён (functools.wraps)", name_preserved, "add"),
        Scenario("исходная функция не изменилась", original_untouched, 5),
        Scenario("декоратор возвращает функцию", is_function, True),
        Scenario("работает через синтаксис @double_result", with_decorator_syntax, (18, 200)),
    ]


_TASKS["double_result"] = {
    "title": "Мини-задача 8. double_result",
    "name": "double_result",
    "checks": {"require_names": ("wraps",)},
    "cases": _double_result_cases(),
}


def _evens_cases():
    return [
        Scenario("list(evens_up_to(10)) == [0, 2, 4, 6, 8, 10]",
                 lambda f: list(f(10)), [0, 2, 4, 6, 8, 10]),
        Scenario("list(evens_up_to(7)) == [0, 2, 4, 6]", lambda f: list(f(7)), [0, 2, 4, 6]),
        Scenario("list(evens_up_to(0)) == [0]", lambda f: list(f(0)), [0]),
        Scenario("list(evens_up_to(-1)) == []", lambda f: list(f(-1)), []),
        Scenario("evens_up_to(10) — это генератор, а не список",
                 lambda f: _is_generator(f(10)), True),
        Scenario("next(evens_up_to(10)) == 0", lambda f: next(f(10)), 0),
        Scenario("sum(evens_up_to(100)) == 2550", lambda f: sum(f(100)), 2550),
    ]


_TASKS["evens_up_to"] = {
    "title": "Мини-задача 9. evens_up_to",
    "name": "evens_up_to",
    "checks": {"require_yield": True},
    "cases": _evens_cases(),
}


# ===========================================================================
# ЧАСТЬ I. ФУНКЦИИ — 10 заданий
# ===========================================================================

_TASKS["clamp"] = {
    "title": "Задача 1. clamp",
    "name": "clamp",
    "checks": {"require_return": True, "forbid_print": True},
    "no_print": True,
    "cases": [
        Call(5, 5, 0, 10),
        Call(0, -3, 0, 10),
        Call(10, 42, 0, 10),
        Call(0, 0, 0, 10),
        Call(10, 10, 0, 10),
        Call(2, 2.5, 1, 2),
        Call(-1.5, -1.5, -2, -1),
        Call(7, 7, 7, 7),
        Call(-2, -100, -2, 2),
    ],
}

_TASKS["snake_to_camel"] = {
    "title": "Задача 2. snake_to_camel",
    "name": "snake_to_camel",
    "checks": {"require_return": True},
    "cases": [
        Call("helloWorld", "hello_world"),
        Call("x", "x"),
        Call("aBC", "a_b_c"),
        Call("makeHttpRequest", "make_http_request"),
        Call("already", "already"),
        Call("helloWorld", "HELLO_WORLD"),
        Call("userId2", "user_id_2"),
        Call("oneTwoThreeFour", "one_two_three_four"),
        Call("привет", "ПРИВЕТ"),
    ],
}


def _chunk_cases():
    def original_untouched(chunk):
        items = [1, 2, 3, 4, 5]
        chunk(items, 2)
        return items

    def returns_list_of_lists(chunk):
        result = chunk([1, 2, 3], 2)
        return isinstance(result, list) and all(isinstance(part, list) for part in result)

    return [
        Call([[1, 2], [3, 4], [5]], [1, 2, 3, 4, 5], 2),
        Call([], [], 3),
        Call([[1, 2, 3]], [1, 2, 3], 3),
        Call([[1, 2, 3]], [1, 2, 3], 5),
        Call([["a", "b", "c"], ["d", "e", "f"], ["g"]], list("abcdefg"), 3),
        Call([[1], [2], [3], [4]], [1, 2, 3, 4], 1),
        Call([[0, 0], [0, 0]], [0, 0, 0, 0], 2),
        Scenario("исходный список не меняется", original_untouched, [1, 2, 3, 4, 5]),
        Scenario("результат — список списков", returns_list_of_lists, True),
    ]


_TASKS["chunk"] = {
    "title": "Задача 3. chunk",
    "name": "chunk",
    "cases": _chunk_cases(),
}

_TASKS["make_email"] = {
    "title": "Задача 4. make_email",
    "name": "make_email",
    "checks": {"require_defaults": {"domain": "example.com", "separator": "."}},
    "cases": [
        Call("ivan.petrov@example.com", "Ivan", "Petrov"),
        Call("anna.smith@mail.ru", "Anna", "Smith", domain="mail.ru"),
        Call("anna_smith@example.com", "Anna", "Smith", separator="_"),
        Call("bo-lee@x.io", separator="-", last="Lee", first="Bo", domain="x.io"),
        Call("john.doe@example.com", "JOHN", "DOE"),
        Call("maria.lopez@uni.edu", "maria", "lopez", "uni.edu"),
        Call("aleksey.ivanov@corp.ru", "Aleksey", "Ivanov", "corp.ru", "."),
    ],
}

_TASKS["average"] = {
    "title": "Задача 5. average",
    "name": "average",
    "checks": {"require_varargs": True},
    "cases": [
        Call(2.0, 1, 2, 3),
        Call(10.0, 10),
        Call(0.0),
        Call(1.5, 1, 2),
        Call(0.0, -1, 1),
        Call(3.0, 2.5, 3.5),
        Call(5.0, *[4, 5, 6]),
        Call(2.5, 1, 2, 3, 4),
    ],
}

_TASKS["make_tag"] = {
    "title": "Задача 6. make_tag",
    "name": "make_tag",
    "checks": {"require_varkw": True},
    "cases": [
        Call("<br>", "br"),
        Call('<a href="x" target="_blank">', "a", href="x", target="_blank"),
        Call('<img src="cat.png" width="100">', "img", src="cat.png", width=100),
        Call('<div id="main">', "div", id="main"),
        Call('<input type="text" value="">', "input", type="text", value=""),
        Call("<hr>", "hr"),
        Call('<p lang="ru">', "p", lang="ru"),
    ],
}


def _pipeline_cases():
    return [
        Scenario("apply_pipeline(3, lambda x: x + 1, lambda x: x * 2) == 8",
                 lambda f: f(3, lambda x: x + 1, lambda x: x * 2), 8),
        Scenario("apply_pipeline(5) == 5 (функций нет — значение как есть)", lambda f: f(5), 5),
        Scenario("apply_pipeline('  hi ', str.strip, str.upper) == 'HI'",
                 lambda f: f("  hi ", str.strip, str.upper), "HI"),
        Scenario("apply_pipeline([3, 1, 2], sorted, lambda l: l[0]) == 1",
                 lambda f: f([3, 1, 2], sorted, lambda l: l[0]), 1),
        Scenario("apply_pipeline(2, str, len) == 1", lambda f: f(2, str, len), 1),
        Scenario("apply_pipeline(-3, abs, str, len) == 1", lambda f: f(-3, abs, str, len), 1),
        Scenario("порядок применения: apply_pipeline(3, lambda x: x * 2, lambda x: x + 1) == 7",
                 lambda f: f(3, lambda x: x * 2, lambda x: x + 1), 7),
        Scenario("функции можно передать распаковкой: apply_pipeline(1, *[f1, f2, f3])",
                 lambda f: f(1, *[lambda x: x + 1, lambda x: x + 1, lambda x: x * 10]), 30),
    ]


_TASKS["apply_pipeline"] = {
    "title": "Задача 7. apply_pipeline",
    "name": "apply_pipeline",
    "checks": {"require_varargs": True},
    "cases": _pipeline_cases(),
}

_TASKS["even_squares_desc"] = {
    "title": "Задача 8. even_squares_desc",
    "name": "even_squares_desc",
    "checks": {"require_lambda": True, "forbid_listcomp": True,
               "require_names": ("map", "filter", "sorted")},
    "cases": [
        Call([36, 16, 4], [1, 2, 3, 4, 5, 6]),
        Call([], []),
        Call([], [1, 3, 5]),
        Call([4, 4], [-2, 2]),
        Call([100, 0], [0, 7, 10]),
        Call([36, 16, 4], [6, 2, 4]),
        Call([64, 16], [-8, 3, 4, 9]),
    ],
}


def _flatten_cases():
    def original_untouched(flatten):
        nested = [1, [2, [3]]]
        flatten(nested)
        return nested

    return [
        Call([1, 2, 3, 4], [1, [2, [3, [4]]]]),
        Call([], []),
        Call([1], [[], [[]], 1]),
        Call([1, 2, 3], [1, 2, 3]),
        Call([1, 2, 3, 4, 5, 6], [[1, 2], [3, [4, 5]], 6]),
        Call(["a", "b", "c"], ["a", ["b", ["c"]]]),
        Call([1, 2, 3, 4, 5, 6, 7, 8], [[[[1]], 2], [3, [4, [5, [6]]]], 7, [8]]),
        Scenario("исходный список не меняется", original_untouched, [1, [2, [3]]]),
    ]


_TASKS["flatten"] = {
    "title": "Задача 9. flatten",
    "name": "flatten",
    "checks": {"require_recursion": True},
    "cases": _flatten_cases(),
}


def _memoize_cases():
    def correct_value(memoize, fib):
        return memoize(lambda x: x * 2)(4)

    def computes_once(memoize, fib):
        calls = [0]

        def slow(x):
            calls[0] += 1
            return x * x

        fast = memoize(slow)
        fast(3); fast(3); fast(3)
        return fast(3), calls[0]

    def separate_args(memoize, fib):
        calls = [0]

        def slow(x):
            calls[0] += 1
            return x * x

        fast = memoize(slow)
        return fast(2), fast(3), fast(2), calls[0]

    def two_arguments(memoize, fib):
        return memoize(lambda a, b: a ** b)(2, 10)

    def name_preserved(memoize, fib):
        def slow(x):
            return x
        return memoize(slow).__name__

    def fib_small(memoize, fib):
        return fib(0), fib(1), fib(2), fib(10)

    def fib_wrapped(memoize, fib):
        return hasattr(fib, "__wrapped__")

    def fib_big(memoize, fib):
        if not hasattr(fib, "__wrapped__"):
            return "fib не обёрнута декоратором memoize (нужны @memoize и functools.wraps)"
        started = time.perf_counter()
        fib(30)
        if time.perf_counter() - started > 0.05:
            return "fib(30) считалась слишком долго — кэш memoize не работает"
        return fib(90)

    return [
        Scenario("memoize(lambda x: x * 2)(4) == 8", correct_value, 8),
        Scenario("повторный вызов с теми же аргументами не вычисляет заново", computes_once, (9, 1)),
        Scenario("разные аргументы кэшируются отдельно", separate_args, (4, 9, 4, 2)),
        Scenario("работает с несколькими позиционными аргументами", two_arguments, 1024),
        Scenario("__name__ обёрнутой функции сохранён (functools.wraps)", name_preserved, "slow"),
        Scenario("fib(0), fib(1), fib(2), fib(10) == (0, 1, 1, 55)", fib_small, (0, 1, 1, 55)),
        Scenario("fib обёрнута декоратором memoize", fib_wrapped, True),
        Scenario("fib(90) считается мгновенно благодаря кэшу", fib_big, 2880067194370816120),
    ]


_TASKS["memoize_fib"] = {
    "title": "Задача 10. memoize + fib",
    "name": "memoize",
    "objects": 2,
    "checks": {0: {"require_names": ("wraps",)}, 1: {"require_recursion": True}},
    "cases": _memoize_cases(),
}

# ===========================================================================
# ЧАСТЬ II. ООП — мини-задачи из теории
# ===========================================================================

def _point_cases():
    def attributes(Point):
        p = Point(3, -2)
        return p.x, p.y

    def distance(Point):
        return Point(0, 0).distance_to(Point(3, 4))

    def distance_symmetric(Point):
        a, b = Point(1, 1), Point(4, 5)
        return a.distance_to(b), b.distance_to(a)

    def move_changes_state(Point):
        p = Point(1, 1)
        p.move(2, -3)
        return p.x, p.y

    def move_returns_none(Point):
        return Point(0, 0).move(1, 1)

    def independent(Point):
        a, b = Point(0, 0), Point(0, 0)
        a.move(5, 5)
        return b.x, b.y

    return [
        Scenario("Point(3, -2) хранит x и y", attributes, (3, -2)),
        Scenario("Point(0, 0).distance_to(Point(3, 4)) == 5.0", distance, 5.0),
        Scenario("расстояние симметрично", distance_symmetric, (5.0, 5.0)),
        Scenario("move(2, -3) сдвигает точку (1, 1) в (3, -2)", move_changes_state, (3, -2)),
        Scenario("move возвращает None (меняет объект, а не создаёт новый)", move_returns_none, None),
        Scenario("две точки не влияют друг на друга", independent, (0, 0)),
    ]


_TASKS["point"] = {
    "title": "Мини-задача 10. класс Point",
    "name": "Point",
    "cases": _point_cases(),
}


def _playlist_cases():
    def song_attrs(Playlist, Song):
        s = Song("Yesterday", 125)
        return s.title, s.duration

    def empty_playlist(Playlist, Song):
        p = Playlist("Дорога")
        return p.name, p.songs, p.total_duration(), p.longest()

    def add_and_total(Playlist, Song):
        p = Playlist("Утро")
        p.add(Song("A", 100))
        p.add(Song("B", 250))
        return len(p.songs), p.total_duration()

    def longest_song(Playlist, Song):
        p = Playlist("Утро")
        p.add(Song("A", 100)); p.add(Song("B", 250)); p.add(Song("C", 200))
        return p.longest().title

    def stores_same_objects(Playlist, Song):
        p = Playlist("X")
        song = Song("A", 10)
        p.add(song)
        return p.songs[0] is song

    def independent_playlists(Playlist, Song):
        p1, p2 = Playlist("1"), Playlist("2")
        p1.add(Song("A", 10))
        return len(p2.songs)

    return [
        Scenario("Song('Yesterday', 125) хранит title и duration", song_attrs, ("Yesterday", 125)),
        Scenario("пустой плейлист: songs == [], total_duration() == 0, longest() is None",
                 empty_playlist, ("Дорога", [], 0, None)),
        Scenario("add добавляет песни, total_duration суммирует длительности", add_and_total, (2, 350)),
        Scenario("longest() возвращает самую длинную песню", longest_song, "B"),
        Scenario("плейлист хранит тот же объект Song, а не копию", stores_same_objects, True),
        Scenario("у каждого плейлиста свой список песен", independent_playlists, 0),
    ]


_TASKS["playlist"] = {
    "title": "Мини-задача 11. классы Song и Playlist",
    "name": "Playlist",
    "objects": 2,
    "cases": _playlist_cases(),
}


def _vector_cases():
    def add(Vector):
        v = Vector(1, 2) + Vector(3, 4)
        return v.x, v.y

    def add_returns_vector(Vector):
        return isinstance(Vector(1, 2) + Vector(3, 4), Vector)

    def eq_true(Vector):
        return Vector(1, 2) == Vector(1, 2)

    def eq_false(Vector):
        return Vector(1, 2) == Vector(2, 1)

    def ne(Vector):
        return Vector(1, 2) != Vector(2, 1)

    def to_str(Vector):
        return str(Vector(3, -4))

    def length(Vector):
        return abs(Vector(3, 4))

    def combined(Vector):
        return Vector(1, 2) + Vector(3, 4) == Vector(4, 6)

    def operands_untouched(Vector):
        a, b = Vector(1, 2), Vector(3, 4)
        a + b
        return (a.x, a.y, b.x, b.y)

    return [
        Scenario("Vector(1, 2) + Vector(3, 4) даёт (4, 6)", add, (4, 6)),
        Scenario("результат сложения — тоже Vector", add_returns_vector, True),
        Scenario("Vector(1, 2) == Vector(1, 2)", eq_true, True),
        Scenario("Vector(1, 2) == Vector(2, 1) → False", eq_false, False),
        Scenario("!= работает автоматически через __eq__", ne, True),
        Scenario("str(Vector(3, -4)) == '(3, -4)'", to_str, "(3, -4)"),
        Scenario("abs(Vector(3, 4)) == 5.0", length, 5.0),
        Scenario("Vector(1, 2) + Vector(3, 4) == Vector(4, 6)", combined, True),
        Scenario("сложение не меняет исходные векторы", operands_untouched, (1, 2, 3, 4)),
    ]


_TASKS["vector"] = {
    "title": "Мини-задача 12. класс Vector",
    "name": "Vector",
    "cases": _vector_cases(),
}


def _shapes_cases():
    def subclasses(Shape, Rectangle, Circle):
        return issubclass(Rectangle, Shape), issubclass(Circle, Shape)

    def rect_area(Shape, Rectangle, Circle):
        return Rectangle(3, 4).area()

    def circle_area(Shape, Rectangle, Circle):
        return Circle(1).area()

    def names(Shape, Rectangle, Circle):
        return Rectangle(1, 1).name, Circle(1).name

    def describe_rect(Shape, Rectangle, Circle):
        return Rectangle(3, 4).describe()

    def describe_circle(Shape, Rectangle, Circle):
        return Circle(2).describe()

    def base_area_not_implemented(Shape, Rectangle, Circle):
        return Shape("фигура").area()

    def polymorphic_sum(Shape, Rectangle, Circle):
        shapes = [Rectangle(2, 5), Circle(1), Rectangle(1, 1)]
        return round(sum(s.area() for s in shapes), 4)

    def instances(Shape, Rectangle, Circle):
        return isinstance(Circle(1), Shape), isinstance(Rectangle(1, 2), Circle)

    return [
        Scenario("Rectangle и Circle — подклассы Shape", subclasses, (True, True)),
        Scenario("Rectangle(3, 4).area() == 12", rect_area, 12),
        Scenario("Circle(1).area() == math.pi", circle_area, math.pi),
        Scenario("name задаётся через super().__init__: 'прямоугольник' и 'круг'",
                 names, ("прямоугольник", "круг")),
        Scenario("Rectangle(3, 4).describe() == 'прямоугольник: площадь 12.00'",
                 describe_rect, "прямоугольник: площадь 12.00"),
        Scenario("Circle(2).describe() == 'круг: площадь 12.57'", describe_circle, "круг: площадь 12.57"),
        Scenario("Shape('фигура').area() бросает NotImplementedError",
                 base_area_not_implemented, Raises(NotImplementedError)),
        Scenario("сумма площадей списка фигур считается полиморфно", polymorphic_sum, round(11 + math.pi, 4)),
        Scenario("isinstance(Circle(1), Shape) и не isinstance(Rectangle, Circle)", instances, (True, False)),
    ]


_TASKS["shapes"] = {
    "title": "Мини-задача 13. Shape, Rectangle, Circle",
    "name": "Shape",
    "objects": 3,
    "checks": {1: {"require_super": True}, 2: {"require_super": True}},
    "cases": _shapes_cases(),
}


def _safe_sqrt_cases():
    return [
        Scenario("safe_sqrt(16) == 4.0", lambda f, E: f(16), 4.0),
        Scenario("safe_sqrt(0) == 0.0", lambda f, E: f(0), 0.0),
        Scenario("safe_sqrt(2) == 1.4142...", lambda f, E: f(2), 2 ** 0.5),
        Scenario("safe_sqrt(-1) бросает NegativeNumberError", lambda f, E: f(-1), Raises("NegativeNumberError")),
        Scenario("safe_sqrt(-0.5) бросает NegativeNumberError", lambda f, E: f(-0.5), Raises("NegativeNumberError")),
        Scenario("NegativeNumberError — подкласс ValueError", lambda f, E: issubclass(E, ValueError), True),
        Scenario("его можно поймать как ValueError", lambda f, E: _catch(lambda: f(-4), ValueError), True),
        Scenario("в сообщении исключения есть само число",
                 lambda f, E: _message(lambda: f(-9)), Raises("NegativeNumberError", contains="-9")),
    ]


def _catch(func, exc_type):
    try:
        func()
    except exc_type:
        return True
    return False


def _message(func):
    func()   # ожидается исключение; проверка сообщения — через Raises(contains=...)


_TASKS["safe_sqrt"] = {
    "title": "Мини-задача 14. safe_sqrt и NegativeNumberError",
    "name": "safe_sqrt",
    "objects": 2,
    "cases": _safe_sqrt_cases(),
}

_TASKS["read_numbers"] = {
    "title": "Мини-задача 15. read_numbers",
    "name": "read_numbers",
    "checks": {"require_try": True},
    "cases": [
        Call([10, -5, 7], ["10", "abc", "-5", "3.5", "7"]),
        Call([], []),
        Call([], ["a", "b"]),
        Call([4], [" 4 "]),
        Call([1, 2, 3], ["1", "2", "3"]),
        Call([0, -0], ["0", "-0", "", "x1"]),
        Call([100], ["1e2", "100", "0x10"]),
    ],
}


def _shapes_module_cases():
    def _load():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            if "shapes" in sys.modules:
                module = importlib.reload(sys.modules["shapes"])
            else:
                module = importlib.import_module("shapes")
        return module, buffer.getvalue()

    def importable():
        module, _ = _load()
        return module.__name__

    def rect():
        module, _ = _load()
        return module.rectangle_area(3, 4)

    def circle():
        module, _ = _load()
        return round(module.circle_area(2), 4)

    def silent_import():
        _, printed = _load()
        return printed

    def has_main_guard():
        module, _ = _load()
        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = ast.dump(node.test)
                if "__name__" in test and "__main__" in test:
                    return True
        return False

    return [
        Scenario("модуль shapes импортируется", importable, "shapes"),
        Scenario("shapes.rectangle_area(3, 4) == 12", rect, 12),
        Scenario("shapes.circle_area(2) == 12.5664", circle, 12.5664),
        Scenario("при импорте модуль ничего не печатает", silent_import, ""),
        Scenario("в модуле есть блок if __name__ == '__main__'", has_main_guard, True),
    ]


_TASKS["shapes_module"] = {
    "title": "Мини-задача 16. модуль shapes.py",
    "name": "shapes",
    "objects": 0,
    "cases": _shapes_module_cases(),
}


# ===========================================================================
# ЧАСТЬ II. ООП — уровень 1: BankAccount
# ===========================================================================

def _acc_step1_cases():
    def owner(C):
        return C("Аня").owner

    def two_accounts(C):
        a, b = C("Аня"), C("Боря")
        return a.owner, b.owner

    def is_class(C):
        return inspect.isclass(C)

    def has_attribute(C):
        return hasattr(C("Вера"), "owner")

    return [
        Scenario("BankAccount — это класс", is_class, True),
        Scenario("BankAccount('Аня').owner == 'Аня'", owner, "Аня"),
        Scenario("у экземпляра есть атрибут owner", has_attribute, True),
        Scenario("два счёта хранят разных владельцев", two_accounts, ("Аня", "Боря")),
    ]


_TASKS["account_step1"] = {
    "title": "BankAccount, шаг 1: класс и __init__",
    "name": "BankAccount",
    "cases": _acc_step1_cases(),
}


def _acc_step2_cases():
    def default_balance(C):
        return C("Аня").balance

    def given_balance(C):
        return C("Аня", 250).balance

    def history_empty(C):
        return C("Аня").history

    def class_attr(C):
        return C.currency

    def instance_sees_class_attr(C):
        return C("Аня").currency

    def separate_histories(C):
        a, b = C("Аня"), C("Боря")
        return a.history is b.history

    def owner_kept(C):
        return C("Вера", 10).owner

    return [
        Scenario("баланс по умолчанию равен 0", default_balance, 0),
        Scenario("BankAccount('Аня', 250).balance == 250", given_balance, 250),
        Scenario("history у нового счёта — пустой список", history_empty, []),
        Scenario("BankAccount.currency == 'RUB' (атрибут класса)", class_attr, "RUB"),
        Scenario("экземпляр видит атрибут класса: acc.currency == 'RUB'", instance_sees_class_attr, "RUB"),
        Scenario("у разных счетов разные списки history (не общий объект)", separate_histories, False),
        Scenario("owner по-прежнему сохраняется", owner_kept, "Вера"),
    ]


_TASKS["account_step2"] = {
    "title": "BankAccount, шаг 2: поля balance, history и атрибут класса currency",
    "name": "BankAccount",
    "cases": _acc_step2_cases(),
}


def _acc_step3_cases():
    def deposit(C):
        acc = C("Аня", 50)
        returned = acc.deposit(100)
        return returned, acc.balance

    def withdraw(C):
        acc = C("Аня", 150)
        returned = acc.withdraw(30)
        return returned, acc.balance

    def sequence(C):
        acc = C("Аня")
        acc.deposit(100); acc.deposit(50); acc.withdraw(70)
        return acc.balance

    def deposit_returns_value(C):
        return C("Аня").deposit(10) is None

    def floats(C):
        acc = C("Аня", 10.5)
        acc.deposit(0.5)
        return acc.balance

    def independent(C):
        a, b = C("Аня", 100), C("Боря", 100)
        a.deposit(50)
        return b.balance

    return [
        Scenario("deposit(100) при балансе 50 возвращает 150 и меняет balance", deposit, (150, 150)),
        Scenario("withdraw(30) при балансе 150 возвращает 120 и меняет balance", withdraw, (120, 120)),
        Scenario("серия операций: +100, +50, -70 → 80", sequence, 80),
        Scenario("deposit возвращает новый баланс, а не None", deposit_returns_value, False),
        Scenario("дробные суммы тоже работают", floats, 11.0),
        Scenario("операции одного счёта не влияют на другой", independent, 100),
    ]


_TASKS["account_step3"] = {
    "title": "BankAccount, шаг 3: методы deposit и withdraw",
    "name": "BankAccount",
    "cases": _acc_step3_cases(),
}


def _acc_step4_cases():
    def history_records(C):
        acc = C("Аня", 100)
        acc.deposit(100); acc.withdraw(30)
        return acc.history

    def totals(C):
        acc = C("Аня")
        acc.deposit(100); acc.deposit(50); acc.withdraw(30); acc.withdraw(20)
        return acc.total_deposited(), acc.total_withdrawn()

    def new_account_totals(C):
        acc = C("Аня", 500)
        return acc.total_deposited(), acc.total_withdrawn(), acc.history

    def order_kept(C):
        acc = C("Аня")
        acc.deposit(1); acc.deposit(2); acc.deposit(3)
        return [amount for _, amount in acc.history]

    def balance_still_correct(C):
        acc = C("Аня", 10)
        acc.deposit(5); acc.withdraw(3)
        return acc.balance

    return [
        Scenario("после deposit(100) и withdraw(30) history == [('deposit', 100), ('withdraw', 30)]",
                 history_records, [("deposit", 100), ("withdraw", 30)]),
        Scenario("total_deposited() и total_withdrawn() суммируют операции", totals, (150, 50)),
        Scenario("начальный баланс не считается операцией", new_account_totals, (0, 0, [])),
        Scenario("операции записываются в порядке выполнения", order_kept, [1, 2, 3]),
        Scenario("баланс по-прежнему считается верно", balance_still_correct, 12),
    ]


_TASKS["account_step4"] = {
    "title": "BankAccount, шаг 4: история операций",
    "name": "BankAccount",
    "cases": _acc_step4_cases(),
}


def _acc_step5_cases():
    def negative_deposit(C, E):
        C("Аня").deposit(-5)

    def zero_withdraw(C, E):
        C("Аня", 100).withdraw(0)

    def string_amount(C, E):
        C("Аня").deposit("10")

    def insufficient(C, E):
        C("Аня", 100).withdraw(1000)

    def unchanged_after_failure(C, E):
        acc = C("Аня", 100)
        try:
            acc.withdraw(1000)
        except Exception:
            pass
        try:
            acc.deposit(-1)
        except Exception:
            pass
        return acc.balance, acc.history

    def exception_hierarchy(C, E):
        return issubclass(E, Exception)

    def empty_owner(C, E):
        C("")

    def negative_start(C, E):
        C("Аня", -10)

    def available(C, E):
        return C("Аня", 70).available()

    def exact_balance_ok(C, E):
        acc = C("Аня", 100)
        acc.withdraw(100)
        return acc.balance

    return [
        Scenario("deposit(-5) бросает ValueError", negative_deposit, Raises(ValueError)),
        Scenario("withdraw(0) бросает ValueError", zero_withdraw, Raises(ValueError)),
        Scenario("deposit('10') бросает TypeError", string_amount, Raises(TypeError)),
        Scenario("withdraw(1000) при балансе 100 бросает InsufficientFundsError",
                 insufficient, Raises("InsufficientFundsError")),
        Scenario("после неудачных операций balance и history не меняются",
                 unchanged_after_failure, (100, [])),
        Scenario("InsufficientFundsError — подкласс Exception", exception_hierarchy, True),
        Scenario("BankAccount('') бросает ValueError", empty_owner, Raises(ValueError)),
        Scenario("отрицательный начальный баланс — ValueError", negative_start, Raises(ValueError)),
        Scenario("available() возвращает доступную сумму (пока это баланс)", available, 70),
        Scenario("снять ровно весь баланс можно", exact_balance_ok, 0),
    ]


_TASKS["account_step5"] = {
    "title": "BankAccount, шаг 5: проверка входных данных и своё исключение",
    "name": "BankAccount",
    "objects": 2,
    "cases": _acc_step5_cases(),
}


def _acc_step6_cases():
    def to_str(C):
        return str(C("Аня", 150))

    def to_repr(C):
        return repr(C("Аня", 150))

    def eq_true(C):
        return C("Аня", 150) == C("Аня", 150)

    def eq_false(C):
        return C("Аня", 150) == C("Аня", 151)

    def eq_other_type(C):
        return C("Аня", 150) == "Аня"

    def sorted_by_balance(C):
        accounts = [C("Аня", 300), C("Боря", 100), C("Вера", 200)]
        return [acc.owner for acc in sorted(accounts)]

    def max_account(C):
        accounts = [C("Аня", 300), C("Боря", 100), C("Вера", 200)]
        return max(accounts).owner

    def lt(C):
        return C("Аня", 100) < C("Боря", 200), C("Аня", 300) < C("Боря", 200)

    def ne(C):
        return C("Аня", 1) != C("Аня", 2)

    return [
        Scenario("str(BankAccount('Аня', 150)) == 'Аня: 150 RUB'", to_str, "Аня: 150 RUB"),
        Scenario("repr(BankAccount('Аня', 150)) == \"BankAccount('Аня', 150)\"", to_repr, "BankAccount('Аня', 150)"),
        Scenario("счета с одинаковыми owner и balance равны", eq_true, True),
        Scenario("счета с разным balance не равны", eq_false, False),
        Scenario("сравнение со строкой даёт False, а не ошибку", eq_other_type, False),
        Scenario("sorted(список счетов) сортирует по balance", sorted_by_balance, ["Боря", "Вера", "Аня"]),
        Scenario("max(список счетов) — счёт с наибольшим balance", max_account, "Аня"),
        Scenario("оператор < сравнивает балансы", lt, (True, False)),
        Scenario("!= работает через __eq__", ne, True),
    ]


_TASKS["account_step6"] = {
    "title": "BankAccount, шаг 6: магические методы",
    "name": "BankAccount",
    "cases": _acc_step6_cases(),
}


def _acc_step7_cases():
    def transfer(C):
        a, b = C("Аня", 100), C("Боря", 100)
        a.transfer(b, 50)
        return a.balance, b.balance

    def transfer_history(C):
        a, b = C("Аня", 100), C("Боря", 100)
        a.transfer(b, 50)
        return a.history, b.history

    def transfer_too_much(C):
        a, b = C("Аня", 100), C("Боря", 100)
        a.transfer(b, 500)

    def unchanged_after_failed_transfer(C):
        a, b = C("Аня", 100), C("Боря", 100)
        try:
            a.transfer(b, 500)
        except Exception:
            pass
        return a.balance, b.balance, a.history, b.history

    def transfer_negative(C):
        a, b = C("Аня", 100), C("Боря", 100)
        a.transfer(b, -1)

    def length(C):
        acc = C("Аня")
        acc.deposit(10); acc.deposit(20); acc.withdraw(5)
        return len(acc)

    def length_new(C):
        return len(C("Аня", 100))

    def length_after_transfer(C):
        a, b = C("Аня", 100), C("Боря")
        a.transfer(b, 40)
        return len(a), len(b)

    return [
        Scenario("transfer(other, 50): балансы 100/100 → 50/150", transfer, (50, 150)),
        Scenario("перевод записывается как withdraw у отправителя и deposit у получателя",
                 transfer_history, ([("withdraw", 50)], [("deposit", 50)])),
        Scenario("перевод суммы больше баланса — InsufficientFundsError",
                 transfer_too_much, Raises("InsufficientFundsError")),
        Scenario("после неудачного перевода ничего не изменилось",
                 unchanged_after_failed_transfer, (100, 100, [], [])),
        Scenario("перевод отрицательной суммы — ValueError", transfer_negative, Raises(ValueError)),
        Scenario("len(acc) — число операций", length, 3),
        Scenario("len у нового счёта == 0", length_new, 0),
        Scenario("перевод добавляет по одной операции каждому счёту", length_after_transfer, (1, 1)),
    ]


_TASKS["account_step7"] = {
    "title": "BankAccount, шаг 7: переводы и __len__",
    "name": "BankAccount",
    "cases": _acc_step7_cases(),
}


# ===========================================================================
# ЧАСТЬ II. ООП — уровень 2: Bank
# ===========================================================================

def _bank_step1_cases():
    def name(Bank):
        return Bank("PyBank").name

    def accounts_empty(Bank):
        return Bank("PyBank").accounts

    def numbers(Bank):
        bank = Bank("PyBank")
        return bank.open_account("Аня"), bank.open_account("Боря", 100)

    def stored(Bank):
        bank = Bank("PyBank")
        bank.open_account("Аня", 30)
        acc = bank.accounts[1]
        return acc.owner, acc.balance

    def account_is_bank_account(Bank):
        bank = Bank("PyBank")
        bank.open_account("Аня")
        return hasattr(bank.accounts[1], "deposit")

    def separate_banks(Bank):
        b1, b2 = Bank("A"), Bank("B")
        b1.open_account("Аня")
        return len(b2.accounts)

    return [
        Scenario("Bank('PyBank').name == 'PyBank'", name, "PyBank"),
        Scenario("accounts у нового банка — пустой словарь", accounts_empty, {}),
        Scenario("open_account возвращает номера 1, 2, ...", numbers, (1, 2)),
        Scenario("счёт хранится в accounts под своим номером", stored, ("Аня", 30)),
        Scenario("в accounts лежит объект BankAccount (у него есть deposit)", account_is_bank_account, True),
        Scenario("у разных банков свои счета", separate_banks, 0),
    ]


_TASKS["bank_step1"] = {
    "title": "Bank, шаг 1: класс, словарь счетов и open_account",
    "name": "Bank",
    "cases": _bank_step1_cases(),
}


def _bank_fixture(Bank):
    bank = Bank("PyBank")
    bank.open_account("Аня", 300)
    bank.open_account("Боря", 100)
    bank.open_account("Вера", 200)
    return bank


def _bank_step2_cases():
    def get(Bank, E):
        return _bank_fixture(Bank).get_account(2).owner

    def get_missing(Bank, E):
        _bank_fixture(Bank).get_account(99)

    def close(Bank, E):
        bank = Bank("PyBank")
        bank.open_account("Аня")
        bank.close_account(1)
        return 1 in bank.accounts

    def close_with_money(Bank, E):
        bank = _bank_fixture(Bank)
        bank.close_account(1)

    def still_there(Bank, E):
        bank = _bank_fixture(Bank)
        try:
            bank.close_account(1)
        except Exception:
            pass
        return 1 in bank.accounts

    def close_missing(Bank, E):
        _bank_fixture(Bank).close_account(42)

    def numbers_not_reused(Bank, E):
        bank = Bank("PyBank")
        bank.open_account("Аня"); bank.open_account("Боря")
        bank.close_account(2)
        return bank.open_account("Вера")

    def hierarchy(Bank, E):
        return issubclass(E, Exception)

    return [
        Scenario("get_account(2).owner == 'Боря'", get, "Боря"),
        Scenario("get_account(99) бросает AccountNotFoundError", get_missing, Raises("AccountNotFoundError")),
        Scenario("close_account удаляет счёт с нулевым балансом", close, False),
        Scenario("нельзя закрыть счёт с деньгами — ValueError", close_with_money, Raises(ValueError)),
        Scenario("после неудачного закрытия счёт остаётся", still_there, True),
        Scenario("закрытие несуществующего счёта — AccountNotFoundError", close_missing, Raises("AccountNotFoundError")),
        Scenario("номера закрытых счетов не переиспользуются", numbers_not_reused, 3),
        Scenario("AccountNotFoundError — подкласс Exception", hierarchy, True),
    ]


_TASKS["bank_step2"] = {
    "title": "Bank, шаг 2: поиск, закрытие счёта и своё исключение",
    "name": "Bank",
    "objects": 2,
    "cases": _bank_step2_cases(),
}


def _bank_step3_cases():
    def length(Bank):
        return len(_bank_fixture(Bank)), len(Bank("X"))

    def contains(Bank):
        bank = _bank_fixture(Bank)
        return 2 in bank, 5 in bank

    def getitem(Bank):
        return _bank_fixture(Bank)[1].owner

    def getitem_missing(Bank):
        _bank_fixture(Bank)[7]

    def iteration(Bank):
        return [acc.owner for acc in _bank_fixture(Bank)]

    def to_str(Bank):
        return str(_bank_fixture(Bank))

    def after_close(Bank):
        bank = Bank("PyBank")
        bank.open_account("Аня"); bank.open_account("Боря"); bank.open_account("Вера")
        bank.close_account(2)
        return len(bank), [acc.owner for acc in bank], 2 in bank

    return [
        Scenario("len(bank) — число счетов", length, (3, 0)),
        Scenario("оператор in проверяет номер счёта", contains, (True, False)),
        Scenario("bank[1] — счёт с номером 1", getitem, "Аня"),
        Scenario("bank[7] — AccountNotFoundError", getitem_missing, Raises("AccountNotFoundError")),
        Scenario("for acc in bank перебирает счета по возрастанию номера", iteration, ["Аня", "Боря", "Вера"]),
        Scenario("str(bank) == 'Банк PyBank: счетов — 3'", to_str, "Банк PyBank: счетов — 3"),
        Scenario("len, перебор и in согласованы после закрытия счёта", after_close, (2, ["Аня", "Вера"], False)),
    ]


_TASKS["bank_step3"] = {
    "title": "Bank, шаг 3: магические методы контейнера",
    "name": "Bank",
    "cases": _bank_step3_cases(),
}


def _bank_step4_cases():
    def total(Bank):
        return _bank_fixture(Bank).total_balance

    def is_property(Bank):
        return isinstance(Bank.total_balance, property)

    def total_updates(Bank):
        bank = _bank_fixture(Bank)
        bank.get_account(2).deposit(50)
        return bank.total_balance

    def richest(Bank):
        return _bank_fixture(Bank).richest().owner

    def richest_empty(Bank):
        return Bank("X").richest()

    def find(Bank):
        return [acc.owner for acc in _bank_fixture(Bank).find(lambda acc: acc.balance > 150)]

    def find_none(Bank):
        return _bank_fixture(Bank).find(lambda acc: acc.balance > 1000)

    def find_by_owner(Bank):
        return [acc.balance for acc in _bank_fixture(Bank).find(lambda acc: acc.owner.startswith("В"))]

    def sorted_asc(Bank):
        return [acc.owner for acc in _bank_fixture(Bank).sorted_by_balance()]

    def sorted_desc(Bank):
        return [acc.owner for acc in _bank_fixture(Bank).sorted_by_balance(descending=True)]

    return [
        Scenario("total_balance == 600 для счетов 300/100/200", total, 600),
        Scenario("total_balance — свойство (property), а не метод", is_property, True),
        Scenario("total_balance пересчитывается после пополнения", total_updates, 650),
        Scenario("richest() — счёт с наибольшим балансом", richest, "Аня"),
        Scenario("richest() пустого банка — None", richest_empty, None),
        Scenario("find(lambda acc: acc.balance > 150) → Аня, Вера", find, ["Аня", "Вера"]),
        Scenario("find без совпадений — пустой список", find_none, []),
        Scenario("find по владельцу", find_by_owner, [200]),
        Scenario("sorted_by_balance() — по возрастанию", sorted_asc, ["Боря", "Вера", "Аня"]),
        Scenario("sorted_by_balance(descending=True) — по убыванию", sorted_desc, ["Аня", "Вера", "Боря"]),
    ]


_TASKS["bank_step4"] = {
    "title": "Bank, шаг 4: свойство total_balance, поиск и сортировка",
    "name": "Bank",
    "cases": _bank_step4_cases(),
}


def _bank_step5_cases():
    def deposit(Bank):
        bank = _bank_fixture(Bank)
        returned = bank.deposit(2, 50)
        return returned, bank[2].balance

    def withdraw(Bank):
        bank = _bank_fixture(Bank)
        returned = bank.withdraw(1, 100)
        return returned, bank[1].balance

    def transfer(Bank):
        bank = _bank_fixture(Bank)
        bank.transfer(1, 2, 100)
        return bank[1].balance, bank[2].balance, bank.total_balance

    def transfer_missing(Bank):
        _bank_fixture(Bank).transfer(1, 9, 10)

    def transfer_insufficient(Bank):
        _bank_fixture(Bank).transfer(2, 1, 500)

    def unchanged(Bank):
        bank = _bank_fixture(Bank)
        try:
            bank.transfer(2, 1, 500)
        except Exception:
            pass
        return bank[1].balance, bank[2].balance

    def deposit_negative(Bank):
        _bank_fixture(Bank).deposit(1, -5)

    def deposit_missing(Bank):
        _bank_fixture(Bank).deposit(9, 5)

    def history_through_bank(Bank):
        bank = _bank_fixture(Bank)
        bank.transfer(1, 3, 25)
        return bank[1].history, bank[3].history

    return [
        Scenario("deposit(2, 50) возвращает 150 и меняет счёт №2", deposit, (150, 150)),
        Scenario("withdraw(1, 100) возвращает 200", withdraw, (200, 200)),
        Scenario("transfer(1, 2, 100): 300/100 → 200/200, общая сумма прежняя", transfer, (200, 200, 600)),
        Scenario("перевод на несуществующий счёт — AccountNotFoundError",
                 transfer_missing, Raises("AccountNotFoundError")),
        Scenario("перевод без средств — InsufficientFundsError", transfer_insufficient, Raises("InsufficientFundsError")),
        Scenario("после неудачного перевода балансы прежние", unchanged, (300, 100)),
        Scenario("deposit(1, -5) — ValueError (проверка из BankAccount)", deposit_negative, Raises(ValueError)),
        Scenario("deposit на несуществующий счёт — AccountNotFoundError", deposit_missing, Raises("AccountNotFoundError")),
        Scenario("перевод через банк попадает в history обоих счетов",
                 history_through_bank, ([("withdraw", 25)], [("deposit", 25)])),
    ]


_TASKS["bank_step5"] = {
    "title": "Bank, шаг 5: операции по номеру счёта",
    "name": "Bank",
    "cases": _bank_step5_cases(),
}

# ===========================================================================
# ЧАСТЬ II. ООП — уровень 3: наследование, полиморфизм, отчёты, программа
# ===========================================================================

def _l3_step1_cases():
    def subclasses(B, S, Cr):
        return issubclass(S, B), issubclass(Cr, B)

    def interest(B, S, Cr):
        acc = S("Аня", 1000)
        returned = acc.add_interest()
        return returned, acc.balance

    def interest_history(B, S, Cr):
        acc = S("Аня", 1000)
        acc.add_interest()
        return acc.history

    def custom_rate(B, S, Cr):
        acc = S("Аня", 200, rate=0.1)
        acc.add_interest()
        return acc.balance

    def zero_balance(B, S, Cr):
        acc = S("Аня")
        return acc.add_interest(), acc.history

    def savings_month_end(B, S, Cr):
        acc = S("Аня", 100)
        return acc.month_end(), acc.balance

    def base_month_end(B, S, Cr):
        acc = B("Аня", 100)
        return acc.month_end(), acc.balance, acc.history

    def credit_withdraw(B, S, Cr):
        acc = Cr("Боря", 100, credit_limit=500)
        acc.withdraw(400)
        return acc.balance

    def credit_available(B, S, Cr):
        return Cr("Боря", 100, credit_limit=500).available()

    def credit_too_much(B, S, Cr):
        Cr("Боря", 100, credit_limit=500).withdraw(601)

    def credit_fee(B, S, Cr):
        acc = Cr("Боря", 0, credit_limit=1000)
        acc.withdraw(300)
        fee = acc.month_end()
        return fee, acc.balance, acc.history[-1]

    def credit_no_fee(B, S, Cr):
        acc = Cr("Боря", 50)
        return acc.month_end(), acc.balance

    def repr_uses_subclass_name(B, S, Cr):
        return repr(S("Аня", 10)), repr(Cr("Боря", 5))

    def inherited(B, S, Cr):
        acc = S("Аня", 10)
        acc.deposit(5)
        return str(acc), len(acc)

    def default_credit_limit(B, S, Cr):
        return Cr("Боря").credit_limit

    return [
        Scenario("SavingsAccount и CreditAccount — подклассы BankAccount", subclasses, (True, True)),
        Scenario("add_interest() при 1000 и ставке 0.05 возвращает 50.0, баланс 1050.0", interest, (50.0, 1050.0)),
        Scenario("проценты записываются в history как deposit", interest_history, [("deposit", 50.0)]),
        Scenario("ставка задаётся при создании: rate=0.1", custom_rate, 220.0),
        Scenario("при нулевом балансе проценты 0 и history пуста", zero_balance, (0, [])),
        Scenario("SavingsAccount.month_end() начисляет проценты", savings_month_end, (5.0, 105.0)),
        Scenario("BankAccount.month_end() ничего не делает и возвращает 0", base_month_end, (0, 100, [])),
        Scenario("CreditAccount: withdraw(400) при балансе 100 и лимите 500 → -300", credit_withdraw, -300),
        Scenario("available() кредитного счёта = баланс + лимит", credit_available, 600),
        Scenario("снятие сверх лимита — InsufficientFundsError", credit_too_much, Raises("InsufficientFundsError")),
        Scenario("month_end() при долге 300 списывает комиссию 6.0 и пишет ('fee', 6.0)",
                 credit_fee, (6.0, -306.0, ("fee", 6.0))),
        Scenario("без долга комиссии нет", credit_no_fee, (0, 50)),
        Scenario("repr показывает имя подкласса", repr_uses_subclass_name,
                 ("SavingsAccount('Аня', 10)", "CreditAccount('Боря', 5)")),
        Scenario("унаследованные deposit, __str__ и __len__ работают", inherited, ("Аня: 15 RUB", 1)),
        Scenario("credit_limit по умолчанию 1000", default_credit_limit, 1000),
    ]


_TASKS["level3_step1"] = {
    "title": "Уровень 3, шаг 1: SavingsAccount и CreditAccount",
    "name": "BankAccount",
    "objects": 3,
    "checks": {1: {"require_super": True}, 2: {"require_super": True}},
    "cases": _l3_step1_cases(),
}


def _mixed_bank(Bank, S, Cr):
    bank = Bank("PyBank")
    bank.open_account("Аня", 100)
    bank.add_account(S("Боря", 1000))
    credit = Cr("Вера", 0, credit_limit=1000)
    credit.withdraw(300)
    bank.add_account(credit)
    return bank


def _l3_step2_cases():
    def add_returns_number(Bank, S, Cr):
        bank = Bank("PyBank")
        bank.open_account("Аня")
        return bank.add_account(S("Боря", 10)), bank.add_account(Cr("Вера"))

    def added_is_same_object(Bank, S, Cr):
        bank = Bank("PyBank")
        acc = S("Боря", 10)
        number = bank.add_account(acc)
        return bank[number] is acc

    def add_wrong_type(Bank, S, Cr):
        Bank("PyBank").add_account("не счёт")

    def month_end_polymorphic(Bank, S, Cr):
        bank = _mixed_bank(Bank, S, Cr)
        results = bank.month_end()
        return results, [acc.balance for acc in bank]

    def month_end_total(Bank, S, Cr):
        bank = _mixed_bank(Bank, S, Cr)
        bank.month_end()
        return bank.total_balance

    def mixed_iteration(Bank, S, Cr):
        bank = _mixed_bank(Bank, S, Cr)
        return [type(acc).__name__ for acc in bank]

    def sorted_mixed(Bank, S, Cr):
        bank = _mixed_bank(Bank, S, Cr)
        return [acc.owner for acc in bank.sorted_by_balance()]

    return [
        Scenario("add_account возвращает следующий номер", add_returns_number, (2, 3)),
        Scenario("add_account кладёт в банк тот же объект", added_is_same_object, True),
        Scenario("add_account('не счёт') — TypeError", add_wrong_type, Raises(TypeError)),
        Scenario("month_end() вызывает month_end у каждого счёта: [0, 50.0, 6.0] и балансы [100, 1050.0, -306.0]",
                 month_end_polymorphic, ([0, 50.0, 6.0], [100, 1050.0, -306.0])),
        Scenario("total_balance после month_end == 844.0", month_end_total, 844.0),
        Scenario("в одном банке лежат объекты разных классов", mixed_iteration,
                 ["BankAccount", "SavingsAccount", "CreditAccount"]),
        Scenario("сортировка работает для смешанных счетов", sorted_mixed, ["Вера", "Аня", "Боря"]),
    ]


_TASKS["level3_step2"] = {
    "title": "Уровень 3, шаг 2: Bank принимает любые счета и закрывает месяц",
    "name": "Bank",
    "objects": 3,
    "cases": _l3_step2_cases(),
}


def _report_bank(Bank):
    bank = Bank("PyBank")
    bank.open_account("Аня", 300)
    bank.open_account("Боря", 100)
    bank.open_account("Вера", 200)
    bank.open_account("Гоша", 200)
    bank.deposit(2, 50)
    bank.withdraw(1, 100)
    bank.transfer(3, 2, 20)
    return bank


def _l3_step3_cases():
    def is_gen(iter_ops, top, summary, Bank):
        return _is_generator(iter_ops(_report_bank(Bank)))

    def sequence(iter_ops, top, summary, Bank):
        return list(iter_ops(_report_bank(Bank)))

    def empty(iter_ops, top, summary, Bank):
        return list(iter_ops(Bank("X")))

    def lazy(iter_ops, top, summary, Bank):
        return next(iter_ops(_report_bank(Bank)))

    def top_default(iter_ops, top, summary, Bank):
        return top(_report_bank(Bank))

    def top_one(iter_ops, top, summary, Bank):
        return top(_report_bank(Bank), n=1)

    def top_more_than_accounts(iter_ops, top, summary, Bank):
        return top(_report_bank(Bank), 10)

    def top_empty(iter_ops, top, summary, Bank):
        return top(Bank("X"))

    def summary_dict(iter_ops, top, summary, Bank):
        return summary(_report_bank(Bank))

    def summary_empty(iter_ops, top, summary, Bank):
        return summary(Bank("X"))

    return [
        Scenario("iter_operations(bank) — генератор", is_gen, True),
        Scenario("операции идут по счетам (по номеру) и по порядку внутри счёта", sequence,
                 [(1, "withdraw", 100), (2, "deposit", 50), (2, "deposit", 20), (3, "withdraw", 20)]),
        Scenario("для пустого банка — ничего", empty, []),
        Scenario("генератор ленивый: next даёт первую операцию", lazy, (1, "withdraw", 100)),
        Scenario("top_accounts(bank) — 3 владельца по убыванию баланса, при равенстве — меньший номер",
                 top_default, ["Аня", "Гоша", "Вера"]),
        Scenario("top_accounts(bank, n=1)", top_one, ["Аня"]),
        Scenario("n больше числа счетов — все владельцы", top_more_than_accounts, ["Аня", "Гоша", "Вера", "Боря"]),
        Scenario("top_accounts пустого банка — []", top_empty, []),
        Scenario("operations_summary(bank) == {'withdraw': 120, 'deposit': 70}", summary_dict,
                 {"withdraw": 120, "deposit": 70}),
        Scenario("operations_summary пустого банка — {}", summary_empty, {}),
    ]


_TASKS["level3_step3"] = {
    "title": "Уровень 3, шаг 3: генератор операций и отчёты",
    "name": "iter_operations",
    "objects": 4,
    "checks": {0: {"require_yield": True}, 1: {"require_lambda": True}},
    "cases": _l3_step3_cases(),
}


def _l3_step4_cases():
    def decorator_passes_result(catch, run, Bank):
        return catch(lambda x: x * 2)(21)

    def decorator_catches(catch, run, Bank):
        return catch(lambda: 1 / 0)().startswith("ошибка:")

    def decorator_message(catch, run, Bank):
        def fail():
            raise ValueError("нет денег")
        return catch(fail)()

    def decorator_name(catch, run, Bank):
        def handler():
            return 1
        return catch(handler).__name__

    def commands(catch, run, Bank):
        bank = Bank("PyBank")
        return run(bank, [
            ("open", "Аня", 100),
            ("open", "Боря"),
            ("deposit", 2, 50),
            ("transfer", 1, 2, 30),
            ("balance", 1),
            ("balance", 2),
        ])

    def state_after(catch, run, Bank):
        bank = Bank("PyBank")
        run(bank, [("open", "Аня", 100), ("open", "Боря"), ("transfer", 1, 2, 30)])
        return bank[1].balance, bank[2].balance

    def errors_do_not_stop(catch, run, Bank):
        bank = Bank("PyBank")
        results = run(bank, [
            ("open", "Аня", 100),
            ("withdraw", 1, 500),
            ("deposit", 7, 10),
            ("deposit", 1, -1),
            ("balance", 1),
        ])
        return [r.startswith("ошибка:") for r in results], results[-1]

    def unknown_command(catch, run, Bank):
        results = run(Bank("PyBank"), [("fly", 1)])
        return len(results), results[0].startswith("ошибка:")

    def empty_commands(catch, run, Bank):
        return run(Bank("PyBank"), [])

    def all_strings(catch, run, Bank):
        results = run(Bank("PyBank"), [("open", "Аня"), ("balance", 1), ("zzz",)])
        return all(isinstance(r, str) for r in results)

    return [
        Scenario("catch_errors(lambda x: x * 2)(21) == 42", decorator_passes_result, 42),
        Scenario("ошибка внутри превращается в строку 'ошибка: ...'", decorator_catches, True),
        Scenario("текст исключения попадает в строку", decorator_message, "ошибка: нет денег"),
        Scenario("__name__ обёрнутой функции сохранён", decorator_name, "handler"),
        Scenario("run_commands выполняет команды и возвращает список строк", commands,
                 ["1", "2", "50", "None", "70", "80"]),
        Scenario("команды меняют состояние банка", state_after, (70, 30)),
        Scenario("ошибочные команды не прерывают выполнение", errors_do_not_stop,
                 ([False, True, True, True, False], "100")),
        Scenario("неизвестная команда — строка 'ошибка: ...'", unknown_command, (1, True)),
        Scenario("пустой список команд — пустой результат", empty_commands, []),
        Scenario("все результаты — строки", all_strings, True),
    ]


_TASKS["level3_step4"] = {
    "title": "Уровень 3, шаг 4: декоратор catch_errors и run_commands",
    "name": "catch_errors",
    "objects": 3,
    "checks": {0: {"require_names": ("wraps",)}},
    "cases": _l3_step4_cases(),
}


# ===========================================================================
# ЧАСТЬ II. ООП — итоговый проект: библиотека
# ===========================================================================

def _project_exceptions_cases():
    def hierarchy(LE, NF, NA, RL):
        return issubclass(LE, Exception), issubclass(NF, LE), issubclass(NA, LE), issubclass(RL, LE)

    def distinct(LE, NF, NA, RL):
        return issubclass(NF, NA), issubclass(NA, RL)

    def message(LE, NF, NA, RL):
        return str(NF("книга не найдена"))

    def catch_as_base(LE, NF, NA, RL):
        try:
            raise RL("лимит")
        except LE as e:
            return str(e)

    return [
        Scenario("все исключения — подклассы LibraryError, а он — Exception", hierarchy, (True, True, True, True)),
        Scenario("исключения не наследуются друг от друга", distinct, (False, False)),
        Scenario("str(BookNotFoundError('книга не найдена'))", message, "книга не найдена"),
        Scenario("ReaderLimitError ловится как LibraryError", catch_as_base, "лимит"),
    ]


_TASKS["project_exceptions"] = {
    "title": "Проект, компонент 1: исключения библиотеки",
    "name": "LibraryError",
    "objects": 4,
    "cases": _project_exceptions_cases(),
}


def _project_book_cases():
    def attrs(Book):
        b = Book("Идиот", "Достоевский", 1869)
        return b.title, b.author, b.year, b.available

    def to_str(Book):
        return str(Book("Идиот", "Достоевский", 1869))

    def to_repr(Book):
        return repr(Book("Идиот", "Достоевский", 1869))

    def eq(Book):
        return (Book("Идиот", "Достоевский", 1869) == Book("Идиот", "Достоевский", 1869),
                Book("Идиот", "Достоевский", 1869) == Book("Идиот", "Достоевский", 1870))

    def eq_other_type(Book):
        return Book("Идиот", "Достоевский", 1869) == "Идиот"

    def sorting(Book):
        books = [Book("Идиот", "Достоевский", 1869), Book("Собачье сердце", "Булгаков", 1925),
                 Book("Бесы", "Достоевский", 1872), Book("Мастер и Маргарита", "Булгаков", 1967)]
        return [b.title for b in sorted(books)]

    def empty_title(Book):
        Book("", "Автор", 2000)

    def empty_author(Book):
        Book("Название", "   ", 2000)

    def bad_year(Book):
        Book("Название", "Автор", 100)

    def year_string(Book):
        Book("Название", "Автор", "1999")

    def boundary_years(Book):
        return Book("A", "B", 1450).year, Book("A", "B", 2100).year

    return [
        Scenario("Book хранит title, author, year и available == True", attrs, ("Идиот", "Достоевский", 1869, True)),
        Scenario("str(book) == 'Идиот — Достоевский (1869)'", to_str, "Идиот — Достоевский (1869)"),
        Scenario("repr(book) == \"Book('Идиот', 'Достоевский', 1869)\"", to_repr, "Book('Идиот', 'Достоевский', 1869)"),
        Scenario("книги равны при одинаковых title, author, year", eq, (True, False)),
        Scenario("сравнение со строкой даёт False", eq_other_type, False),
        Scenario("sorted сортирует по (author, title)", sorting,
                 ["Мастер и Маргарита", "Собачье сердце", "Бесы", "Идиот"]),
        Scenario("пустое название — ValueError", empty_title, Raises(ValueError)),
        Scenario("автор из пробелов — ValueError", empty_author, Raises(ValueError)),
        Scenario("год 100 — ValueError", bad_year, Raises(ValueError)),
        Scenario("год строкой — ValueError", year_string, Raises(ValueError)),
        Scenario("границы 1450 и 2100 допустимы", boundary_years, (1450, 2100)),
    ]


_TASKS["project_book"] = {
    "title": "Проект, компонент 2: класс Book",
    "name": "Book",
    "cases": _project_book_cases(),
}


def _project_reader_cases():
    def attrs(Reader, Book):
        r = Reader("Аня")
        return r.name, r.max_books, r.books

    def custom_limit(Reader, Book):
        return Reader("Аня", max_books=1).max_books

    def can_borrow(Reader, Book):
        r = Reader("Аня", max_books=2)
        first = r.can_borrow
        r.books.append(Book("A", "B", 2000))
        second = r.can_borrow
        r.books.append(Book("C", "D", 2001))
        return first, second, r.can_borrow

    def is_property(Reader, Book):
        return isinstance(Reader.can_borrow, property)

    def length(Reader, Book):
        r = Reader("Аня")
        r.books.append(Book("A", "B", 2000))
        return len(r)

    def to_str(Reader, Book):
        r = Reader("Аня")
        r.books.append(Book("A", "B", 2000))
        return str(r)

    def empty_name(Reader, Book):
        Reader("")

    def bad_limit(Reader, Book):
        Reader("Аня", max_books=0)

    def separate_lists(Reader, Book):
        return Reader("Аня").books is Reader("Боря").books

    return [
        Scenario("Reader('Аня'): name, max_books == 3, books == []", attrs, ("Аня", 3, [])),
        Scenario("max_books задаётся при создании", custom_limit, 1),
        Scenario("can_borrow становится False при достижении лимита", can_borrow, (True, True, False)),
        Scenario("can_borrow — свойство (property)", is_property, True),
        Scenario("len(reader) — число книг на руках", length, 1),
        Scenario("str(reader) == 'Читатель Аня: книг — 1'", to_str, "Читатель Аня: книг — 1"),
        Scenario("пустое имя — ValueError", empty_name, Raises(ValueError)),
        Scenario("max_books < 1 — ValueError", bad_limit, Raises(ValueError)),
        Scenario("у разных читателей разные списки книг", separate_lists, False),
    ]


_TASKS["project_reader"] = {
    "title": "Проект, компонент 3: класс Reader",
    "name": "Reader",
    "objects": 2,
    "cases": _project_reader_cases(),
}


def _library_fixture(Library, Book, Reader):
    lib = Library("Городская")
    lib.add_books(Book("Мастер и Маргарита", "Булгаков", 1967),
                  Book("Собачье сердце", "Булгаков", 1925),
                  Book("Идиот", "Достоевский", 1869),
                  Book("Евгений Онегин", "Пушкин", 1833))
    lib.register(Reader("Аня", max_books=2))
    lib.register(Reader("Боря"))
    return lib


def _project_library_cases():
    def basics(L, B, R):
        lib = _library_fixture(L, B, R)
        return lib.name, len(lib), sorted(lib.readers)

    def contains(L, B, R):
        lib = _library_fixture(L, B, R)
        return "Идиот" in lib, "Война и мир" in lib

    def iteration(L, B, R):
        return [b.title for b in _library_fixture(L, B, R)]

    def add_books_varargs(L, B, R):
        lib = L("X")
        lib.add_books(B("A", "B", 2000))
        lib.add_books(B("C", "D", 2001), B("E", "F", 2002))
        return len(lib)

    def find(L, B, R):
        return [b.title for b in _library_fixture(L, B, R).find(lambda b: b.year < 1900)]

    def find_by_title(L, B, R):
        return _library_fixture(L, B, R).find_by_title("Идиот").author

    def find_missing(L, B, R):
        _library_fixture(L, B, R).find_by_title("Нет такой")

    def lend(L, B, R):
        lib = _library_fixture(L, B, R)
        book = lib.lend("Идиот", "Аня")
        return book.title, book.available, [b.title for b in lib.readers["Аня"].books]

    def lend_taken(L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня")
        lib.lend("Идиот", "Боря")

    def lend_unknown_reader(L, B, R):
        _library_fixture(L, B, R).lend("Идиот", "Никто")

    def lend_missing_book(L, B, R):
        _library_fixture(L, B, R).lend("Нет такой", "Аня")

    def limit(L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня")
        lib.lend("Собачье сердце", "Аня")
        lib.lend("Евгений Онегин", "Аня")

    def state_after_limit_error(L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня")
        lib.lend("Собачье сердце", "Аня")
        try:
            lib.lend("Евгений Онегин", "Аня")
        except Exception:
            pass
        return len(lib.readers["Аня"]), lib.find_by_title("Евгений Онегин").available

    def return_book(L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня")
        lib.return_book("Идиот", "Аня")
        return lib.find_by_title("Идиот").available, len(lib.readers["Аня"])

    def return_not_taken(L, B, R):
        _library_fixture(L, B, R).return_book("Идиот", "Аня")

    def available_generator(L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня")
        gen = lib.available()
        return _is_generator(gen), [b.title for b in gen]

    def duplicate_reader(L, B, R):
        _library_fixture(L, B, R).register(R("Аня"))

    def lend_after_return(L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня")
        lib.return_book("Идиот", "Аня")
        return lib.lend("Идиот", "Боря").title

    return [
        Scenario("name, len(lib) == 4 и словарь читателей", basics, ("Городская", 4, ["Аня", "Боря"])),
        Scenario("оператор in проверяет название", contains, (True, False)),
        Scenario("for book in lib перебирает книги по порядку добавления", iteration,
                 ["Мастер и Маргарита", "Собачье сердце", "Идиот", "Евгений Онегин"]),
        Scenario("add_books принимает любое число книг (*books)", add_books_varargs, 3),
        Scenario("find(lambda b: b.year < 1900) → Идиот, Евгений Онегин", find, ["Идиот", "Евгений Онегин"]),
        Scenario("find_by_title('Идиот').author == 'Достоевский'", find_by_title, "Достоевский"),
        Scenario("find_by_title неизвестной книги — BookNotFoundError", find_missing, Raises("BookNotFoundError")),
        Scenario("lend возвращает книгу, помечает её занятой и добавляет читателю", lend,
                 ("Идиот", False, ["Идиот"])),
        Scenario("выдать занятую книгу — BookNotAvailableError", lend_taken, Raises("BookNotAvailableError")),
        Scenario("выдать незарегистрированному читателю — LibraryError", lend_unknown_reader, Raises("LibraryError")),
        Scenario("выдать несуществующую книгу — BookNotFoundError", lend_missing_book, Raises("BookNotFoundError")),
        Scenario("третья книга при лимите 2 — ReaderLimitError", limit, Raises("ReaderLimitError")),
        Scenario("после ошибки лимита состояние не изменилось", state_after_limit_error, (2, True)),
        Scenario("return_book возвращает книгу в библиотеку", return_book, (True, 0)),
        Scenario("вернуть книгу, которой нет у читателя — LibraryError", return_not_taken, Raises("LibraryError")),
        Scenario("available() — генератор свободных книг", available_generator,
                 (True, ["Мастер и Маргарита", "Собачье сердце", "Евгений Онегин"])),
        Scenario("повторная регистрация имени — ValueError", duplicate_reader, Raises(ValueError)),
        Scenario("после возврата книгу можно выдать другому", lend_after_return, "Идиот"),
    ]


_TASKS["project_library"] = {
    "title": "Проект, компонент 4: класс Library",
    "name": "Library",
    "objects": 3,
    "checks": {0: {"require_yield_in_source": True}},
    "cases": _project_library_cases(),
}


def _project_reports_cases():
    def by_author(by_author_fn, top, run, L, B, R):
        return by_author_fn(_library_fixture(L, B, R))

    def by_author_empty(by_author_fn, top, run, L, B, R):
        return by_author_fn(L("X"))

    def top_readers(by_author_fn, top, run, L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня"); lib.lend("Собачье сердце", "Аня"); lib.lend("Евгений Онегин", "Боря")
        return top(lib)

    def top_n(by_author_fn, top, run, L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Аня"); lib.lend("Собачье сердце", "Аня"); lib.lend("Евгений Онегин", "Боря")
        return top(lib, n=1)

    def top_tie(by_author_fn, top, run, L, B, R):
        lib = _library_fixture(L, B, R)
        lib.lend("Идиот", "Боря"); lib.lend("Собачье сердце", "Аня")
        return top(lib)

    def top_nobody(by_author_fn, top, run, L, B, R):
        return top(_library_fixture(L, B, R))

    def events(by_author_fn, top, run, L, B, R):
        lib = L("Городская")
        return run(lib, [
            ("add", {"title": "Идиот", "author": "Достоевский", "year": 1869}),
            ("register", "Аня"),
            ("lend", "Идиот", "Аня"),
            ("return", "Идиот", "Аня"),
        ])

    def events_state(by_author_fn, top, run, L, B, R):
        lib = L("Городская")
        run(lib, [
            ("add", {"title": "Идиот", "author": "Достоевский", "year": 1869}),
            ("add", {"title": "Бесы", "author": "Достоевский", "year": 1872}),
            ("register", "Аня"),
            ("lend", "Идиот", "Аня"),
        ])
        return len(lib), len(lib.readers["Аня"]), lib.find_by_title("Идиот").available

    def events_errors(by_author_fn, top, run, L, B, R):
        lib = L("Городская")
        results = run(lib, [
            ("add", {"title": "Идиот", "author": "Достоевский", "year": 1869}),
            ("lend", "Идиот", "Аня"),
            ("register", "Аня"),
            ("lend", "Нет такой", "Аня"),
            ("lend", "Идиот", "Аня"),
            ("lend", "Идиот", "Аня"),
            ("add", {"title": "", "author": "X", "year": 2000}),
            ("dance",),
        ])
        return [r.startswith("ошибка:") for r in results]

    def events_empty(by_author_fn, top, run, L, B, R):
        return run(L("X"), [])

    return [
        Scenario("books_by_author группирует названия по авторам в порядке добавления", by_author,
                 {"Булгаков": ["Мастер и Маргарита", "Собачье сердце"],
                  "Достоевский": ["Идиот"], "Пушкин": ["Евгений Онегин"]}),
        Scenario("books_by_author пустой библиотеки — {}", by_author_empty, {}),
        Scenario("top_readers: (имя, число книг) по убыванию", top_readers, [("Аня", 2), ("Боря", 1)]),
        Scenario("top_readers(lib, n=1)", top_n, [("Аня", 2)]),
        Scenario("при равном числе книг — по алфавиту", top_tie, [("Аня", 1), ("Боря", 1)]),
        Scenario("читатели без книг тоже в списке", top_nobody, [("Аня", 0), ("Боря", 0)]),
        Scenario("run_events выполняет события и возвращает строки", events,
                 ["добавлена: Идиот", "зарегистрирован: Аня", "выдана: Идиот → Аня", "возвращена: Идиот"]),
        Scenario("события меняют библиотеку", events_state, (2, 1, False)),
        Scenario("ошибки превращаются в строки 'ошибка: ...' и не прерывают обработку", events_errors,
                 [False, True, False, True, False, True, True, True]),
        Scenario("пустой список событий — []", events_empty, []),
    ]


_TASKS["project_reports"] = {
    "title": "Проект, компонент 5: отчёты и сценарий событий",
    "name": "books_by_author",
    "objects": 6,
    "checks": {1: {"require_lambda": True}, 2: {"require_try": True}},
    "cases": _project_reports_cases(),
}


# ---------------------------------------------------------------------------
# Публичные функции проверки
# ---------------------------------------------------------------------------

def _make_test(key):
    def test(*objs):
        _check(key, *objs)
    test.__name__ = f"test_{key}"
    test.__doc__ = _TASKS[key]["title"]
    return test


# Часть I: мини-задачи и 10 заданий
test_rectangle_area = _make_test("rectangle_area")
test_make_counter = _make_test("make_counter")
test_add_bonus = _make_test("add_bonus")
test_join_words = _make_test("join_words")
test_compose = _make_test("compose")
test_sort_by_length = _make_test("sort_by_length")
test_sum_digits = _make_test("sum_digits")
test_double_result = _make_test("double_result")
test_evens_up_to = _make_test("evens_up_to")
test_clamp = _make_test("clamp")
test_snake_to_camel = _make_test("snake_to_camel")
test_chunk = _make_test("chunk")
test_make_email = _make_test("make_email")
test_average = _make_test("average")
test_make_tag = _make_test("make_tag")
test_apply_pipeline = _make_test("apply_pipeline")
test_even_squares_desc = _make_test("even_squares_desc")
test_flatten = _make_test("flatten")
test_memoize_fib = _make_test("memoize_fib")

# Часть II: мини-задачи, уровни и проект
test_point = _make_test("point")
test_playlist = _make_test("playlist")
test_vector = _make_test("vector")
test_shapes = _make_test("shapes")
test_safe_sqrt = _make_test("safe_sqrt")
test_read_numbers = _make_test("read_numbers")
test_shapes_module = _make_test("shapes_module")
test_account_step1 = _make_test("account_step1")
test_account_step2 = _make_test("account_step2")
test_account_step3 = _make_test("account_step3")
test_account_step4 = _make_test("account_step4")
test_account_step5 = _make_test("account_step5")
test_account_step6 = _make_test("account_step6")
test_account_step7 = _make_test("account_step7")
test_bank_step1 = _make_test("bank_step1")
test_bank_step2 = _make_test("bank_step2")
test_bank_step3 = _make_test("bank_step3")
test_bank_step4 = _make_test("bank_step4")
test_bank_step5 = _make_test("bank_step5")
test_level3_step1 = _make_test("level3_step1")
test_level3_step2 = _make_test("level3_step2")
test_level3_step3 = _make_test("level3_step3")
test_level3_step4 = _make_test("level3_step4")
test_project_exceptions = _make_test("project_exceptions")
test_project_book = _make_test("project_book")
test_project_reader = _make_test("project_reader")
test_project_library = _make_test("project_library")
test_project_reports = _make_test("project_reports")

__all__ = ["Call", "Scenario", "Raises", "FLOAT_TOLERANCE"] + [n for n in dir() if n.startswith("test_")]
